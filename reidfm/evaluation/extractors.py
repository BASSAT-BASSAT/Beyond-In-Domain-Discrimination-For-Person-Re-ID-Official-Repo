"""Feature extractors used by the evaluator (paths in, ``[N, D]`` float32 features out).

* :class:`FastReIDExtractor`: any FastReID model. Preprocessing matches the benchmark's
  ``FastReIDEncoder`` (cv2 RGB, bicubic resize to ``INPUT.SIZE_TEST``, pixels 0-255; the
  model normalizes internally).
* :class:`OSNetExtractor`: native torchreid OSNet-AIN head (512-d ``fc`` output), matching
  the benchmark's ``OSNetAINEncoder`` for zero-shot numbers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Extractor(Protocol):
    def extract(self, paths: Sequence[Path], desc: str | None = None) -> np.ndarray: ...


class _ImageFolder:
    """Picklable dataset for DataLoader workers (Windows spawn-safe)."""

    def __init__(self, paths, size_hw, interpolation: str, normalize: bool):
        self.paths = [str(p) for p in paths]
        self.size_hw = size_hw
        self.interpolation = interpolation
        self.normalize = normalize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import cv2
        import torch

        image = cv2.imread(self.paths[index])
        if image is None:
            raise FileNotFoundError(self.paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        flag = cv2.INTER_CUBIC if self.interpolation == "cubic" else cv2.INTER_LINEAR
        image = cv2.resize(image, (self.size_hw[1], self.size_hw[0]), interpolation=flag).astype(np.float32)
        if self.normalize:
            image = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def default_workers() -> int:
    # Each worker is a full Python + torch process on Windows (spawn); 2 keeps a laptop GPU fed.
    return 2 if os.name == "nt" else 4


def _run(forward, paths, size_hw, interpolation, normalize, device, batch_size, workers, amp, desc=None) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    if not paths:
        return np.zeros((0, 0), dtype=np.float32)
    loader = DataLoader(
        _ImageFolder(paths, size_hw, interpolation, normalize),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda" and os.name != "nt",  # pinning fails intermittently on Windows WDDM
        persistent_workers=False,
    )
    chunks = []
    use_amp = amp and device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc or "extract", unit="batch", dynamic_ncols=True, leave=False):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                feats = forward(batch)
            chunks.append(feats.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


class FastReIDExtractor:
    def __init__(self, model, cfg, device=None, batch_size: int = 128, workers: int | None = None, amp: bool = True):
        import torch

        self.model = model
        self.size_hw = tuple(cfg.INPUT.SIZE_TEST)
        self.device = device or next(model.parameters()).device
        self.device = torch.device(self.device)
        self.batch_size = batch_size
        self.workers = default_workers() if workers is None else workers
        self.amp = amp

    def extract(self, paths: Sequence[Path], desc: str | None = None) -> np.ndarray:
        from fastreid.evaluation.evaluator import inference_context

        with inference_context(self.model):
            return _run(
                lambda x: self.model({"images": x}),
                list(paths), self.size_hw, "cubic", False, self.device, self.batch_size, self.workers, self.amp, desc,
            )


class OSNetExtractor:
    def __init__(self, weights: Path, device: str = "cuda", batch_size: int = 128, workers: int | None = None,
                 amp: bool = True):
        import torch

        from reidfm.models.backbones.osnet_ain import osnet_ain_x1_0

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        model = osnet_ain_x1_0(num_classes=1, pretrained=False)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        missing = [k for k in model.load_state_dict(state, strict=False).missing_keys if not k.startswith("classifier.")]
        if missing:
            raise RuntimeError(f"OSNet checkpoint missing {missing[:5]}")
        self.model = model.eval().to(self.device)
        self.batch_size, self.amp = batch_size, amp
        self.workers = default_workers() if workers is None else workers

    def extract(self, paths: Sequence[Path], desc: str | None = None) -> np.ndarray:
        import torch.nn.functional as F

        # The benchmark's OSNet encoder L2-normalizes per frame (FastReID features are not).
        return _run(
            lambda x: F.normalize(self.model(x), dim=-1),
            list(paths), (256, 128), "linear", True, self.device, self.batch_size, self.workers, self.amp, desc,
        )


# ---------------------------------------------------------------- builders


def build_fastreid_model(cfg, weights: str | Path | None, device: str = "cuda"):
    """Build a FastReID model from ``cfg`` and load ``weights`` (zoo, init, or training checkpoint)."""
    import torch

    from fastreid.modeling.meta_arch import build_model

    from reidfm.models.init import _load_state, is_classifier_key, remap_checkpoint

    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
    cfg.MODEL.HEADS.NUM_CLASSES = max(1, int(cfg.MODEL.HEADS.NUM_CLASSES))
    model = build_model(cfg)
    if weights:
        state = _load_state(Path(weights))
        model_state = model.state_dict()
        remapped = remap_checkpoint(state, model_state)
        model.load_state_dict(remapped, strict=False)
        missing = [k for k in model_state if k not in remapped and not is_classifier_key(k) and not k.startswith("pixel_")]
        if missing:
            raise RuntimeError(f"{weights}: missing {len(missing)} tensors, e.g. {missing[:5]}")
    return model.eval(), cfg


def build_extractor(
    model_key: str | None = None,
    checkpoint: str | Path | None = None,
    config_file: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    workers: int | None = None,
    amp: bool = True,
    opts: list[str] | None = None,
):
    """Extractor for a registry key, or for a checkpoint + config (default: config.yaml next to it)."""
    from reidfm.config import load_cfg
    from reidfm.models.init import download_weights
    from reidfm.models.registry import get_model_spec

    if model_key:
        spec = get_model_spec(model_key)
        if spec.backend == "osnet" and checkpoint is None:
            return OSNetExtractor(download_weights(spec), device, batch_size, workers, amp), {"model": model_key}
        if spec.init == "imagenet" and checkpoint is None:
            raise ValueError(f"{model_key} has no ReID weights to evaluate")
        config_file = config_file or spec.config_path
        checkpoint = checkpoint or download_weights(spec)
    if checkpoint is None:
        raise ValueError("pass a model key or a checkpoint")
    checkpoint = Path(checkpoint)
    if config_file is None:
        config_file = checkpoint.parent / "config.yaml"
    cfg = load_cfg(config_file, opts)
    model, cfg = build_fastreid_model(cfg, checkpoint, device)
    extractor = FastReIDExtractor(model, cfg, batch_size=batch_size, workers=workers, amp=amp)
    return extractor, {"model": model_key, "checkpoint": str(checkpoint), "config": str(config_file)}
