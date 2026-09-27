"""Download zoo checkpoints and turn them into clean FastReID init weights.

FastReID release checkpoints predate some head refactors (``heads.bnneck``,
``heads.classifier``, MGN ``b*_pool``), so loading them directly silently drops the BNNeck.
:func:`remap_checkpoint` (ported from Person-ReID-BenchMark) fixes the key layout; the
classifier is always dropped because the pool has a different identity set.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Mapping

from reidfm.models.registry import ModelSpec
from reidfm.utils.paths import weights_root
from reidfm.utils.provenance import sha256_file

logger = logging.getLogger(__name__)

ARCH_KEYS = (
    "MODEL.META_ARCHITECTURE",
    "MODEL.BACKBONE.NAME",
    "MODEL.BACKBONE.DEPTH",
    "MODEL.BACKBONE.WITH_IBN",
    "MODEL.BACKBONE.WITH_SE",
    "MODEL.BACKBONE.WITH_NL",
    "MODEL.BACKBONE.FEAT_DIM",
    "MODEL.HEADS.NAME",
    "MODEL.HEADS.POOL_LAYER",
    "MODEL.HEADS.NECK_FEAT",
    "MODEL.HEADS.EMBEDDING_DIM",
)
_MGN_POOLS = {
    "b1_pool": ["b1_head"],
    "b2_pool": ["b2_head", "b21_head", "b22_head"],
    "b3_pool": ["b3_head", "b31_head", "b32_head", "b33_head"],
}


# ---------------------------------------------------------------- download


def download_weights(spec: ModelSpec, verify: bool = True) -> Path:
    """Return the local checkpoint for ``spec``, downloading it if needed."""
    path = spec.weights_path
    if path is None:
        raise ValueError(f"{spec.key} has no weights (init={spec.init})")
    if not path.is_file():
        if not spec.weights_url:
            raise FileNotFoundError(f"{path} missing and {spec.key} has no download URL")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        logger.info("downloading %s -> %s", spec.weights_url, path)
        if "drive.google.com" in spec.weights_url:
            import gdown

            gdown.download(spec.weights_url, str(tmp), quiet=False)
        else:
            with urllib.request.urlopen(spec.weights_url) as response, open(tmp, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
        tmp.replace(path)
    if verify and spec.sha256:
        digest = sha256_file(path)
        if digest != spec.sha256:
            raise RuntimeError(f"{path}: sha256 {digest} != registry {spec.sha256}; delete it and retry")
    return path


# ---------------------------------------------------------------- key remapping


def remap_checkpoint(state: Mapping, model_state: Mapping) -> dict:
    """Map old FastReID zoo keys onto the current model layout; drop shape mismatches."""
    remapped: dict = {}
    for key, value in state.items():
        new_key = key
        if new_key.endswith(".classifier.weight"):
            new_key = new_key[: -len("classifier.weight")] + "weight"
        elif ".bnneck." in new_key:
            prefix, rest = new_key.split(".bnneck.", 1)
            index = 1 if f"{prefix}.bottleneck.1.weight" in model_state else 0
            new_key = f"{prefix}.bottleneck.{index}.{rest}"
        remapped[new_key] = value

    for key, value in state.items():
        for pool, heads in _MGN_POOLS.items():
            if not key.startswith(pool + "."):
                continue
            index, _, name = key[len(pool) + 1 :].partition(".")
            target = {"0": "pool_layer", "1": "bottleneck.0", "2": "bottleneck.1"}.get(index)
            if target:
                for head in heads:
                    remapped[f"{head}.{target}.{name}"] = value

    for key in list(remapped):
        expected = model_state.get(key)
        if expected is None or tuple(expected.shape) != tuple(remapped[key].shape):
            remapped.pop(key)
    return remapped


def is_classifier_key(key: str) -> bool:
    return key == "heads.weight" or key.endswith("_head.weight")


def strip_osnet_checkpoint(state: Mapping) -> dict:
    """torchreid OSNet-AIN state -> FastReID ``backbone.*`` keys (fc/classifier dropped)."""
    out = {}
    for key, value in state.items():
        key = key[len("module.") :] if key.startswith("module.") else key
        if key.startswith(("classifier.", "fc.")):
            continue
        out[f"backbone.{key}"] = value
    return out


def _load_state(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


# ---------------------------------------------------------------- config helpers


def _get(cfg, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def architecture_diff(cfg_a, cfg_b) -> list[str]:
    diffs = []
    for key in ARCH_KEYS:
        try:
            a, b = _get(cfg_a, key), _get(cfg_b, key)
        except KeyError:
            continue
        if a != b:
            diffs.append(f"{key}: {a!r} vs {b!r}")
    return diffs


def spec_cfg(spec: ModelSpec):
    """FastReID cfg of a registered model (built from its yaml)."""
    from reidfm.config import get_reidfm_cfg

    if spec.config_path is None:
        raise ValueError(f"{spec.key} has no config_file")
    cfg = get_reidfm_cfg()
    cfg.merge_from_file(str(spec.config_path))
    return cfg


# ---------------------------------------------------------------- init weights


def prepare_init_weights(spec: ModelSpec, cfg=None, force: bool = False) -> Path:
    """Write ``$REIDFM_WEIGHTS_ROOT/converted/<key>.pth`` ready for ``MODEL.WEIGHTS``.

    ``cfg`` is the run config the weights will be loaded into (defaults to the spec's own
    config). Keys are remapped against a CPU-built model of that config.
    """
    import torch

    from reidfm.fastreid_compat import ensure_fastreid

    ensure_fastreid()
    out = weights_root() / "converted" / f"{spec.key}.pth"
    if out.is_file() and not force:
        return out

    source = download_weights(spec)
    state = _load_state(source)
    if spec.backend == "osnet":
        state = strip_osnet_checkpoint(state)

    from fastreid.modeling.meta_arch import build_model

    build_cfg = (cfg or spec_cfg(spec)).clone()
    build_cfg.defrost()
    build_cfg.MODEL.DEVICE = "cpu"
    build_cfg.MODEL.BACKBONE.PRETRAIN = False
    build_cfg.MODEL.WEIGHTS = ""
    build_cfg.MODEL.HEADS.NUM_CLASSES = max(1, int(build_cfg.MODEL.HEADS.NUM_CLASSES))
    model_state = build_model(build_cfg).state_dict()

    remapped = remap_checkpoint(state, model_state)
    remapped = {k: v for k, v in remapped.items() if not is_classifier_key(k)}
    expected = [k for k in model_state if not is_classifier_key(k) and not k.startswith("pixel_")]
    missing = [k for k in expected if k not in remapped]
    loaded = len(expected) - len(missing)
    logger.info("%s: %d/%d tensors initialised from %s", spec.key, loaded, len(expected), source.name)
    if missing:
        logger.warning("%s: %d tensors start fresh, e.g. %s", spec.key, len(missing), missing[:5])

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": remapped,
            "reidfm_init": {
                "key": spec.key,
                "source": str(source),
                "sha256": spec.sha256,
                "loaded": loaded,
                "expected": len(expected),
                "missing": missing,
            },
        },
        out,
    )
    return out
