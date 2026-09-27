"""Model registry: one :class:`ModelSpec` per starting point or trained model.

Keys look like ``<family>_<backbone>_<trainset>`` (``sbs_r50_ibn_msmt``,
``sbs_r101_ibn_imagenet``). Zoo entries live in :mod:`reidfm.models.zoo`; models trained
with this repo are appended to ``configs/models/trained.yaml`` by ``scripts/export_weights.py``
and loaded at import time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from reidfm.utils.paths import repo_path, weights_root

Init = Literal["reid_pretrained", "imagenet", "trained"]
Backend = Literal["fastreid", "osnet"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str  # SBS | BoT | AGW | MGN | OSNet
    backbone: str  # R50 | R50-IBN | S50 | R101-IBN | OSNet-AIN
    trained_on: str  # e.g. "MSMT17", "ImageNet", "reidfm pool v1"
    init: Init
    backend: Backend = "fastreid"
    config_file: str | None = None  # FastReID yaml, relative to the repo root
    weights_url: str | None = None
    weights_file: str | None = None  # file name under $REIDFM_WEIGHTS_ROOT (or a repo-relative path)
    sha256: str | None = None
    input_size: tuple[int, int] = (384, 128)
    feat_dim: int = 2048
    source: str = ""
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_weights(self) -> bool:
        return bool(self.weights_file)

    @property
    def weights_path(self) -> Path | None:
        if not self.weights_file:
            return None
        candidate = Path(self.weights_file)
        if candidate.is_absolute() or len(candidate.parts) > 1:
            return repo_path(candidate)
        return weights_root() / self.weights_file

    @property
    def config_path(self) -> Path | None:
        return repo_path(self.config_file) if self.config_file else None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["input_size"] = list(self.input_size)
        data["tags"] = list(self.tags)
        return data


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec, overwrite: bool = False) -> ModelSpec:
    if spec.key in MODEL_REGISTRY and not overwrite:
        raise KeyError(f"model {spec.key!r} is already registered")
    MODEL_REGISTRY[spec.key] = spec
    return spec


def _ensure_loaded() -> None:
    from reidfm.models import zoo  # noqa: F401  (registers entries on import)


def get_model_spec(key: str) -> ModelSpec:
    _ensure_loaded()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        close = [k for k in MODEL_REGISTRY if key.split("_")[0] in k][:8]
        hint = f" Similar: {', '.join(close)}." if close else ""
        raise KeyError(f"unknown model {key!r}.{hint} Run scripts/list_models.py for the full list.") from exc


def iter_specs() -> Iterable[ModelSpec]:
    _ensure_loaded()
    return (MODEL_REGISTRY[key] for key in sorted(MODEL_REGISTRY))


def available_models(
    family: str | None = None,
    trained_on: str | None = None,
    init: str | None = None,
    backend: str | None = None,
) -> list[str]:
    def keep(spec: ModelSpec) -> bool:
        return (
            (family is None or spec.family.lower() == family.lower())
            and (trained_on is None or trained_on.lower() in spec.trained_on.lower())
            and (init is None or spec.init == init)
            and (backend is None or spec.backend == backend)
        )

    return [spec.key for spec in iter_specs() if keep(spec)]
