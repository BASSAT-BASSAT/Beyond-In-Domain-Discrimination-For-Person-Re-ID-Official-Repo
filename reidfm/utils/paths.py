"""Repository paths and the env-var driven data/weights/output roots."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FASTREID_REPO = "https://github.com/JDAI-CV/fast-reid.git"
FASTREID_COMMIT = "c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d"
FASTREID_ROOT = Path(os.environ.get("REIDFM_FASTREID_ROOT", REPO_ROOT / "fast-reid"))


def _resolve(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def env_path(var: str, default: str | os.PathLike) -> Path:
    """Path from ``$var`` (relative paths are relative to the repo root)."""
    return _resolve(os.environ.get(var) or default)


def raw_root() -> Path:
    return env_path("REIDFM_RAW_ROOT", "raw")


def data_roots() -> list[Path]:
    """Roots searched for ``<dataset>/manifest.csv``; ``$REIDFM_DATA_ROOT`` may hold several."""
    value = os.environ.get("REIDFM_DATA_ROOT") or "pool"
    return [_resolve(part) for part in value.split(os.pathsep) if part.strip()]


def output_root() -> Path:
    return env_path("REIDFM_OUTPUT_ROOT", "outputs")


def weights_root() -> Path:
    return env_path("REIDFM_WEIGHTS_ROOT", "weights")


def repo_path(value: str | os.PathLike) -> Path:
    """Resolve a path stored in configs/registry entries (relative to the repo root)."""
    return _resolve(value)
