"""Carry a run across Kaggle sessions: copy the previous session's run dir into OUTPUT_DIR.

Kaggle inputs are read-only, so the previous session's output (attached as a dataset or
notebook output) is copied into the writable ``OUTPUT_DIR`` before ``--resume``. Only the
checkpoint named in ``last_checkpoint`` and ``model_best.pth`` are copied, plus small files.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

SMALL_SUFFIXES = {".yaml", ".json", ".txt", ".log", ".md"}


def find_run_dir(root: Path) -> Path | None:
    """First folder at or below ``root`` (depth <= 3) containing ``last_checkpoint``."""
    if (root / "last_checkpoint").is_file():
        return root
    for depth in range(1, 4):
        for candidate in sorted(root.glob("/".join(["*"] * depth))):
            if candidate.is_dir() and (candidate / "last_checkpoint").is_file():
                return candidate
    return None


def restore_run_dir(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path | None:
    """Copy a resumable run from ``src`` into ``dst``; returns the copied run dir or ``None``."""
    dst = Path(dst)
    if (dst / "last_checkpoint").is_file() and not overwrite:
        logger.info("%s already has a checkpoint; not restoring", dst)
        return None
    run = find_run_dir(Path(src))
    if run is None:
        raise FileNotFoundError(f"no last_checkpoint under {src}")
    last = (run / "last_checkpoint").read_text().strip()
    dst.mkdir(parents=True, exist_ok=True)
    wanted = {last, "model_best.pth", "model_last.pth", "model_final.pth", "last_checkpoint"}
    for path in run.iterdir():
        if path.is_file() and (path.name in wanted or path.suffix in SMALL_SUFFIXES or path.name.startswith("events.out")):
            shutil.copy2(path, dst / path.name)
    if not (dst / last).is_file():
        raise FileNotFoundError(f"{run / last} referenced by last_checkpoint is missing")
    logger.info("restored %s (%s) -> %s", run, last, dst)
    return run
