"""How raw dataset labels map to training person IDs (visual ReID).

ReID matching is appearance-based: the same ``raw_pid`` wearing different outfits should be
different training identities when ``clothes_id`` is known and non-zero.
"""

from __future__ import annotations

from pathlib import Path


def visual_train_pid(raw_pid: str | int, clothes_id: str | int | None = None) -> str:
    """Return the person id used in ``manifest.csv`` / FastReID training.

    * ``clothes_id`` empty, ``None``, or ``0`` → one identity per ``raw_pid`` (unknown outfit).
    * otherwise → ``{raw_pid}c{clothes_id}`` so two outfits for one annotator pid are two ids.
    """
    pid = str(raw_pid).strip()
    if not pid:
        raise ValueError("raw_pid must be non-empty")
    if clothes_id is None:
        return pid
    cloth = str(clothes_id).strip()
    if not cloth or cloth == "0":
        return pid
    return f"{pid}c{cloth}"


def parse_last_filename(filename: str) -> tuple[str, str]:
    """LaST train filenames: ``{pid}_{image}_{video}_{frame}_{bbox}_{clothes}.jpg``."""
    stem = Path(filename).stem if "." in filename else filename
    parts = stem.split("_")
    if len(parts) != 6:
        raise ValueError(f"expected 6 underscore fields in LaST filename, got {filename!r}")
    pid, _image, _video, _frame, _bbox, clothes = parts
    return pid, clothes
