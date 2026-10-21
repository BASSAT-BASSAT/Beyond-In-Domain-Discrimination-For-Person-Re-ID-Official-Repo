"""LaST (Large-Scale Spatio-Temporal person ReID), following the official loader.

Layout::

    train/<pid>/*.jpg
    val/query/*.jpg     val/gallery/<pid>/*.jpg
    test/query/*.jpg    test/gallery/<pid>/*.jpg

``pid = int(name.split('_')[0])``. LaST has no camera labels; the official protocol gives
every query and gallery image its own camid, so no gallery image is ever discarded as
same-camera junk. Train images get camid ``0`` (unused by the loss).

Options:
  ``eval_split``: single split name (``val`` or ``test``) — legacy default ``test``.
  ``eval_splits``: list of split names to expose as query/gallery (e.g. ``[val, test]``).
    When set, overrides ``eval_split``. Official train stays under ``train/`` only.
"""

from __future__ import annotations

from pathlib import Path

from reidfm.data.converters.base import SourceRow, list_images
from reidfm.data.identity import parse_last_filename, visual_train_pid


def _train_pid(path: Path) -> str:
    raw_pid, clothes_id = parse_last_filename(path.name)
    return visual_train_pid(int(raw_pid), clothes_id)


def _eval_pid(path: Path) -> str:
    return str(int(path.name.split("_")[0]))


def _eval_rows(root: Path, split_name: str, cam_prefix: str) -> list[SourceRow]:
    rows: list[SourceRow] = []
    query = list_images(root / split_name / "query")
    gallery = list_images(root / split_name / "gallery", recursive=True)
    ns = f"{split_name}_"
    rows += [
        SourceRow(src=p, pid=f"{ns}{_eval_pid(p)}", camid=f"{cam_prefix}q{i}", split="query")
        for i, p in enumerate(query)
    ]
    rows += [
        SourceRow(src=p, pid=f"{ns}{_eval_pid(p)}", camid=f"{cam_prefix}g{i}", split="gallery")
        for i, p in enumerate(gallery)
    ]
    return rows


def convert(raw: Path, options: dict) -> list[SourceRow]:
    root = raw / "last" if (raw / "last" / "train").is_dir() else raw
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"LaST: missing {root / 'train'}")

    eval_splits: list[str] = list(options.get("eval_splits") or [])
    if not eval_splits:
        eval_splits = [str(options.get("eval_split", "test"))]

    rows = [
        SourceRow(src=p, pid=_train_pid(p), camid="0", split="train")
        for p in list_images(root / "train", recursive=True)
    ]
    for split_name in eval_splits:
        for sub in ("query", "gallery"):
            if not (root / split_name / sub).is_dir():
                raise FileNotFoundError(f"LaST: missing {root / split_name / sub}")
        rows.extend(_eval_rows(root, split_name, cam_prefix=f"{split_name}_"))
    return rows
