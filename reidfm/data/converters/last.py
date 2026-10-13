"""LaST (Large-Scale Spatio-Temporal person ReID), following the official loader.

Layout::

    train/<pid>/*.jpg
    val/query/*.jpg     val/gallery/<pid>/*.jpg
    test/query/*.jpg    test/gallery/<pid>/*.jpg

``pid = int(name.split('_')[0])``. LaST has no camera labels; the official protocol gives
every query and gallery image its own camid, so no gallery image is ever discarded as
same-camera junk. Train images get camid ``0`` (unused by the loss).
Option ``eval_split``: ``test`` (default) or ``val``.
"""

from __future__ import annotations

from pathlib import Path

from reidfm.data.converters.base import SourceRow, list_images
from reidfm.data.identity import parse_last_filename, visual_train_pid


def _train_pid(path: Path) -> str:
    raw_pid, clothes_id = parse_last_filename(path.name)
    return visual_train_pid(int(raw_pid), clothes_id)


def _test_pid(path: Path) -> str:
    return str(int(path.name.split("_")[0]))


def convert(raw: Path, options: dict) -> list[SourceRow]:
    root = raw / "last" if (raw / "last" / "train").is_dir() else raw
    eval_split = options.get("eval_split", "test")
    for folder in (root / "train", root / eval_split / "query", root / eval_split / "gallery"):
        if not folder.is_dir():
            raise FileNotFoundError(f"LaST: missing {folder}")

    rows = [SourceRow(src=p, pid=_train_pid(p), camid="0", split="train") for p in list_images(root / "train", recursive=True)]
    query = list_images(root / eval_split / "query")
    gallery = list_images(root / eval_split / "gallery", recursive=True)
    # Test pids share a namespace across query/gallery but not with train.
    rows += [SourceRow(src=p, pid=f"test{_test_pid(p)}", camid=f"q{i}", split="query") for i, p in enumerate(query)]
    rows += [SourceRow(src=p, pid=f"test{_test_pid(p)}", camid=f"g{i}", split="gallery") for i, p in enumerate(gallery)]
    return rows
