"""MSMT17 (V1 or V2) from the official ``list_*.txt`` files.

Train uses ``list_train.txt + list_val.txt`` (as FastReID does). Lines are ``<rel> <pid>``;
the camera is the third ``_`` field of the file name (``0000_000_01_...`` -> cam 01).
Test pids restart at 0, so they are namespaced as ``test<pid>``.
"""

from __future__ import annotations

from pathlib import Path

from reidfm.data.converters.base import SourceRow

_LAYOUTS = (("train", "test"), ("mask_train_v2", "mask_test_v2"))


def _read_list(path: Path) -> list[tuple[str, int]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            items.append((parts[0], int(parts[1])))
    return items


def convert(raw: Path, options: dict) -> list[SourceRow]:
    train_dir = test_dir = None
    for train_name, test_name in _LAYOUTS:
        if (raw / train_name).is_dir() and (raw / test_name).is_dir():
            train_dir, test_dir = raw / train_name, raw / test_name
            break
    if train_dir is None:
        raise FileNotFoundError(f"MSMT17: expected train/ + test/ (V1) or mask_train_v2/ + mask_test_v2/ under {raw}")

    rows: list[SourceRow] = []
    train_lists = ["list_train.txt"] + ([] if options.get("exclude_val") else ["list_val.txt"])
    for list_name in train_lists:
        for rel, pid in _read_list(raw / list_name):
            cam = int(Path(rel).name.split("_")[2])
            rows.append(SourceRow(src=train_dir / rel, pid=str(pid), camid=str(cam), split="train"))
    for split, list_name in (("query", "list_query.txt"), ("gallery", "list_gallery.txt")):
        for rel, pid in _read_list(raw / list_name):
            cam = int(Path(rel).name.split("_")[2])
            rows.append(SourceRow(src=test_dir / rel, pid=f"test{pid}", camid=str(cam), split=split))
    return rows
