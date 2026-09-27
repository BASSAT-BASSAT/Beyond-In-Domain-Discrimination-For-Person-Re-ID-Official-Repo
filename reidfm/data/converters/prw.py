"""PRW (Person Re-identification in the Wild): person crops cut from full frames.

Layout (``PRW-v16.04.20``)::

    frames/c1s1_000151.jpg
    annotations/c1s1_000151.jpg.mat      # rows [id, x, y, w, h]; id -2 = unlabeled person
    frame_train.mat (img_index_train)   frame_test.mat (img_index_test)
    query_info.txt                      # "<id> <x> <y> <w> <h> <frame>"

Crop-level protocol: train = labeled boxes in train frames. Query = the official 2,057
query boxes. Gallery = labeled boxes in test frames, minus the query boxes themselves.
``include_unlabeled_gallery: true`` adds the unlabeled (-2) boxes as gallery distractors.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from reidfm.data.converters.base import SourceRow

_CAM = re.compile(r"c(\d+)s(\d+)_(\d+)")
_BOX_KEYS = ("box_new", "anno_file", "anno_previous")


def _load_boxes(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    data = loadmat(path)
    for key in _BOX_KEYS:
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1, 5)
    raise KeyError(f"{path}: none of {_BOX_KEYS}")


def _frame_list(path: Path, key: str) -> list[str]:
    from scipy.io import loadmat

    values = np.asarray(loadmat(path)[key]).reshape(-1)
    names = []
    for value in values:
        name = str(value[0] if isinstance(value, np.ndarray) else value).strip()
        names.append(name if name.endswith(".jpg") else f"{name}.jpg")
    return names


def _meta(frame_name: str) -> tuple[str, int]:
    match = _CAM.search(frame_name)
    if not match:
        raise ValueError(f"PRW: unexpected frame name {frame_name}")
    return match.group(1), int(match.group(3))


def _box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def convert(raw: Path, options: dict) -> list[SourceRow]:
    root = raw if (raw / "frames").is_dir() else next((p for p in raw.iterdir() if (p / "frames").is_dir()), raw)
    frames, annotations = root / "frames", root / "annotations"
    if not frames.is_dir() or not annotations.is_dir():
        raise FileNotFoundError(f"PRW: expected frames/ and annotations/ under {root}")
    min_height = float(options.get("min_box_height", 0))
    include_unlabeled = bool(options.get("include_unlabeled_gallery", False))

    rows: list[SourceRow] = []
    query_keys: set[tuple[str, int]] = set()
    for line in (root / "query_info.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        pid, x, y, w, h, frame_name = int(parts[0]), *map(float, parts[1:5]), parts[5]
        frame_name = frame_name if frame_name.endswith(".jpg") else f"{frame_name}.jpg"
        cam, frame = _meta(frame_name)
        query_keys.add((frame_name, pid))
        rows.append(SourceRow(src=frames / frame_name, pid=f"test{pid}", camid=cam, split="query", frame=frame, box=_box(x, y, w, h)))

    for split, list_file, key in (("train", "frame_train.mat", "img_index_train"), ("gallery", "frame_test.mat", "img_index_test")):
        for frame_name in _frame_list(root / list_file, key):
            cam, frame = _meta(frame_name)
            for pid, x, y, w, h in _load_boxes(annotations / f"{frame_name}.mat"):
                pid = int(pid)
                if h < min_height:
                    continue
                if pid < 0:
                    if split == "gallery" and include_unlabeled:
                        rows.append(SourceRow(src=frames / frame_name, pid="0", camid=cam, split=split, frame=frame, box=_box(x, y, w, h)))
                    continue
                if split == "gallery" and (frame_name, pid) in query_keys:
                    continue
                label = str(pid) if split == "train" else f"test{pid}"
                rows.append(SourceRow(src=frames / frame_name, pid=label, camid=cam, split=split, frame=frame, box=_box(x, y, w, h)))
    return rows
