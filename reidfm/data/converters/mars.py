"""MARS video ReID (same protocol as Person-ReID-BenchMark ``reid/data/mars.py``).

Layout::

    bbox_train/<pid4>/<name>.jpg   bbox_test/<pid4>/<name>.jpg
    info/train_name.txt  info/test_name.txt
    info/tracks_train_info.mat (track_train_info)   info/tracks_test_info.mat (track_test_info)
    info/query_IDX.mat (query_IDX, 1-based)

Track rows are ``[start, end, pid, cam]`` (1-based name indices). ``pid == -1`` tracks are
dropped; pid 0 distractor tracks stay in the gallery. Gallery = every test track not in
``query_IDX``. Test pids are namespaced ``test<pid>`` (MARS test ids are disjoint anyway).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reidfm.data.converters.base import SourceRow


def _names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mat(path: Path, key: str) -> np.ndarray:
    from scipy.io import loadmat

    return np.asarray(loadmat(path)[key])


def _tracks(names, metadata, images_dir: Path, split: str, indices, pid_prefix: str) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for index in indices:
        start, end, pid, cam = (int(v) for v in metadata[index])
        if pid == -1 or (pid == 0 and split != "gallery"):
            continue
        tracklet = f"{split}{index + 1:06d}"
        for offset, name in enumerate(names[start - 1 : end]):
            rows.append(
                SourceRow(
                    src=images_dir / name[:4] / name,
                    pid=f"{pid_prefix}{pid}",
                    camid=str(cam),
                    split=split,
                    tracklet_id=tracklet,
                    frame=offset,
                )
            )
    return rows


def convert(raw: Path, options: dict) -> list[SourceRow]:
    info = raw / "info"
    for required in ("train_name.txt", "test_name.txt", "tracks_train_info.mat", "tracks_test_info.mat", "query_IDX.mat"):
        if not (info / required).is_file():
            raise FileNotFoundError(f"MARS: missing info/{required} under {raw}")
    train_meta = _mat(info / "tracks_train_info.mat", "track_train_info")
    test_meta = _mat(info / "tracks_test_info.mat", "track_test_info")
    query_idx = (_mat(info / "query_IDX.mat", "query_IDX").reshape(-1).astype(np.int64) - 1).tolist()
    query_set = set(query_idx)
    gallery_idx = [i for i in range(len(test_meta)) if i not in query_set]

    train_names, test_names = _names(info / "train_name.txt"), _names(info / "test_name.txt")
    rows = _tracks(train_names, train_meta, raw / "bbox_train", "train", range(len(train_meta)), "")
    rows += _tracks(test_names, test_meta, raw / "bbox_test", "query", query_idx, "test")
    rows += _tracks(test_names, test_meta, raw / "bbox_test", "gallery", gallery_idx, "test")
    return rows
