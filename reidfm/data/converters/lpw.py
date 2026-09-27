"""LPW (Labeled Pedestrian in the Wild), video tracklets.

Layout: ``pep_256x128/data_slim/scen{1,2,3}/<view>/<pid>/*.jpg``. The same ``<pid>`` folder
across views of one scene is the same person, so identity = ``<scene>-<pid>``, camera =
``<scene>-<view>``, tracklet = ``<scene>-<view>-<pid>``.

Standard protocol: scen2 + scen3 train; scen1 test with ``view2`` as query and the other
views as gallery. Override with ``train_scenes``, ``test_scene``, ``query_view``.
"""

from __future__ import annotations

import re
from pathlib import Path

from reidfm.data.converters.base import SourceRow, find_dir, list_images

_DIGITS = re.compile(r"(\d+)")


def _frame(path: Path) -> int | None:
    numbers = _DIGITS.findall(path.stem)
    return int(numbers[-1]) if numbers else None


def _scene_rows(scene_dir: Path, split_for_view) -> list[SourceRow]:
    rows: list[SourceRow] = []
    scene = scene_dir.name
    for view_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
        split = split_for_view(view_dir.name)
        if split is None:
            continue
        for pid_dir in sorted(p for p in view_dir.iterdir() if p.is_dir()):
            pid = f"{scene}-{pid_dir.name}"
            if split != "train":
                pid = f"test{pid}"
            tracklet = f"{scene}-{view_dir.name}-{pid_dir.name}"
            for path in list_images(pid_dir):
                rows.append(SourceRow(src=path, pid=pid, camid=f"{scene}-{view_dir.name}", split=split, tracklet_id=tracklet, frame=_frame(path)))
    return rows


def convert(raw: Path, options: dict) -> list[SourceRow]:
    base = raw if (raw / "scen1").is_dir() else find_dir(raw, ["data_slim"], max_depth=3)
    if base is None or not (base / "scen1").is_dir():
        raise FileNotFoundError(f"LPW: expected .../data_slim/scen1 under {raw}")
    train_scenes = options.get("train_scenes", ["scen2", "scen3"])
    test_scene = options.get("test_scene", "scen1")
    query_view = options.get("query_view", "view2")

    rows: list[SourceRow] = []
    for scene in train_scenes:
        rows += _scene_rows(base / scene, lambda _view: "train")
    rows += _scene_rows(base / test_scene, lambda view: "query" if view == query_view else "gallery")
    return rows
