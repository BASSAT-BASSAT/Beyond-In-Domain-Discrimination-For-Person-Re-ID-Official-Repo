"""PRID2011 multi-shot: train candidates + official cam_a query / cam_b gallery test.

Layout: ``prid_2011/multi_shot/cam_a/person_XXXX/*.png`` and ``.../cam_b/person_XXXX/*.png``.
Persons 1-200 appear in both cameras. With ``include_train: true``, multi-shot frames from
both cameras become ``split=train`` with pid ``tr<id>`` so they do not overlap test pids.
"""

from __future__ import annotations

import re
from pathlib import Path

from reidfm.data.converters.base import SourceRow, find_dir, list_images

_PERSON = re.compile(r"person_(\d+)")
SHARED_IDS = 200


def _persons(cam_dir: Path) -> dict[int, Path]:
    found = {}
    for folder in cam_dir.iterdir():
        match = _PERSON.fullmatch(folder.name)
        if match and folder.is_dir():
            found[int(match.group(1))] = folder
    return found


def _train_rows(cam_a: dict[int, Path], cam_b: dict[int, Path], shared: list[int]) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for pid in shared:
        for cam_tag, persons in (("a", cam_a), ("b", cam_b)):
            folder = persons.get(pid)
            if folder is None:
                continue
            tracklet = f"tr{cam_tag}{pid:04d}"
            for frame, path in enumerate(list_images(folder)):
                rows.append(
                    SourceRow(
                        src=path,
                        pid=f"tr{pid}",
                        camid=cam_tag,
                        split="train",
                        tracklet_id=tracklet,
                        frame=frame,
                    )
                )
    return rows


def convert(raw: Path, options: dict) -> list[SourceRow]:
    shots = find_dir(raw, [options.get("variant", "multi_shot")], max_depth=2)
    if shots is None or not (shots / "cam_a").is_dir():
        raise FileNotFoundError(f"PRID2011: expected multi_shot/cam_a and cam_b under {raw}")
    cam_a, cam_b = _persons(shots / "cam_a"), _persons(shots / "cam_b")
    num_ids = int(options.get("num_ids", SHARED_IDS))
    shared = [pid for pid in sorted(cam_a) if pid <= SHARED_IDS and pid in cam_b][:num_ids]
    gallery_ids = list(shared)
    if options.get("gallery_distractors"):
        gallery_ids += [pid for pid in sorted(cam_b) if pid > SHARED_IDS]

    rows: list[SourceRow] = []
    if options.get("include_train", False):
        rows.extend(_train_rows(cam_a, cam_b, shared))
    for split, cam, persons, ids in (("query", "a", cam_a, shared), ("gallery", "b", cam_b, gallery_ids)):
        for pid in ids:
            for frame, path in enumerate(list_images(persons[pid])):
                rows.append(
                    SourceRow(
                        src=path,
                        pid=str(pid),
                        camid=cam,
                        split=split,
                        tracklet_id=f"{cam}{pid:04d}",
                        frame=frame,
                    )
                )
    return rows
