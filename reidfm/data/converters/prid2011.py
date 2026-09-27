"""PRID2011 multi-shot, used only as an out-of-distribution test set.

Layout: ``prid_2011/multi_shot/cam_a/person_XXXX/*.png`` and ``.../cam_b/person_XXXX/*.png``.
Persons 1-200 appear in both cameras. Query = cam_a tracklets of those ids, gallery =
cam_b tracklets. ``gallery_distractors: true`` also adds cam_b-only persons (201-749).
``num_ids`` limits the shared ids (default all 200; the classic protocol tests on 100).
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
    for split, cam, persons, ids in (("query", "a", cam_a, shared), ("gallery", "b", cam_b, gallery_ids)):
        for pid in ids:
            for frame, path in enumerate(list_images(persons[pid])):
                rows.append(SourceRow(src=path, pid=str(pid), camid=cam, split=split, tracklet_id=f"{cam}{pid:04d}", frame=frame))
    return rows
