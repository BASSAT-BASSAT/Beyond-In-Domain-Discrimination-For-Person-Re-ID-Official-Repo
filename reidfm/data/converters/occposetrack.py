"""Occluded-PoseTrack-ReID (images generated from PoseTrack with the KPR scripts).

File names look like ``1703_000028_10000280043.jpg``: ``<pid>_<video>_<frameid>``, where the
last 4 digits of ``frameid`` are the frame index. Each video is treated as one camera and
one tracklet.

Train images are every image under the ``train`` folder. For evaluation:

* if the eval folder contains ``query/`` and ``gallery/``, those are used as-is;
* otherwise (``reidfm`` protocol) each identity gets one query image (the middle frame of
  its longest track) and the rest of its images become gallery, excluding frames within
  ``min_frame_gap`` of the query frame in the same video so near-duplicates don't count.
  Query/gallery get camids ``q``/``g`` so same-video positives are allowed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from reidfm.data.converters.base import SourceRow, find_dir, list_images

_NAME = re.compile(r"^(\d+)_(\d+)_(\d+)")


def _parse(path: Path) -> tuple[str, str, int] | None:
    match = _NAME.match(path.stem)
    if not match:
        return None
    pid, video, frame_id = match.groups()
    return str(int(pid)), video, int(frame_id[-4:])


def convert(raw: Path, options: dict) -> list[SourceRow]:
    train_dir = find_dir(raw, [options.get("train_dir", "train")], max_depth=3)
    eval_names = [options["eval_dir"]] if options.get("eval_dir") else ["val", "test"]
    eval_dir = find_dir(raw, eval_names, max_depth=3)
    if train_dir is None or eval_dir is None:
        raise FileNotFoundError(f"Occluded-PoseTrack: expected train/ and val|test/ folders under {raw}")

    rows: list[SourceRow] = []
    for path in list_images(train_dir, recursive=True):
        parsed = _parse(path)
        if parsed:
            pid, video, frame = parsed
            rows.append(SourceRow(src=path, pid=pid, camid=video, split="train", tracklet_id=f"{pid}-{video}", frame=frame))

    if (eval_dir / "query").is_dir() and (eval_dir / "gallery").is_dir():
        for split in ("query", "gallery"):
            for path in list_images(eval_dir / split, recursive=True):
                parsed = _parse(path)
                if parsed:
                    pid, video, frame = parsed
                    rows.append(SourceRow(src=path, pid=f"test{pid}", camid=video, split=split, frame=frame))
        return rows

    gap = int(options.get("min_frame_gap", 15))
    by_pid: dict[str, dict[str, list[tuple[int, Path]]]] = defaultdict(lambda: defaultdict(list))
    for path in list_images(eval_dir, recursive=True):
        parsed = _parse(path)
        if parsed:
            pid, video, frame = parsed
            by_pid[pid][video].append((frame, path))

    for pid in sorted(by_pid, key=int):
        videos = by_pid[pid]
        for items in videos.values():
            items.sort()
        query_video = max(sorted(videos), key=lambda v: len(videos[v]))
        q_frame, q_path = videos[query_video][len(videos[query_video]) // 2]
        gallery = [
            (video, frame, path)
            for video, items in sorted(videos.items())
            for frame, path in items
            if path != q_path and not (video == query_video and abs(frame - q_frame) < gap)
        ]
        if not gallery:
            continue
        rows.append(SourceRow(src=q_path, pid=f"test{pid}", camid="q", split="query", frame=q_frame))
        rows += [SourceRow(src=p, pid=f"test{pid}", camid="g", split="gallery", frame=f) for _, f, p in gallery]
    return rows
