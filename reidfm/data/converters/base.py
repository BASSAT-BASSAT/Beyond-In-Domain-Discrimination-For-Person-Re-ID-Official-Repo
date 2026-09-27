"""Shared converter machinery: source rows, subsampling, materialization, manifest writing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from reidfm.data.manifest import META_NAME, MANIFEST_NAME, ManifestRow, manifest_counts, validate_rows, write_manifest

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MATERIALIZE_MODES = ("copy", "link", "none")


@dataclass
class SourceRow:
    """One image produced by a converter before it is written to the pool."""

    src: Path
    pid: str
    camid: str
    split: str
    tracklet_id: str = ""
    frame: int | None = None
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 in source pixels


# ---------------------------------------------------------------- filesystem helpers


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_dir(root: Path, names: Sequence[str], max_depth: int = 3) -> Path | None:
    """Breadth-first search for a directory whose name is in ``names``."""
    wanted = {name.lower() for name in names}
    frontier = [root]
    for _ in range(max_depth + 1):
        next_frontier: list[Path] = []
        for folder in frontier:
            if folder.name.lower() in wanted and folder.is_dir():
                return folder
            try:
                next_frontier.extend(sorted(p for p in folder.iterdir() if p.is_dir()))
            except OSError:
                continue
        frontier = next_frontier
    return None


def require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{what}: expected folder {path}")
    return path


# ---------------------------------------------------------------- subsampling


def linspace_indices(n: int, k: int) -> list[int]:
    """Benchmark sampling rule: ``np.linspace(0, n-1, k)`` truncated to ints, de-duplicated."""
    if n <= k:
        return list(range(n))
    return [int(i) for i in np.unique(np.linspace(0, n - 1, num=k, dtype=np.int64))]


def _frame_sort_key(row: SourceRow):
    return (row.frame if row.frame is not None else -1, str(row.src), row.box or ())


def group_tracklets(rows: Iterable[SourceRow]) -> dict[tuple[str, str, str], list[SourceRow]]:
    groups: dict[tuple[str, str, str], list[SourceRow]] = defaultdict(list)
    for row in rows:
        groups[(row.split, row.pid, row.tracklet_id)].append(row)
    for items in groups.values():
        items.sort(key=_frame_sort_key)
    return groups


def subsample_train_tracklets(rows: list[SourceRow], stride: int = 1, max_frames: int | None = None) -> list[SourceRow]:
    """Thin redundant video frames in train tracklets (every ``stride``-th, then at most ``max_frames``)."""
    if stride <= 1 and not max_frames:
        return rows
    kept: list[SourceRow] = []
    for (split, _, tracklet), items in group_tracklets(rows).items():
        if split != "train" or not tracklet:
            kept.extend(items)
            continue
        items = items[:: max(1, stride)]
        if max_frames:
            items = [items[i] for i in linspace_indices(len(items), max_frames)]
        kept.extend(items)
    return kept


def sample_eval_tracklets(rows: list[SourceRow], num_frames: int | None) -> list[SourceRow]:
    """Keep only the frames the evaluator would sample from each query/gallery tracklet.

    Evaluation uses the same ``linspace`` rule on the stored frames, so pre-sampling here
    gives identical metrics while shrinking the pool by 10-100x for video test sets.
    """
    if not num_frames:
        return rows
    kept: list[SourceRow] = []
    for (split, _, tracklet), items in group_tracklets(rows).items():
        if split == "train" or not tracklet:
            kept.extend(items)
            continue
        kept.extend(items[i] for i in linspace_indices(len(items), num_frames))
    return kept


def _water_fill(sizes: list[int], budget: int) -> list[int]:
    quotas = [0] * len(sizes)
    remaining = budget
    open_groups = [i for i, size in enumerate(sizes) if size > 0]
    while remaining > 0 and open_groups:
        share = max(1, remaining // len(open_groups))
        still_open = []
        for i in open_groups:
            take = min(share, sizes[i] - quotas[i], remaining)
            quotas[i] += take
            remaining -= take
            if quotas[i] < sizes[i]:
                still_open.append(i)
            if remaining == 0:
                break
        open_groups = still_open
    return quotas


def cap_per_identity(rows: list[SourceRow], max_per_id: int | None, split: str = "train") -> list[SourceRow]:
    """Cap images per identity, spreading the kept ones evenly over cameras and tracklets."""
    if not max_per_id:
        return rows
    by_pid: dict[str, list[SourceRow]] = defaultdict(list)
    others: list[SourceRow] = []
    for row in rows:
        (by_pid[row.pid] if row.split == split else others).append(row)
    kept = others
    for items in by_pid.values():
        if len(items) <= max_per_id:
            kept.extend(items)
            continue
        groups: dict[tuple[str, str], list[SourceRow]] = defaultdict(list)
        for row in items:
            groups[(row.camid, row.tracklet_id)].append(row)
        keys = sorted(groups)
        for key in keys:
            groups[key].sort(key=_frame_sort_key)
        quotas = _water_fill([len(groups[key]) for key in keys], max_per_id)
        for key, quota in zip(keys, quotas):
            members = groups[key]
            kept.extend(members[i] for i in linspace_indices(len(members), quota))
    return kept


# ---------------------------------------------------------------- materialization

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(text: str) -> str:
    return _SAFE.sub("_", text).strip("_") or "x"


def _relative_name(src: Path, raw_root: Path) -> Path:
    try:
        return src.resolve().relative_to(raw_root.resolve())
    except ValueError:
        digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
        return Path(f"{digest}_{src.name}")


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def _crop_path(row: SourceRow, index: int) -> Path:
    frame = f"f{row.frame:06d}" if row.frame is not None else f"n{index:07d}"
    track = f"_{_safe(row.tracklet_id)}" if row.tracklet_id else ""
    return Path("images") / row.split / _safe(row.pid) / f"{_safe(row.camid)}{track}_{frame}_{index:07d}.jpg"


def _save_crops(frame_src: Path, jobs: list[tuple[tuple[float, float, float, float], Path]], quality: int) -> None:
    from PIL import Image

    with Image.open(frame_src) as image:
        image = image.convert("RGB")
        width, height = image.size
        for (x1, y1, x2, y2), dst in jobs:
            box = (max(0, int(round(x1))), max(0, int(round(y1))), min(width, int(round(x2))), min(height, int(round(y2))))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            image.crop(box).save(dst, quality=quality)


def materialize(
    rows: list[SourceRow],
    dataset: str,
    raw_root: Path,
    out_dir: Path,
    mode: str = "copy",
    workers: int = 8,
    jpeg_quality: int = 95,
) -> list[ManifestRow]:
    """Write images into ``out_dir/images`` (or reference them in place) and return manifest rows.

    Crop rows (``box`` set) are always written as new JPEGs, whatever ``mode`` says.
    """
    if mode not in MATERIALIZE_MODES:
        raise ValueError(f"materialize must be one of {MATERIALIZE_MODES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[ManifestRow] = []
    copy_jobs: list[tuple[Path, Path]] = []
    crop_jobs: dict[Path, list[tuple[tuple[float, float, float, float], Path]]] = defaultdict(list)

    for index, row in enumerate(rows):
        if row.box is not None:
            rel = _crop_path(row, index)
            crop_jobs[row.src].append((row.box, out_dir / rel))
            path_str = rel.as_posix()
        elif mode == "none":
            path_str = str(row.src.resolve())
        else:
            rel = Path("images") / row.split / _relative_name(row.src, raw_root)
            copy_jobs.append((row.src, out_dir / rel))
            path_str = rel.as_posix()
        manifest_rows.append(
            ManifestRow(
                path=path_str,
                pid=row.pid,
                camid=row.camid,
                dataset=dataset,
                split=row.split,
                tracklet_id=row.tracklet_id,
                frame="" if row.frame is None else str(row.frame),
            )
        )

    def _copy(job: tuple[Path, Path]) -> None:
        src, dst = job
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "link":
            _link_or_copy(src, dst)
        else:
            shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_copy, copy_jobs))
        list(pool.map(lambda item: _save_crops(item[0], item[1], jpeg_quality), crop_jobs.items()))

    # Drop rows whose crop was degenerate (too small after clipping).
    crop_paths = {dst for jobs in crop_jobs.values() for _, dst in jobs}
    return [row for row in manifest_rows if (out_dir / row.path) not in crop_paths or (out_dir / row.path).is_file()]


def write_dataset(out_dir: Path, rows: list[ManifestRow], meta: dict) -> dict:
    errors = validate_rows(rows)
    if errors:
        raise ValueError(f"{out_dir.name}: invalid manifest:\n  " + "\n  ".join(errors))
    write_manifest(out_dir / MANIFEST_NAME, rows)
    meta = {**meta, "counts": manifest_counts(rows)}
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return meta
