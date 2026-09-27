"""Raw dataset -> unified manifest converters.

Every converter is ``convert(raw: Path, options: dict) -> list[SourceRow]`` and only knows
its dataset's layout. :func:`build_dataset` applies the generic options, writes images and
``manifest.csv``/``meta.json``.

Generic options (per dataset, from the pool's ``converters:`` block):

* ``raw_dir``: explicit raw folder (absolute, or relative to ``REIDFM_RAW_ROOT``).
* ``max_train_per_id``: cap train images per identity (spread over cams/tracklets).
* ``train_frame_stride`` / ``train_max_frames_per_tracklet``: thin video train tracklets.
* ``eval_frames_per_tracklet``: keep only the frames evaluation samples (video; default 8).
"""

from __future__ import annotations

import time
from pathlib import Path

from reidfm import __version__
from reidfm.data.converters.base import (
    SourceRow,
    cap_per_identity,
    materialize,
    sample_eval_tracklets,
    subsample_train_tracklets,
    write_dataset,
)
from reidfm.data.registry import get_dataset_spec
from reidfm.utils.paths import raw_root as default_raw_root

GENERIC_OPTIONS = (
    "raw_dir",
    "max_train_per_id",
    "train_frame_stride",
    "train_max_frames_per_tracklet",
    "eval_frames_per_tracklet",
)


def locate_raw(key: str, options: dict, raw_root: Path | None = None) -> Path:
    spec = get_dataset_spec(key)
    raw_root = raw_root or default_raw_root()
    explicit = options.get("raw_dir")
    if explicit:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else raw_root / path
        if not path.exists():
            raise FileNotFoundError(f"{key}: raw_dir {path} does not exist")
        return path
    for name in spec.raw_dirs:
        if (raw_root / name).exists():
            return raw_root / name
    raise FileNotFoundError(f"{key}: none of {list(spec.raw_dirs)} found under {raw_root}; set converters.{key}.raw_dir")


def collect_rows(
    key: str,
    options: dict,
    raw_root: Path | None = None,
    include_train: bool = True,
) -> tuple[Path, list[SourceRow]]:
    spec = get_dataset_spec(key)
    raw = locate_raw(key, options, raw_root)
    rows = spec.load_converter()(raw, options)
    if not spec.has_train or not include_train:
        rows = [row for row in rows if row.split != "train"]
    if not spec.has_test:
        rows = [row for row in rows if row.split == "train"]
    rows = subsample_train_tracklets(
        rows,
        stride=int(options.get("train_frame_stride") or 1),
        max_frames=options.get("train_max_frames_per_tracklet"),
    )
    rows = cap_per_identity(rows, options.get("max_train_per_id"))
    eval_frames = options.get("eval_frames_per_tracklet", 8 if spec.kind == "video" else None)
    rows = sample_eval_tracklets(rows, eval_frames)
    if not rows:
        raise RuntimeError(f"{key}: converter produced no rows from {raw}")
    return raw, rows


def build_dataset(
    key: str,
    out_root: Path,
    options: dict | None = None,
    raw_root: Path | None = None,
    mode: str = "copy",
    workers: int = 8,
    include_train: bool = True,
) -> dict:
    """Convert one dataset into ``out_root/<key>/`` and return its ``meta.json`` payload.

    ``include_train=False`` keeps only query/gallery (datasets that are eval-only in a pool,
    such as OOD sets, never ship a train split).
    """
    options = dict(options or {})
    spec = get_dataset_spec(key)
    started = time.time()
    raw, rows = collect_rows(key, options, raw_root, include_train=include_train)
    out_dir = out_root / key
    manifest_rows = materialize(rows, key, raw, out_dir, mode=mode, workers=workers)
    meta = {
        "dataset": key,
        "fastreid_name": spec.fastreid_name,
        "kind": spec.kind,
        "protocol": spec.protocol,
        "restricted": spec.restricted,
        "raw_dir": str(raw),
        "materialize": mode,
        "include_train": include_train,
        "options": options,
        "reidfm_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 1),
    }
    return write_dataset(out_dir, manifest_rows, meta)


__all__ = ["GENERIC_OPTIONS", "SourceRow", "build_dataset", "collect_rows", "locate_raw"]
