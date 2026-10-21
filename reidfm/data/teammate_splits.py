"""Load per-dataset train/val identity lists pushed by teammates (optional)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from reidfm.data.manifest import Manifest
from reidfm.utils.paths import repo_path

SPLITS_DATASETS_DIR = repo_path("splits/datasets")


def splits_root() -> Path:
    override = os.environ.get("REIDFM_SPLITS_ROOT")
    if override:
        return Path(override)
    return SPLITS_DATASETS_DIR


def teammate_split_path(dataset_key: str, root: Path | None = None) -> Path:
    root = root or splits_root()
    return root / f"{dataset_key}.json"


def load_teammate_split(dataset_key: str, root: Path | None = None) -> dict | None:
    path = teammate_split_path(dataset_key, root)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "train" not in data or "val" not in data:
        raise ValueError(f"{path}: must contain 'train' and 'val' pid lists")
    data = dict(data)
    data["_path"] = str(path)
    return data


def validate_teammate_split(dataset_key: str, manifest: Manifest, data: dict) -> tuple[list[str], list[str]]:
    allowed = {row.pid for row in manifest.train}
    train = sorted({str(p) for p in data["train"]})
    val = sorted({str(p) for p in data["val"]})
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"{dataset_key}: train/val pid overlap: {sorted(overlap)[:5]}")
    unknown = (set(train) | set(val)) - allowed
    if unknown:
        sample = sorted(unknown)[:5]
        raise ValueError(f"{dataset_key}: split pids not in manifest.train (e.g. {sample})")
    if not train:
        raise ValueError(f"{dataset_key}: empty train list in teammate split")
    return train, val


def list_pending_teammate_splits(pool_sources: list[str]) -> list[str]:
    from reidfm.data.split_policies import get_split_policy

    pending = []
    for key in pool_sources:
        if get_split_policy(key).kind != "teammate_or_dhash":
            continue
        if load_teammate_split(key) is None:
            pending.append(key)
    return pending
