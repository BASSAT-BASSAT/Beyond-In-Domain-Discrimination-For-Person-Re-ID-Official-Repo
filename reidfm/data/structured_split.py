"""Train/val splits when no official test: camera/scene groups, then identities, then dHash."""

from __future__ import annotations

import random
from collections import defaultdict

from reidfm.data.manifest import Manifest, ManifestRow
from reidfm.data.split_policies import identity_disjoint_train_val, train_pids_from_manifest

DEFAULT_TRAIN_FRACTION = 0.8


def _primary_camid(rows: list[ManifestRow]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.camid] += 1
    return max(counts, key=counts.get)


def _pid_scene_token(pid: str, dataset_key: str) -> str | None:
    """Coarse scene/sequence token when camid is not informative."""
    if dataset_key in ("mot17", "mot20") and "-" in pid:
        return pid.rsplit("-", 1)[0]
    if "-" in pid:
        return pid.split("-", 1)[0]
    return None


def camera_or_scene_groups(manifest: Manifest, dataset_key: str) -> dict[str, list[str]]:
    """Group train identities by camera (preferred) or scene/name prefix."""
    by_pid: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in manifest.train:
        by_pid[row.pid].append(row)

    by_cam: dict[str, list[str]] = defaultdict(list)
    for pid, rows in by_pid.items():
        by_cam[_primary_camid(rows)].append(pid)

    if len(by_cam) >= 2:
        return {f"cam:{k}": sorted(v) for k, v in by_cam.items() if v}

    by_scene: dict[str, list[str]] = defaultdict(list)
    for pid in by_pid:
        token = _pid_scene_token(pid, dataset_key)
        if token:
            by_scene[token].append(pid)
    if len(by_scene) >= 2:
        return {f"scene:{k}": sorted(v) for k, v in by_scene.items() if v}

    return {}


def assign_pid_groups(
    groups: dict[str, list[str]],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], dict]:
    """Assign whole groups to train/val targeting ~train_fraction of identities."""
    all_pids = sorted({p for ps in groups.values() for p in ps})
    if len(all_pids) <= 1:
        return all_pids, [], {"strategy": "camera_scene", "n_groups": len(groups), "skipped": "too_few_ids"}

    rng = random.Random(seed)
    total = len(all_pids)
    target_val = max(1, int(round((1.0 - train_fraction) * total)))
    target_val = min(target_val, total - 1)

    keys = list(groups.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]))

    val_set: set[str] = set()
    for key in keys:
        if len(val_set) >= target_val:
            break
        val_set |= set(groups[key])
    train_set = set(all_pids) - val_set
    if not train_set and len(keys) > 1:
        largest_val_key = max(keys, key=lambda k: len(groups[k]) if set(groups[k]) <= val_set else 0)
        val_set -= set(groups[largest_val_key])
        train_set = set(all_pids) - val_set
    train = sorted(train_set)
    val = sorted(val_set)
    return train, val, {
        "strategy": "camera_scene",
        "n_groups": len(groups),
        "n_train_ids": len(train),
        "n_val_ids": len(val),
        "train_fraction_target": train_fraction,
        "train_fraction_actual": len(train) / total if total else None,
    }


def split_structured_or_dhash(
    dataset_key: str,
    manifest: Manifest,
    *,
    train_fraction: float,
    max_samples_per_id: int,
    dhash_threshold: int,
    seed: int,
    rng: random.Random,
    dhash_fn,
) -> tuple[list[str], list[str], list[list[str]], dict]:
    """Camera/scene groups → identity holdout → perceptual-hash clusters."""
    groups = camera_or_scene_groups(manifest, dataset_key)
    if groups:
        train, val, meta = assign_pid_groups(groups, train_fraction=train_fraction, seed=seed)
        if val:
            clusters = [[f"{dataset_key}:{p}"] for p in train + val]
            meta["fallback"] = None
            return train, val, clusters, meta
        meta["fallback"] = "camera_scene_no_val"

    pids = train_pids_from_manifest(manifest)
    if len(pids) >= 2:
        train, val, meta = identity_disjoint_train_val(pids, train_fraction=train_fraction, seed=seed)
        meta["strategy"] = "identity_disjoint"
        clusters = [[f"{dataset_key}:{p}"] for p in train + val]
        if val:
            return train, val, clusters, meta

    train, val, clusters, meta = dhash_fn(
        dataset_key,
        manifest,
        train_fraction=train_fraction,
        max_samples_per_id=max_samples_per_id,
        dhash_threshold=dhash_threshold,
        rng=rng,
    )
    meta["strategy"] = "dhash_cluster"
    meta["fallback"] = "pixel_dhash"
    return train, val, clusters, meta
