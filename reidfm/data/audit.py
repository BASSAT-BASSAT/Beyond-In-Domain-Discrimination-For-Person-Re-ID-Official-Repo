"""Pool audit: per-dataset stats, manifest sanity, identity leakage, OOD isolation, near-duplicates."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reidfm.data.manifest import Manifest, manifest_counts, validate_rows
from reidfm.data.pool import PoolConfig

# Pairs that share cameras/sites, so identities or even crops can overlap across datasets.
KNOWN_OVERLAPS: tuple[tuple[str, str, str], ...] = (
    ("mars", "market1501", "MARS was recorded with the Market-1501 campus cameras"),
    ("prw", "market1501", "PRW was filmed on the Market-1501 campus"),
    ("market1501", "mars", "Market-1501 and MARS share the campus cameras"),
    ("market1501", "prw", "Market-1501 and PRW share the campus"),
    ("mars", "prw", "MARS and PRW share the Market-1501 campus"),
)


@dataclass
class DatasetAudit:
    key: str
    counts: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def audit_manifest(manifest: Manifest, check_files: int = 200, seed: int = 0) -> DatasetAudit:
    report = DatasetAudit(key=manifest.name, counts=manifest_counts(manifest.rows))
    report.errors += validate_rows(manifest.rows)

    rng = random.Random(seed)
    sample = rng.sample(manifest.rows, min(check_files, len(manifest.rows)))
    missing = [row.path for row in sample if not manifest.resolve(row).is_file()]
    if missing:
        report.errors.append(f"{len(missing)}/{len(sample)} sampled files missing, e.g. {missing[0]}")

    train_ids = {row.pid for row in manifest.train}
    test_ids = {row.pid for row in manifest.query + manifest.gallery if row.pid != "0"}
    overlap = train_ids & test_ids
    if overlap:
        report.errors.append(f"identity leakage: {len(overlap)} ids in both train and query/gallery, e.g. {sorted(overlap)[:3]}")

    if manifest.query:
        gallery_ids = {row.pid for row in manifest.gallery}
        orphan = {row.pid for row in manifest.query} - gallery_ids
        if orphan:
            report.warnings.append(f"{len(orphan)} query ids have no gallery match (skipped by the metric)")
    return report


def check_pool_roles(pool: PoolConfig, manifests: dict[str, Manifest]) -> list[str]:
    errors: list[str] = []
    for key in set(pool.sources) & set(pool.ood):
        errors.append(f"{key} is both a training source and OOD")
    for key in pool.sources:
        if key in manifests and not manifests[key].has_train:
            errors.append(f"source {key} has no train rows")
    for key in [*pool.val_sets, *pool.test_sets, *pool.ood]:
        if key in manifests and not manifests[key].has_test:
            errors.append(f"eval set {key} has no query+gallery rows")
    return errors


# ---------------------------------------------------------------- near-duplicate check


def dhash(path: Path, size: int = 8) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    return int(np.packbits(bits).view(">u8")[0])


def _popcount64(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    return np.unpackbits(values.view(np.uint8).reshape(*values.shape, 8), axis=-1).sum(-1)


def near_duplicates(train_paths: list[Path], test_paths: list[Path], threshold: int = 4, chunk: int = 512) -> int:
    """Count test images whose dHash is within ``threshold`` bits of any train image."""
    if not train_paths or not test_paths:
        return 0
    train = np.array([dhash(p) for p in train_paths], dtype=np.uint64)
    test = np.array([dhash(p) for p in test_paths], dtype=np.uint64)
    hits = 0
    for start in range(0, len(test), chunk):
        block = test[start : start + chunk, None] ^ train[None, :]
        hits += int((_popcount64(block).min(axis=1) <= threshold).sum())
    return hits


def cross_dataset_duplicates(
    pool: PoolConfig,
    manifests: dict[str, Manifest],
    sample: int = 3000,
    threshold: int = 4,
    seed: int = 0,
) -> list[dict]:
    """dHash near-duplicate check for known-overlap pairs (train of A vs query/gallery of B)."""
    rng = random.Random(seed)
    eval_sets = set(pool.val_sets) | set(pool.test_sets) | set(pool.ood)
    results = []
    for train_key, test_key, reason in KNOWN_OVERLAPS:
        if train_key not in pool.sources or test_key not in eval_sets:
            continue
        if train_key not in manifests or test_key not in manifests:
            continue
        a, b = manifests[train_key], manifests[test_key]
        train_rows = rng.sample(a.train, min(sample, len(a.train)))
        test_rows = rng.sample(b.query + b.gallery, min(sample, len(b.query) + len(b.gallery)))
        hits = near_duplicates([a.resolve(r) for r in train_rows], [b.resolve(r) for r in test_rows], threshold)
        results.append(
            {
                "train": train_key,
                "test": test_key,
                "reason": reason,
                "sampled_train": len(train_rows),
                "sampled_test": len(test_rows),
                "near_duplicates": hits,
                "threshold_bits": threshold,
            }
        )
    return results
