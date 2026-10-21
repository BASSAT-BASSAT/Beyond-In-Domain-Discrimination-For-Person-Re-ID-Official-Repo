"""Per-dataset train/val split policies (distinct from held-out test benchmarks).

Official query/gallery test protocols stay in each dataset manifest; this module only
partitions **training identities** into train vs in-training validation without leakage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from reidfm.data.manifest import Manifest

PolicyKind = Literal[
    "dhash_cluster",
    "identity_disjoint",
    "orbench_official_train",
    "teammate_or_dhash",
]

# Pool sources waiting for optional splits/datasets/<key>.json (fallback: dhash cluster).
TEAMMATE_SPLIT_SOURCES: tuple[str, ...] = (
    "cuhk03np",
    "last",
    "occposetrack",
    "deepchange",
    "lpw",
    "iust",
    "mot17",
    "mot20",
)

CUSTOM_VAL_LABEL = "reidfm_identity_disjoint_val"


@dataclass(frozen=True)
class SplitPolicy:
    kind: PolicyKind
    label: str
    notes: str


DATASET_SPLIT_POLICIES: dict[str, SplitPolicy] = {
    "orbenchrgb": SplitPolicy(
        kind="orbench_official_train",
        label="orbench_official_train_annos",
        notes=(
            "Training identities come only from official train_annos.json (600 IDs). "
            "No official validation split; val IDs are a custom identity holdout "
            f"({CUSTOM_VAL_LABEL}). Official test uses test_gallery_and_queries.json."
        ),
    ),
    "prid2011": SplitPolicy(
        kind="identity_disjoint",
        label="prid2011_multishot_train_identity_disjoint",
        notes=(
            "Official evaluation is cam_a query vs cam_b gallery (fixed 200 shared IDs). "
            "Pool training uses multi-shot train rows (tr-prefixed pids). "
            f"Val is {CUSTOM_VAL_LABEL} over train pids only; test query/gallery never enter train/val."
        ),
    ),
    "rpifield": SplitPolicy(
        kind="identity_disjoint",
        label="rpifield_identity_disjoint",
        notes=(
            "No official train/val/test partition in the original release. "
            f"Train/val use {CUSTOM_VAL_LABEL} at identity level (not random images). "
            "Temporal retrieval protocol is not used for pool fine-tuning val."
        ),
    ),
}

DEFAULT_POLICY = SplitPolicy(
    kind="dhash_cluster",
    label="reidfm_dhash_cluster_val",
    notes="Custom val via perceptual-hash clustering over training identities (leakage-aware).",
)

for _key in TEAMMATE_SPLIT_SOURCES:
    DATASET_SPLIT_POLICIES[_key] = SplitPolicy(
        kind="teammate_or_dhash",
        label=f"{_key}_teammate_or_dhash",
        notes=(
            f"Optional teammate file splits/datasets/{_key}.json (train/val pid lists). "
            "Until present, falls back to reidfm_dhash_cluster_val."
        ),
    )


def get_split_policy(dataset_key: str) -> SplitPolicy:
    return DATASET_SPLIT_POLICIES.get(dataset_key, DEFAULT_POLICY)


def train_pids_from_manifest(manifest: Manifest) -> list[str]:
    return sorted({row.pid for row in manifest.train})


def identity_disjoint_train_val(
    pids: list[str],
    *,
    train_fraction: float,
    seed: int,
    min_val: int = 1,
) -> tuple[list[str], list[str], dict]:
    """Hold out whole identities for validation (never split images of the same pid)."""
    unique = sorted(set(pids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    if len(unique) <= 1:
        return unique, [], {"val_ids": 0, "strategy": "identity_disjoint", "custom_val": CUSTOM_VAL_LABEL}
    n_val = max(min_val, int(round((1.0 - train_fraction) * len(unique))))
    n_val = min(n_val, len(unique) - 1)
    val_set = set(unique[-n_val:])
    train = [p for p in unique if p not in val_set]
    val = [p for p in unique if p in val_set]
    return train, val, {
        "strategy": "identity_disjoint",
        "custom_val": CUSTOM_VAL_LABEL,
        "train_fraction": train_fraction,
        "n_train_ids": len(train),
        "n_val_ids": len(val),
    }


def policy_summary(pool_sources: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in pool_sources:
        pol = get_split_policy(key)
        out[key] = {"kind": pol.kind, "label": pol.label, "notes": pol.notes}
    return out
