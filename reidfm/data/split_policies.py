"""Per-dataset train/val split policies (distinct from held-out test benchmarks).

``pool.split_mode`` selects the policy table:

- ``cluster`` (default, paper loss study): dHash / identity holdout + ``poolv2val``
- ``official_native``: all ``manifest.train`` IDs; in-training eval via ``pool.val_sets``
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from reidfm.data.manifest import Manifest

PolicyKind = Literal[
    "dhash_cluster",
    "identity_disjoint",
    "official_native",
    "mot_train_only",
    "train_only_no_eval",
    "structured_or_dhash",
    "teammate_or_dhash",
]

SplitMode = Literal["cluster", "official_native"]

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


_OFFICIAL_NATIVE = SplitPolicy(
    kind="official_native",
    label="official_train_official_eval_val",
    notes=(
        "Train on all manifest.train identities. In-training validation uses the dataset's "
        "official query/gallery (pool.val_sets), not a random identity holdout from train."
    ),
)

_MOT_TRAIN_ONLY = SplitPolicy(
    kind="mot_train_only",
    label="mot_train_sequences_only",
    notes="All MOT train-sequence identities; no public test GT for in-training ReID eval.",
)

OFFICIAL_NATIVE_POLICIES: dict[str, SplitPolicy] = {
    "cuhk03np": _OFFICIAL_NATIVE,
    "last": _OFFICIAL_NATIVE,
    "occposetrack": _OFFICIAL_NATIVE,
    "deepchange": _OFFICIAL_NATIVE,
    "lpw": _OFFICIAL_NATIVE,
    "orbenchrgb": _OFFICIAL_NATIVE,
    "iust": _OFFICIAL_NATIVE,
    "prid2011": _OFFICIAL_NATIVE,
    "mot17": SplitPolicy(
        kind="structured_or_dhash",
        label="mot_camera_scene_or_dhash_80_20",
        notes="No public test GT: ~80/20 by sequence (camera) → identity → dHash; val via poolv2val.",
    ),
    "mot20": SplitPolicy(
        kind="structured_or_dhash",
        label="mot_camera_scene_or_dhash_80_20",
        notes="Same as MOT17.",
    ),
    "rpifield": SplitPolicy(
        kind="structured_or_dhash",
        label="rpifield_camera_or_dhash_80_20",
        notes=(
            "~80/20 by folder camera/scene → identity → dHash when no official query/gallery."
        ),
    ),
}

CLUSTER_POLICIES: dict[str, SplitPolicy] = {
    "orbenchrgb": SplitPolicy(
        kind="identity_disjoint",
        label="orbench_official_train_annos",
        notes=(
            "Training identities from official train_annos only; "
            f"custom val = {CUSTOM_VAL_LABEL} on those train IDs."
        ),
    ),
    "prid2011": SplitPolicy(
        kind="identity_disjoint",
        label="prid2011_multishot_train_identity_disjoint",
        notes=(
            "Pool train uses tr* rows; official cam_a/cam_b test stays out of train. "
            f"Val = {CUSTOM_VAL_LABEL} on tr* IDs."
        ),
    ),
    "rpifield": SplitPolicy(
        kind="identity_disjoint",
        label="rpifield_identity_disjoint",
        notes=f"No official split; {CUSTOM_VAL_LABEL} on train-folder IDs.",
    ),
}

DEFAULT_POLICY = SplitPolicy(
    kind="dhash_cluster",
    label="reidfm_dhash_cluster_val",
    notes="Custom val via perceptual-hash clustering over training identities (leakage-aware).",
)

for _key in TEAMMATE_SPLIT_SOURCES:
    CLUSTER_POLICIES[_key] = SplitPolicy(
        kind="teammate_or_dhash",
        label=f"{_key}_teammate_or_dhash",
        notes=(
            f"Optional splits/datasets/{_key}.json; else dhash cluster val."
        ),
    )


def get_split_policy(dataset_key: str, split_mode: SplitMode = "cluster") -> SplitPolicy:
    if split_mode == "official_native":
        return OFFICIAL_NATIVE_POLICIES.get(dataset_key, DEFAULT_POLICY)
    return CLUSTER_POLICIES.get(dataset_key, DEFAULT_POLICY)


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


def policy_summary(pool_sources: list[str], split_mode: SplitMode = "cluster") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in pool_sources:
        pol = get_split_policy(key, split_mode)
        out[key] = {"kind": pol.kind, "label": pol.label, "notes": pol.notes}
    return out
