"""Official train + official query/gallery as in-training validation (pool v2 native mode).

Training uses every ``manifest.train`` identity. In-training eval uses ``pool.val_sets``
(FastReID ``DATASETS.TESTS``), not ``pool_v2_val_ids.json`` identity holdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from reidfm.data.manifest import Manifest
from reidfm.data.split_policies import train_pids_from_manifest

# Sources with query/gallery in the converter manifest (``pool.val_sets``).
OFFICIAL_IN_TRAINING_VAL_KEYS: tuple[str, ...] = (
    "cuhk03np",
    "last",
    "occposetrack",
    "deepchange",
    "lpw",
    "orbenchrgb",
    "iust",
    "prid2011",
)

# MOT train sequences only (no public ReID test GT).
MOT_TRAIN_ONLY_KEYS: tuple[str, ...] = ("mot17", "mot20")

# RPIfield: no standard query/gallery in default pool layout — keep identity_disjoint holdout.
RPIFIELD_KEY = "rpifield"


@dataclass(frozen=True)
class SplitRatioReport:
    dataset: str
    n_train_ids: int
    n_train_images: int
    n_eval_ids: int
    n_query_images: int
    n_gallery_images: int
    id_ratio_train: float | None  # train_ids / (train_ids + eval_ids)
    image_ratio_train: float | None  # train_images / all_images
    in_training_val: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def eval_pids_from_manifest(manifest: Manifest) -> set[str]:
    pids: set[str] = set()
    for row in (*manifest.query, *manifest.gallery):
        pids.add(row.pid)
    return pids


def manifest_ratio_report(key: str, manifest: Manifest, *, in_training_val: bool, notes: str = "") -> SplitRatioReport:
    train_ids = train_pids_from_manifest(manifest)
    eval_ids = eval_pids_from_manifest(manifest)
    n_train_img = len(manifest.train)
    n_q = len(manifest.query)
    n_g = len(manifest.gallery)
    n_all = n_train_img + n_q + n_g
    id_denom = len(train_ids) + len(eval_ids)
    return SplitRatioReport(
        dataset=key,
        n_train_ids=len(train_ids),
        n_train_images=n_train_img,
        n_eval_ids=len(eval_ids),
        n_query_images=n_q,
        n_gallery_images=n_g,
        id_ratio_train=(len(train_ids) / id_denom) if id_denom else None,
        image_ratio_train=(n_train_img / n_all) if n_all else None,
        in_training_val=in_training_val,
        notes=notes,
    )


# Reference splits when manifests are not materialized yet (literature / converter defaults).
CANONICAL_REFERENCE_RATIOS: dict[str, dict[str, Any]] = {
    "cuhk03np": {
        "protocol": "CUHK03-NP detected, Market-style folders",
        "n_train_ids": 767,
        "n_query_images": 140,
        "n_gallery_images": 700,
        "n_eval_ids": 700,
        "id_ratio_train": 767 / (767 + 700),
        "notes": "Distinct person IDs in gallery vs train; query is a subset of gallery IDs.",
    },
    "last": {
        "protocol": "LaST official train + test (pool uses eval_split=test for val)",
        "n_train_ids": "~12.6k visual IDs (see paper)",
        "id_ratio_train": "high (test is smaller than train)",
        "notes": "Set converters.last.eval_split to val if you want to reserve official test for a final run.",
    },
    "occposetrack": {
        "protocol": "train/ + val|test/ (query/gallery or reidfm synthetic)",
        "notes": "If only a flat eval folder exists, converter builds query/gallery per ID.",
    },
    "deepchange": {
        "protocol": "train-set vs test-set-query / test-set-gallery",
        "notes": "Verify folder names on your copy (pattern_dirs).",
    },
    "lpw": {
        "protocol": "scen2+scen3 train; scen1 test (view2 query, other views gallery)",
        "train_scenes": ["scen2", "scen3"],
        "test_scene": "scen1",
    },
    "orbenchrgb": {
        "protocol": "600 train_annos IDs; test_gallery_and_queries.json for eval",
        "n_train_ids": 600,
        "notes": "Test pids are namespaced with test prefix in the manifest.",
    },
    "iust": {
        "protocol": "Market-1501 style (bounding_box_train / query / bounding_box_test)",
        "notes": "Same layout as Market; restricted campus dataset.",
    },
    "prid2011": {
        "protocol": "tr* multi-shot train (200 IDs); cam_a query vs cam_b gallery (200 IDs)",
        "n_train_ids": 200,
        "n_eval_ids": 200,
        "id_ratio_train": 0.5,
        "notes": "Train and eval ID namespaces are disjoint (tr1 vs 1).",
    },
    "mot17": {
        "protocol": "MOTChallenge train sequences only (FRCNN detections in pool)",
        "in_training_val": False,
        "notes": "No public test GT; use cluster holdout on MOT train IDs or skip MOT in EvalHook.",
    },
    "mot20": {
        "protocol": "MOT20 train sequences only",
        "in_training_val": False,
        "notes": "Same as MOT17.",
    },
    "rpifield": {
        "protocol": "No official ReID split in pool default (images/ train only)",
        "in_training_val": False,
        "notes": "Uses identity_disjoint ~85/15 on train IDs unless you add query/gallery folders to the converter.",
    },
}
