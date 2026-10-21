"""Frozen scientific contract for the angular-loss study and fair CNN appendix baselines."""

from __future__ import annotations

from typing import FrozenSet

# Primary table: nine SBS loss conditions (ImageNet init, pool v2).
LOSS_STUDY_CONDITIONS: tuple[str, ...] = (
    "R0",
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
)

PRIMARY_INIT_MODEL = "sbs_r50_ibn_imagenet"

HELD_OUT_TEST_KEYS: tuple[str, ...] = (
    "market1501",
    "msmt17",
    "dukemtmc",
    "occludedduke",
    "mars",
)

FAIR_BASELINE_KEYS: tuple[str, ...] = ("bot_r50_ibn", "agw_r50_ibn")

# Config keys that may differ between loss-study conditions (everything else must match base).
LOSS_STUDY_WHITELIST: FrozenSet[str] = frozenset(
    {
        "MODEL.META_ARCHITECTURE",
        "MODEL.HEADS.NAME",
        "MODEL.HEADS.CLS_LAYER",
        "MODEL.HEADS.SCALE",
        "MODEL.HEADS.MARGIN",
        "MODEL.HEADS.NUM_SUBCENTERS",
        "MODEL.LOSSES.NAME",
        "MODEL.LOSSES.SPHERE.SCALE",
        "MODEL.LOSSES.SPHERE.MARGIN",
        "MODEL.LOSSES.SPHERE.LAMBDA",
        "MODEL.LOSSES.SPHERE.GAMMA",
        "OUTPUT_DIR",
        "REIDFM.LOSS_STUDY_CONDITION",
        "REIDFM.STUDY_MODE",
    }
)
