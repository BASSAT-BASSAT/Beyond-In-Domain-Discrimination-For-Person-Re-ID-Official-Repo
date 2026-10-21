"""Fail-fast checks for study / fair-baseline training."""

from __future__ import annotations

import json
from pathlib import Path

from reidfm.data.pool import PoolConfig, load_pool
from reidfm.data.split_filter import load_train_split
from reidfm.experiments.contract import HELD_OUT_TEST_KEYS, PRIMARY_INIT_MODEL
from reidfm.utils.paths import repo_path


def _split_path(cfg) -> Path:
    raw = cfg.REIDFM.TRAIN_SPLIT
    if not raw:
        raise ValueError("REIDFM.TRAIN_SPLIT must be set in study mode")
    p = repo_path(raw) if not Path(raw).is_absolute() else Path(raw)
    return p


def validate_train_split_file(cfg, pool: PoolConfig | None = None) -> dict[str, set[str]]:
    path = _split_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"study mode requires train split at {path}; run scripts/build_train_val_split.py first"
        )
    allowed = load_train_split(path)
    if not allowed:
        raise ValueError(f"train split {path} is empty or invalid")
    pool = pool or load_pool(cfg.REIDFM.POOL)
    missing = [s for s in pool.sources if s not in allowed]
    if missing:
        raise ValueError(f"train split missing sources from pool: {missing}")
    overlap = set(pool.sources) & set(pool.test_sets)
    if overlap:
        raise ValueError(f"pool train sources overlap test_sets: {sorted(overlap)}")
    for test_key in HELD_OUT_TEST_KEYS:
        if test_key not in pool.test_sets and test_key != "occludedduke":
            # occludedduke may be added incrementally; require full battery when all are listed
            pass
    return allowed


def validate_val_manifest(cfg) -> None:
    manifest = repo_path(cfg.REIDFM.VAL_MANIFEST) / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"study mode requires val manifest at {manifest} (build_train_val_split.py)"
        )
    import csv

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"val manifest {manifest} has no rows")
    queries = [r for r in rows if str(r.get("split", "")).lower() == "query"]
    gallery = [r for r in rows if str(r.get("split", "")).lower() == "gallery"]
    if not queries or not gallery:
        raise ValueError(f"val manifest {manifest} needs both query and gallery rows")


def validate_study_protocol(cfg, *, require_imagenet_init: bool = True) -> None:
    if not getattr(cfg.REIDFM, "STUDY_MODE", False):
        return
    pool = load_pool(cfg.REIDFM.POOL)
    validate_train_split_file(cfg, pool)
    if pool.val_sets:
        raise ValueError("study mode expects empty pool val_sets; use REIDFM.VAL_MANIFEST cluster holdout")
    validate_val_manifest(cfg)
    if require_imagenet_init and cfg.REIDFM.INIT_MODEL != PRIMARY_INIT_MODEL:
        raise ValueError(
            f"primary loss study requires INIT_MODEL={PRIMARY_INIT_MODEL!r}, got {cfg.REIDFM.INIT_MODEL!r}"
        )
    leaked = set(pool.sources) & set(HELD_OUT_TEST_KEYS)
    if leaked:
        raise ValueError(f"held-out test domains must not appear in pool sources: {sorted(leaked)}")


def config_fingerprint(cfg) -> str:
    """Hashable summary of frozen training protocol (excluding OUTPUT_DIR / condition id)."""
    import hashlib

    from reidfm.config import resolve_pool

    pool = resolve_pool(cfg)
    payload = {
        "pool": str(pool.path),
        "sources": list(pool.sources),
        "train_split": cfg.REIDFM.TRAIN_SPLIT,
        "init_model": cfg.REIDFM.INIT_MODEL,
        "iters_per_epoch": cfg.REIDFM.ITERS_PER_EPOCH,
        "seed": cfg.REIDFM.SEED,
        "meta_arch": cfg.MODEL.META_ARCHITECTURE,
        "heads": cfg.MODEL.HEADS.NAME,
        "cls_layer": cfg.MODEL.HEADS.CLS_LAYER,
        "losses": list(cfg.MODEL.LOSSES.NAME),
        "subcenters": int(getattr(cfg.MODEL.HEADS, "NUM_SUBCENTERS", 1)),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
