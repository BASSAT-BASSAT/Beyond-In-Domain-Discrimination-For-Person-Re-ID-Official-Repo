"""Fail-fast checks for study / fair-baseline training."""

from __future__ import annotations

import json
from pathlib import Path

from reidfm.data.pool import PoolConfig, load_pool
from reidfm.data.split_filter import load_train_split
from reidfm.experiments.contract import HELD_OUT_TEST_KEYS, IMAGENET_INIT_MODELS, PRIMARY_INIT_MODEL
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



def validate_v3_study_protocol(
    cfg,
    *,
    require_imagenet_init: bool = True,
) -> dict:
    """Validate the pinned V3 structural study contract.

    This does not establish scientific P4 clearance,
    model readiness, or evaluation correctness.
    """
    import csv
    import hashlib

    protocol_id = str(cfg.REIDFM.V3_PROTOCOL_ID)

    expected = {
        "dg_reid_global_p3_p5_v1": {
            "train": "40a5d4d66363c3300b801f8cf79b96cd84b4b44c9b22bf8d7f5e455bc89aa7a6",
            "validation": "55c3de1fe53960daef0a30e13961f926794beafc0f89bd4376261daa21630033",
        }
    }

    if protocol_id not in expected:
        raise ValueError(
            f"Unsupported V3 protocol: {protocol_id}"
        )

    if require_imagenet_init:
        if cfg.REIDFM.INIT_MODEL not in IMAGENET_INIT_MODELS:
            raise ValueError(
                "V3 loss study requires ImageNet initialization "
                f"from one of {sorted(IMAGENET_INIT_MODELS)}"
            )

    def resolve_path(value):
        path = Path(str(value))
        return path if path.is_absolute() else repo_path(path)

    train_path = resolve_path(
        cfg.REIDFM.V3_TRAIN_MANIFEST
    )

    val_path = resolve_path(
        cfg.REIDFM.V3_VAL_MANIFEST
    )

    pack_root = resolve_path(
        cfg.REIDFM.V3_PACK_ROOT
    )

    if not train_path.is_file():
        raise FileNotFoundError(
            f"V3 train manifest missing: {train_path}"
        )

    if not val_path.is_file():
        raise FileNotFoundError(
            f"V3 validation manifest missing: {val_path}"
        )

    if not pack_root.is_dir():
        raise FileNotFoundError(
            f"V3 PACK root missing: {pack_root}"
        )

    if train_path.resolve() == val_path.resolve():
        raise ValueError(
            "V3 train and validation manifests must differ"
        )

    def sha256_file(path):
        digest = hashlib.sha256()

        with path.open("rb") as stream:
            while True:
                block = stream.read(4 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)

        return digest.hexdigest()

    actual_train_hash = sha256_file(train_path)
    actual_val_hash = sha256_file(val_path)

    if actual_train_hash != expected[protocol_id]["train"]:
        raise ValueError(
            "V3 training manifest SHA256 mismatch"
        )

    if actual_val_hash != expected[protocol_id]["validation"]:
        raise ValueError(
            "V3 validation manifest SHA256 mismatch"
        )

    def fields(path):
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            return set(csv.DictReader(stream).fieldnames or [])

    train_fields = fields(train_path)
    val_fields = fields(val_path)

    required_train = {
        "sample_uid",
        "dataset_root_id",
        "relative_path",
        "identity_key",
        "class_id",
    }

    required_val = {
        "sample_uid",
        "dataset_root_id",
        "relative_path",
        "identity_key",
        "validation_protocol_id",
        "retrieval_role",
        "raw_camera_or_view",
        "camera_semantics",
    }

    if not required_train.issubset(train_fields):
        raise ValueError(
            "V3 training manifest schema mismatch"
        )

    if not required_val.issubset(val_fields):
        raise ValueError(
            "V3 validation manifest schema mismatch"
        )

    if "class_id" in val_fields:
        raise ValueError(
            "V3 retrieval validation must not use training class IDs"
        )

    return {
        "status": "STRUCTURAL_CONTRACT_PASS",
        "protocol_id": protocol_id,
        "train_sha256": actual_train_hash,
        "validation_sha256": actual_val_hash,
        "scientific_clearance": "NOT_ASSESSED",
        "model_readiness": "NOT_ASSESSED",
        "evaluation_readiness": "NOT_ASSESSED",
    }


def validate_study_protocol(cfg, *, require_imagenet_init: bool = True) -> None:
    backend = getattr(cfg.REIDFM, "DATA_FORMAT", "LEGACY")

    if backend == "REIDPACK3":
        if not (
            getattr(cfg.REIDFM, "STUDY_MODE", False)
            or getattr(cfg.REIDFM, "STRICT_SPLIT", False)
        ):
            return

        validate_v3_study_protocol(
            cfg,
            require_imagenet_init=require_imagenet_init,
        )
        return

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

    if str(getattr(cfg.REIDFM, "DATA_FORMAT", "LEGACY")) == "REIDPACK3":
        # V3 fingerprint: never resolve Legacy Pool V2.
        import math
        from pathlib import Path
        from reidfm.config import parse_domain_weights

        def canonical(value):
            if hasattr(value, "items"):
                return {
                    str(k): canonical(v)
                    for k, v in sorted(value.items())
                }
            if isinstance(value, (list, tuple)):
                return [canonical(v) for v in value]
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(
                        "V3 fingerprint contains non-finite config float"
                    )
                return value
            if value is None or isinstance(value, (str, int, bool)):
                return value
            raise TypeError(
                f"Unsupported V3 fingerprint value type: {type(value)!r}"
            )

        def digest_file(raw_path, name):
            path = Path(raw_path)

            if not path.is_file():
                raise FileNotFoundError(
                    f"V3 fingerprint input {name} missing: {path}"
                )

            hasher = hashlib.sha256()

            with path.open("rb") as stream:
                for block in iter(
                    lambda: stream.read(8 * 1024 * 1024),
                    b""
                ):
                    hasher.update(block)

            return hasher.hexdigest()

        num_classes = int(
            cfg.MODEL.HEADS.NUM_CLASSES
        )

        if num_classes <= 0:
            raise ValueError(
                "V3 fingerprint requires positive MODEL.HEADS.NUM_CLASSES"
            )

        domain_weights = parse_domain_weights(
            cfg.REIDFM.DOMAIN_WEIGHTS
        )

        for domain, multiplier in domain_weights.items():
            if not math.isfinite(multiplier) or multiplier < 0:
                raise ValueError(
                    f"Invalid V3 domain multiplier: {domain}={multiplier}"
                )

        train_path = Path(
            cfg.REIDFM.V3_TRAIN_MANIFEST
        )

        val_path = Path(
            cfg.REIDFM.V3_VAL_MANIFEST
        )

        pack_root = Path(
            cfg.REIDFM.V3_PACK_ROOT
        )

        if not pack_root.is_dir():
            raise FileNotFoundError(
                f"V3 PACK root missing: {pack_root}"
            )

        repository = Path(__file__).resolve().parents[2]

        source_paths = (
            "reidfm/config.py",
            "reidfm/experiments/protocol.py",
            "reidfm/engine/trainer.py",
            "reidfm/data/v3_pack_reader.py",
            "reidfm/data/v3_global_dataset.py",
            "reidfm/data/v3_fastreid_dataset.py",
            "reidfm/data/samplers.py",
        )

        payload = {
            "fingerprint_schema": "reidfm_v3_study_fingerprint_v1",
            "data_format": "REIDPACK3",
            "protocol_id": str(cfg.REIDFM.V3_PROTOCOL_ID),

            "train_manifest_sha256": digest_file(
                train_path,
                "train"
            ),

            "validation_manifest_sha256": digest_file(
                val_path,
                "validation"
            ),

            "source_sha256": {
                name: digest_file(repository / name, name)
                for name in source_paths
            },

            "pack_root": str(pack_root.resolve()),
            "seed": int(cfg.REIDFM.SEED),
            "init_model": str(cfg.REIDFM.INIT_MODEL),

            "iters_per_epoch": int(
                cfg.REIDFM.ITERS_PER_EPOCH
            ),

            "num_classes": num_classes,

            "sampler_weight_mode": str(
                cfg.REIDFM.SAMPLER_WEIGHT_MODE
            ),

            "domain_weights": {
                k: float(v)
                for k, v in sorted(domain_weights.items())
            },

            "model": {
                "meta_arch": str(
                    cfg.MODEL.META_ARCHITECTURE
                ),
                "backbone": canonical(
                    cfg.MODEL.BACKBONE
                ),
                "heads": canonical(
                    cfg.MODEL.HEADS
                ),
                "losses": canonical(
                    cfg.MODEL.LOSSES
                ),
                "pixel_mean": canonical(
                    cfg.MODEL.PIXEL_MEAN
                ),
                "pixel_std": canonical(
                    cfg.MODEL.PIXEL_STD
                ),
                "freeze_layers": canonical(
                    cfg.MODEL.FREEZE_LAYERS
                ),
            },

            "solver": canonical(cfg.SOLVER),
            "input": canonical(cfg.INPUT),
            "dataloader": canonical(cfg.DATALOADER),
            "test": canonical(cfg.TEST),

            "evaluation": {
                "val_max_queries": int(
                    cfg.REIDFM.VAL_MAX_QUERIES
                ),
                "eval_num_frames": int(
                    cfg.REIDFM.EVAL_NUM_FRAMES
                ),
            },
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()

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
