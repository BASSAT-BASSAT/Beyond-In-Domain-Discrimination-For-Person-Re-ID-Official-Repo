"""FastReID config + the ``REIDFM`` node, and the single place that builds run configs.

``setup_cfg`` applies, in order: the yaml, command-line opts, env expansion, the pool
(``DATASETS.NAMES``/``TESTS``), the init model from the registry, and the output dir.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from reidfm.utils.config import expand_env
from reidfm.utils.paths import output_root, repo_path

logger = logging.getLogger(__name__)


def add_reidfm_config(cfg) -> None:
    from fastreid.config import CfgNode as CN

    cfg.REIDFM = CN()
    # Pool yaml (configs/datasets/*.yaml): sources -> DATASETS.NAMES, val_sets -> DATASETS.TESTS.
    cfg.REIDFM.POOL = "configs/datasets/pool_v2.yaml"
    # Data backend: legacy Pool V2 or packed Global P3/P5.
    cfg.REIDFM.DATA_FORMAT = "LEGACY"
    cfg.REIDFM.V3_TRAIN_MANIFEST = ""
    cfg.REIDFM.V3_VAL_MANIFEST = ""
    cfg.REIDFM.V3_PACK_ROOT = ""
    cfg.REIDFM.V3_PROTOCOL_ID = "dg_reid_global_p3_p5_v1"
    # Non-empty lists override the pool's sources / val_sets (still validated against its OOD sets).
    cfg.REIDFM.SOURCES = []
    cfg.REIDFM.VAL_SETS = []
    # DomainBalancedIdentitySampler: "" = pool's weight_mode; entries "key=multiplier" update the
    # pool's domain_weights.
    cfg.REIDFM.SAMPLER_WEIGHT_MODE = ""
    cfg.REIDFM.DOMAIN_WEIGHTS = []
    # Global RNG seed; < 0 lets FastReID pick one.
    cfg.REIDFM.SEED = -1
    # Registry key of the starting point (scripts/list_models.py). Empty = use MODEL.WEIGHTS as is.
    cfg.REIDFM.INIT_MODEL = ""
    # Stop cleanly (and checkpoint) once this many hours have elapsed; 0 disables.
    # $REIDFM_TIME_BUDGET_H, when set, overrides the yaml value (see load_cfg).
    cfg.REIDFM.TIME_BUDGET_H = 11.0
    # Stop when the remaining budget < slowest epoch * this factor.
    cfg.REIDFM.TIME_GUARD_SAFETY = 1.25
    # >0: fixed "virtual" epoch length in iterations (pool epochs are huge; Kaggle needs granularity).
    cfg.REIDFM.ITERS_PER_EPOCH = 0
    # Keep this many model_XXXX.pth epoch checkpoints (model_best/model_final/model_last are never pruned).
    cfg.REIDFM.KEEP_CHECKPOINTS = 2
    # If OUTPUT_DIR already has last_checkpoint, train.py resumes without requiring --resume.
    cfg.REIDFM.AUTO_RESUME = True
    # Evaluation: frames per tracklet for video sets, query cap for in-training val (0 = all).
    cfg.REIDFM.EVAL_NUM_FRAMES = 8
    cfg.REIDFM.VAL_MAX_QUERIES = 0
    cfg.REIDFM.EVAL_AMP = True
    cfg.REIDFM.EVAL_WORKERS = 4
    # Identity splits from scripts/build_train_val_split.py (empty = use all train rows).
    cfg.REIDFM.TRAIN_SPLIT = "splits/pool_v2_train_ids.json"
    # Folder with manifest.csv for in-training val when pool val_sets is empty.
    cfg.REIDFM.VAL_MANIFEST = "pool/poolv2val"
    # Loss-study protocol. Checkpoints are chosen on the source identity holdout.
    cfg.REIDFM.STUDY_MODE = False
    cfg.REIDFM.STRICT_SPLIT = False
    cfg.REIDFM.LOSS_STUDY_CONDITION = ""
    cfg.MODEL.HEADS.NUM_SUBCENTERS = 1
    # SphereFace2-C defaults from the OpenSphere reference head: radius r, margin m,
    # positive/negative balance lambda, similarity adjustment t, loss weight lw.
    cfg.MODEL.LOSSES.SPHERE = CN()
    cfg.MODEL.LOSSES.SPHERE.SCALE = 40.0
    cfg.MODEL.LOSSES.SPHERE.MARGIN = 0.4
    cfg.MODEL.LOSSES.SPHERE.LAMBDA = 0.9
    cfg.MODEL.LOSSES.SPHERE.T = 3.0
    cfg.MODEL.LOSSES.SPHERE.LOSS_WEIGHT = 50.0


def get_reidfm_cfg():
    from reidfm.fastreid_compat import ensure_fastreid

    ensure_fastreid()
    from fastreid.config import get_cfg

    cfg = get_cfg()
    add_reidfm_config(cfg)
    return cfg


def _expand_cfg(node) -> None:
    for key, value in node.items():
        if isinstance(value, str) and "${" in value:
            node[key] = expand_env(value)
        elif hasattr(value, "items"):
            _expand_cfg(value)


def load_cfg(config_file: str | Path, opts: list[str] | None = None):
    import yaml

    path = repo_path(config_file)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("_BASE_CONFIG", None)
    cfg = get_reidfm_cfg()
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            candidate = (path.parent / base).resolve()
            base_path = candidate if candidate.is_file() else repo_path(base)
        cfg.merge_from_file(str(base_path))
    if data:
        overlay = path.parent / f".{path.stem}.overlay.yml"
        overlay.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        try:
            cfg.merge_from_file(str(overlay))
        finally:
            overlay.unlink(missing_ok=True)
    elif not base:
        cfg.merge_from_file(str(path))
    if os.environ.get("REIDFM_TIME_BUDGET_H"):
        cfg.REIDFM.TIME_BUDGET_H = float(os.environ["REIDFM_TIME_BUDGET_H"])
    if opts:
        cfg.merge_from_list(list(opts))
    _expand_cfg(cfg)
    return cfg


def parse_domain_weights(entries) -> dict[str, float]:
    weights = {}
    for entry in entries:
        key, sep, value = str(entry).partition("=")
        if not sep:
            raise ValueError(f"REIDFM.DOMAIN_WEIGHTS entries must be 'key=multiplier', got {entry!r}")
        weights[key.strip()] = float(value)
    return weights


def resolve_pool(cfg, pool_path: str | Path | None = None):
    """The pool yaml with the ``REIDFM`` source / val / sampler overrides applied."""
    from dataclasses import replace

    from reidfm.data.pool import load_pool

    pool = load_pool(pool_path or cfg.REIDFM.POOL)
    overrides = {}
    if cfg.REIDFM.SOURCES:
        overrides["sources"] = list(cfg.REIDFM.SOURCES)
    if cfg.REIDFM.VAL_SETS:
        overrides["val_sets"] = list(cfg.REIDFM.VAL_SETS)
    if cfg.REIDFM.SAMPLER_WEIGHT_MODE:
        overrides["weight_mode"] = cfg.REIDFM.SAMPLER_WEIGHT_MODE
    sources = overrides.get("sources", pool.sources)
    weights = {k: v for k, v in pool.domain_weights.items() if k in sources}
    weights.update(parse_domain_weights(cfg.REIDFM.DOMAIN_WEIGHTS))
    overrides["domain_weights"] = weights
    pool = replace(pool, **overrides)
    pool.validate()
    return pool


def apply_pool(cfg, pool_path: str | Path | None = None):
    pool = resolve_pool(cfg, pool_path)
    cfg.REIDFM.POOL = str(pool.path)
    cfg.DATASETS.NAMES = tuple(pool.fastreid_names(pool.sources))
    tests: list[str] = []
    if pool.val_sets:
        tests.extend(pool.fastreid_names(pool.val_sets))
    val_manifest = repo_path(cfg.REIDFM.VAL_MANIFEST) / "manifest.csv" if cfg.REIDFM.VAL_MANIFEST else None
    if val_manifest and val_manifest.is_file():
        if "PoolV2ValM" not in tests:
            tests.append("PoolV2ValM")
    elif cfg.REIDFM.VAL_MANIFEST and not pool.val_sets:
        tests.append("PoolV2ValM")
    cfg.DATASETS.TESTS = tuple(tests)
    return pool


def apply_init_model(cfg, model_key: str | None = None, prepare_weights: bool = True):
    """Point ``MODEL.WEIGHTS``/``PRETRAIN`` at a registered starting point."""
    from reidfm.models.init import architecture_diff, prepare_init_weights, spec_cfg
    from reidfm.models.registry import get_model_spec

    key = model_key or cfg.REIDFM.INIT_MODEL
    if not key:
        return None
    spec = get_model_spec(key)
    cfg.REIDFM.INIT_MODEL = key
    if spec.config_path is not None and spec.config_path.is_file():
        diffs = architecture_diff(spec_cfg(spec), cfg)
        if diffs:
            logger.warning("run config differs from %s architecture (those weights start fresh):\n  %s", key, "\n  ".join(diffs))
    if spec.init == "imagenet":
        cfg.MODEL.BACKBONE.PRETRAIN = True
        cfg.MODEL.WEIGHTS = ""
    else:
        cfg.MODEL.BACKBONE.PRETRAIN = False
        if prepare_weights:
            cfg.MODEL.WEIGHTS = str(prepare_init_weights(spec, cfg))
    return spec


def resolve_output_dir(cfg, output_dir: str | None = None) -> str:
    path = Path(output_dir or cfg.OUTPUT_DIR)
    if not path.is_absolute():
        path = output_root() / path
    cfg.OUTPUT_DIR = str(path)
    return cfg.OUTPUT_DIR



def _v3_num_classes_from_manifest(manifest_path: str | Path) -> int:
    """Validate Global train identity/class mapping and infer dense classifier size."""
    import csv

    path = Path(manifest_path)

    if not path.is_file():
        raise FileNotFoundError(
            f'V3 training manifest missing: {path}'
        )

    class_to_identity = {}
    identity_to_class = {}

    with path.open(
        'r',
        encoding='utf-8-sig',
        newline=''
    ) as stream:

        reader = csv.DictReader(stream)

        required = {
            'class_id',
            'identity_key',
            'dataset_root_id',
        }

        missing = required - set(
            reader.fieldnames or ()
        )

        if missing:
            raise ValueError(
                f'V3 training manifest missing columns: '
                f'{sorted(missing)}'
            )

        for line_no, row in enumerate(
            reader,
            start=2
        ):

            identity = row['identity_key']
            domain = row['dataset_root_id']

            if not identity or not domain:
                raise ValueError(
                    f'V3 training manifest row {line_no}: '
                    f'missing identity/domain'
                )

            try:
                class_id = int(
                    row['class_id']
                )

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'V3 training manifest row {line_no}: '
                    f'invalid class ID'
                ) from exc

            if class_id < 0:
                raise ValueError(
                    f'V3 training manifest row {line_no}: '
                    f'negative class ID'
                )

            old_identity = class_to_identity.setdefault(
                class_id,
                identity
            )

            old_class = identity_to_class.setdefault(
                identity,
                class_id
            )

            if (
                old_identity != identity
                or old_class != class_id
            ):
                raise ValueError(
                    f'V3 training manifest row {line_no}: '
                    f'inconsistent identity/class mapping'
                )

    n_classes = len(class_to_identity)

    if (
        not n_classes
        or set(class_to_identity) != set(range(n_classes))
    ):
        raise ValueError(
            'V3 class IDs must be a nonempty dense range 0..N-1'
        )

    return n_classes


def _apply_v3_class_count(cfg) -> int:
    """Resolve V3 classifier size before model construction; reject conflicts."""

    count = _v3_num_classes_from_manifest(
        cfg.REIDFM.V3_TRAIN_MANIFEST
    )

    configured = int(
        cfg.MODEL.HEADS.NUM_CLASSES
    )

    if configured not in (0, count):
        raise ValueError(
            f'V3 MODEL.HEADS.NUM_CLASSES={configured} '
            f'conflicts with {count} Global training classes'
        )

    cfg.MODEL.HEADS.NUM_CLASSES = count

    return count


def setup_cfg(
    config_file: str | Path,
    opts: list[str] | None = None,
    model_key: str | None = None,
    pool: str | None = None,
    output_dir: str | None = None,
    time_budget_h: float | None = None,
    prepare_weights: bool = True,
    validate_study: bool = True,
):
    cfg = load_cfg(config_file, opts)
    if cfg.REIDFM.DATA_FORMAT == "REIDPACK3":
        if pool is not None:
            raise ValueError(
                "--pool is incompatible with REIDPACK3 mode"
            )
        if not cfg.REIDFM.V3_TRAIN_MANIFEST:
            raise ValueError(
                "REIDFM.V3_TRAIN_MANIFEST is required"
            )
        if not cfg.REIDFM.V3_PACK_ROOT:
            raise ValueError(
                "REIDFM.V3_PACK_ROOT is required"
            )
        if (
            cfg.REIDFM.STUDY_MODE
            or cfg.REIDFM.STRICT_SPLIT
        ) and not cfg.REIDFM.V3_VAL_MANIFEST:
            raise ValueError(
                "REIDFM.V3_VAL_MANIFEST is required "
                "for V3 study validation"
            )
        # V3 uses its own authoritative Global manifest.
        # Do not resolve Pool V2 datasets in this mode.
        cfg.DATASETS.NAMES = ()
        cfg.DATASETS.TESTS = ()
    elif cfg.REIDFM.DATA_FORMAT == "LEGACY":
        apply_pool(cfg, pool)
    else:
        raise ValueError(
            f"Unsupported DATA_FORMAT: {cfg.REIDFM.DATA_FORMAT}"
        )
    if cfg.REIDFM.DATA_FORMAT == "REIDPACK3":
        _apply_v3_class_count(cfg)
    apply_init_model(cfg, model_key, prepare_weights=prepare_weights)
    if time_budget_h is not None:
        cfg.REIDFM.TIME_BUDGET_H = float(time_budget_h)
    resolve_output_dir(cfg, output_dir)
    if validate_study and (cfg.REIDFM.STUDY_MODE or cfg.REIDFM.STRICT_SPLIT):
        from reidfm.experiments.protocol import validate_study_protocol

        validate_study_protocol(
            cfg,
            require_imagenet_init=bool(getattr(cfg.REIDFM, "LOSS_STUDY_CONDITION", "")),
        )
    cfg.freeze()
    return cfg


def setup_eval_cfg(
    config_file: str | Path,
    opts: list[str] | None = None,
    model_key: str | None = None,
    output_dir: str | None = None,
    num_classes: int = 10036,
):
    """Build a frozen cfg for held-out ``.reidraw`` evaluation only.

    Does not load ``pool_v2.yaml`` or V3 pack paths (often absent on Kaggle eval kernels).
    """
    cfg = load_cfg(config_file, opts)
    cfg.DATASETS.NAMES = ()
    cfg.DATASETS.TESTS = ()
    if int(cfg.MODEL.HEADS.NUM_CLASSES) <= 0:
        cfg.MODEL.HEADS.NUM_CLASSES = int(num_classes)
    apply_init_model(cfg, model_key, prepare_weights=False)
    resolve_output_dir(cfg, output_dir)
    cfg.freeze()
    return cfg
