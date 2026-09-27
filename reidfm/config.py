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
    cfg.REIDFM.POOL = "configs/datasets/pool_v1.yaml"
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
    # Keep this many model_XXXX.pth epoch checkpoints (model_best/model_final are always kept).
    cfg.REIDFM.KEEP_CHECKPOINTS = 2
    # Evaluation: frames per tracklet for video sets, query cap for in-training val (0 = all).
    cfg.REIDFM.EVAL_NUM_FRAMES = 8
    cfg.REIDFM.VAL_MAX_QUERIES = 0
    cfg.REIDFM.EVAL_AMP = True
    cfg.REIDFM.EVAL_WORKERS = 4


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
    cfg = get_reidfm_cfg()
    cfg.merge_from_file(str(repo_path(config_file)))
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
    cfg.DATASETS.TESTS = tuple(pool.fastreid_names(pool.val_sets))
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


def setup_cfg(
    config_file: str | Path,
    opts: list[str] | None = None,
    model_key: str | None = None,
    pool: str | None = None,
    output_dir: str | None = None,
    time_budget_h: float | None = None,
    prepare_weights: bool = True,
):
    cfg = load_cfg(config_file, opts)
    apply_pool(cfg, pool)
    apply_init_model(cfg, model_key, prepare_weights=prepare_weights)
    if time_budget_h is not None:
        cfg.REIDFM.TIME_BUDGET_H = float(time_budget_h)
    resolve_output_dir(cfg, output_dir)
    cfg.freeze()
    return cfg
