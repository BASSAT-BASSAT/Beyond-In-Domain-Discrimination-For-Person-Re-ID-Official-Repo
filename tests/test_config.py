"""Training yaml loading and the REIDFM data/sampler overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from reidfm.utils.paths import repo_path

pytestmark = pytest.mark.fastreid

TRAIN_CONFIGS = sorted(
    f"configs/{p.relative_to(repo_path('configs')).as_posix()}"
    for p in [*repo_path("configs").glob("*.yml"), *repo_path("configs/models").glob("*.yml")]
    if "kaggle" not in p.parts
)

KAGGLE_CONFIGS = sorted(
    f"configs/kaggle/{p.name}" for p in repo_path("configs/kaggle").glob("*.yml")
)


@pytest.mark.parametrize("config", TRAIN_CONFIGS)
def test_train_configs_are_self_contained_and_load(config: str, monkeypatch):
    from reidfm.config import setup_cfg

    monkeypatch.delenv("REIDFM_TIME_BUDGET_H", raising=False)
    lines = Path(repo_path(config)).read_text().splitlines()
    assert not any(line.startswith("_BASE_:") for line in lines)
    cfg = setup_cfg(config, prepare_weights=False)
    assert cfg.DATASETS.NAMES and cfg.DATASETS.TESTS


def _cfg(opts):
    from reidfm.config import load_cfg

    return load_cfg("configs/smoke_sbs_r50ibn.yml", opts)


def test_pool_overrides_sources_val_and_sampler():
    from reidfm.config import resolve_pool

    cfg = _cfg(["REIDFM.SOURCES", "['market1501']", "REIDFM.VAL_SETS", "['market1501']",
                "REIDFM.SAMPLER_WEIGHT_MODE", "uniform", "REIDFM.DOMAIN_WEIGHTS", "['market1501=2.0']"])
    pool = resolve_pool(cfg)
    assert pool.sources == ["market1501"]
    assert pool.weight_mode == "uniform"
    assert pool.domain_weights == {"market1501": 2.0}


def test_ood_set_cannot_become_a_source():
    from reidfm.config import resolve_pool

    with pytest.raises(ValueError, match="OOD"):
        resolve_pool(_cfg(["REIDFM.SOURCES", "['market1501', 'mars']"]))


def test_env_time_budget_overrides_yaml(monkeypatch):
    monkeypatch.setenv("REIDFM_TIME_BUDGET_H", "3.5")
    assert _cfg([]).REIDFM.TIME_BUDGET_H == 3.5


def test_setup_eval_cfg_skips_pool_yaml(monkeypatch):
    from reidfm.config import setup_eval_cfg

    monkeypatch.delenv("REIDFM_TIME_BUDGET_H", raising=False)
    cfg = setup_eval_cfg(
        "configs/scratch_sbs_r50ibn_imagenet.yml",
        num_classes=10036,
    )
    assert cfg.DATASETS.NAMES == ()
    assert cfg.DATASETS.TESTS == ()
    assert cfg.MODEL.HEADS.NUM_CLASSES == 10036


@pytest.mark.parametrize("config", KAGGLE_CONFIGS)
def test_kaggle_presets_merge_base(config: str, monkeypatch):
    from reidfm.config import load_cfg

    monkeypatch.delenv("REIDFM_TIME_BUDGET_H", raising=False)
    cfg = load_cfg(config)
    assert cfg.REIDFM.INIT_MODEL
    assert cfg.OUTPUT_DIR
    if config.endswith("primary_sbs_r50_poolv2.yml"):
        assert cfg.MODEL.HEADS.CLS_LAYER == "CircleSoftmax"
