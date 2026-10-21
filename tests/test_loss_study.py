"""Loss-study manifest, objectives, and protocol helpers."""

from __future__ import annotations

import json

import pytest
import torch

from reidfm.experiments.contract import LOSS_STUDY_CONDITIONS, LOSS_STUDY_WHITELIST, PRIMARY_INIT_MODEL
from reidfm.experiments.loss_study import expand_loss_study_run, load_loss_study_manifest
from reidfm.modeling.losses.sphereface2 import sphereface2_loss


def test_manifest_has_nine_unique_conditions():
    manifest = load_loss_study_manifest()
    assert set(manifest["conditions"]) == set(LOSS_STUDY_CONDITIONS)
    assert manifest["init_model"] == PRIMARY_INIT_MODEL


def test_expand_run_sets_study_flags():
    run = expand_loss_study_run("B2", 1)
    assert "REIDFM.STUDY_MODE" in run.opts and True in run.opts
    assert run.opts.count("REIDFM.LOSS_STUDY_CONDITION") == 1
    assert "B2" in run.opts


def test_whitelist_covers_condition_keys():
    manifest = load_loss_study_manifest()
    for cond, entry in manifest["conditions"].items():
        for key in (entry.get("opts") or {}):
            assert key in LOSS_STUDY_WHITELIST, f"{cond}: {key}"


def test_sphereface2_finite_grad():
    logits = torch.randn(4, 8, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    loss = sphereface2_loss(logits, targets, scale=64.0, margin=0.2, gamma=1.0)
    loss.backward()
    assert torch.isfinite(loss).all()
    assert logits.grad is not None


def test_dry_run_loss_study_cli():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_loss_study.py"), "--condition", "R0", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert out.returncode == 0
    assert "train.py" in out.stdout


def test_config_fingerprint_stable():
    pytest.importorskip("yacs")
    from reidfm.config import setup_cfg
    from reidfm.experiments.protocol import config_fingerprint

    run = expand_loss_study_run("R0", 0)
    cfg = setup_cfg(
        run.config_file,
        opts=run.opts,
        model_key=run.init_model,
        output_dir="/tmp/loss_study_test",
        prepare_weights=False,
        validate_study=False,
    )
    a = config_fingerprint(cfg)
    b = config_fingerprint(cfg)
    assert a == b


def test_study_protocol_requires_val_manifest(tmp_path):
    from types import SimpleNamespace

    from reidfm.experiments.protocol import validate_study_protocol

    split = tmp_path / "train_ids.json"
    sources = load_loss_study_manifest()
    pool_sources = load_pool_sources()
    split.write_text(json.dumps({k: ["1"] for k in pool_sources}), encoding="utf-8")
    cfg = SimpleNamespace(
        REIDFM=SimpleNamespace(
            STUDY_MODE=True,
            TRAIN_SPLIT=str(split),
            VAL_MANIFEST=str(tmp_path / "missing_val"),
            INIT_MODEL=PRIMARY_INIT_MODEL,
            LOSS_STUDY_CONDITION="R0",
            POOL="configs/datasets/pool_v2.yaml",
        )
    )
    with pytest.raises(FileNotFoundError):
        validate_study_protocol(cfg, require_imagenet_init=True)


def load_pool_sources():
    from reidfm.data.pool import load_pool

    return load_pool("configs/datasets/pool_v2.yaml").sources
