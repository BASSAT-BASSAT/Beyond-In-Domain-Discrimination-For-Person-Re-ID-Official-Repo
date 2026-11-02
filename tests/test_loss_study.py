"""Loss-study manifest, objectives, and protocol helpers."""

from __future__ import annotations

import json

import pytest
import torch

from reidfm.experiments.contract import LOSS_STUDY_CONDITIONS, LOSS_STUDY_WHITELIST, PRIMARY_INIT_MODEL
from reidfm.experiments.loss_study import expand_loss_study_run, load_loss_study_manifest
from reidfm.modeling.losses.sphereface2 import SphereFace2Loss, sphereface2_loss


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


SPHERE_HPARAMS = dict(alpha=0.7, scale=40.0, margin=0.4, t=3.0, loss_weight=50.0)


def _opensphere_reference(cos_theta, y, b, *, alpha, scale, margin, t, loss_weight):
    """Transcription of OpenSphere ``SphereFace2.forward`` after the cosine is computed."""
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, y.view(-1, 1), 1.0)
    with torch.no_grad():
        g_cos_theta = 2.0 * ((cos_theta + 1.0) / 2.0).pow(t) - 1.0
        g_cos_theta = g_cos_theta - margin * (2.0 * one_hot - 1.0)
        d_theta = g_cos_theta - cos_theta
    logits = scale * (cos_theta + d_theta) + b
    weight = alpha * one_hot + (1.0 - alpha) * (1.0 - one_hot)
    weight = loss_weight * cos_theta.size(1) / scale * weight
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)


def _random_cosines(n, c, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=1)
    w = torch.nn.functional.normalize(torch.randn(c, 16, generator=g), dim=1)
    return x @ w.t()


def test_sphereface2_matches_opensphere_reference():
    cos = _random_cosines(6, 10).requires_grad_(True)
    cos_ref = cos.detach().clone().requires_grad_(True)
    targets = torch.tensor([0, 3, 9, 1, 3, 5])
    b = torch.tensor([1.5])

    ours = sphereface2_loss(cos, targets, b, **SPHERE_HPARAMS)
    ref = _opensphere_reference(cos_ref, targets, b, **SPHERE_HPARAMS)
    ours.backward()
    ref.backward()
    assert torch.allclose(ours, ref, atol=1e-6)
    assert torch.allclose(cos.grad, cos_ref.grad, atol=1e-6)


def test_sphereface2_init_bias_matches_opensphere():
    import math

    c, alpha, r, m, t = 10036, 0.7, 40.0, 0.4, 3.0
    z = alpha / ((1.0 - alpha) * (c - 1.0))
    ay = r * (2.0 * 0.5**t - 1.0 - m)
    ai = r * (2.0 * 0.5**t - 1.0 + m)
    expected = math.log(2.0 * z) - ai - math.log(1.0 - z + math.sqrt((1.0 - z) ** 2 + 4.0 * z * math.exp(ay - ai)))
    module = SphereFace2Loss(c, **SPHERE_HPARAMS)
    assert module.bias.item() == pytest.approx(expected, rel=1e-6)


def test_sphereface2_is_binary_not_softmax():
    # Per-class binary terms: changing a negative logit's cosine must not change another negative's gradient.
    cos = _random_cosines(1, 5).requires_grad_(True)
    targets = torch.tensor([0])
    module = SphereFace2Loss(5, **SPHERE_HPARAMS)
    module(cos, targets).backward()
    grad_before = cos.grad[0, 2].item()

    cos2 = cos.detach().clone()
    cos2[0, 4] = 0.9
    cos2.requires_grad_(True)
    module(cos2, targets).backward()
    assert cos2.grad[0, 2].item() == pytest.approx(grad_before, rel=1e-6)


def test_sphereface2_grad_reaches_bias_and_ignores_unlabeled():
    cos = _random_cosines(4, 8).requires_grad_(True)
    module = SphereFace2Loss(8, **SPHERE_HPARAMS)
    loss = module(cos, torch.tensor([0, -1, 2, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert module.bias.grad is not None and torch.isfinite(module.bias.grad).all()
    assert torch.all(cos.grad[1] == 0)


@pytest.mark.fastreid
def test_linear_softmax_conditions_use_unit_scale():
    from reidfm.config import setup_cfg

    for cond, expected in (("B0", 1), ("B1", 1), ("R0", 64), ("B2", 64)):
        run = expand_loss_study_run(cond, 0)
        cfg = setup_cfg(
            run.config_file,
            opts=run.opts,
            model_key=run.init_model,
            output_dir="/tmp/loss_study_test",
            prepare_weights=False,
            validate_study=False,
        )
        assert float(cfg.MODEL.HEADS.SCALE) == expected, cond


@pytest.mark.fastreid
def test_subcenter_arcface_with_one_center_equals_fastreid_arcsoftmax():
    from fastreid.layers.any_softmax import ArcSoftmax

    from reidfm.modeling.heads.angular_head import _SubCenterArcSoftmax

    cos = _random_cosines(5, 7)
    targets = torch.tensor([0, 6, 2, -1, 2])
    ours = _SubCenterArcSoftmax(7, 1, 64.0, 0.35)(cos.clone(), targets)
    ref = ArcSoftmax(7, 64.0, 0.35)(cos.clone(), targets)
    assert torch.allclose(ours, ref, atol=1e-4)


def test_subcenter_arcface_uses_best_center_and_target_margin():
    from reidfm.modeling.heads.angular_head import _SubCenterArcSoftmax

    num_classes, k, s, m = 4, 3, 64.0, 0.35
    cos = _random_cosines(3, num_classes * k).requires_grad_(True)
    targets = torch.tensor([1, 3, 0])
    out = _SubCenterArcSoftmax(num_classes, k, s, m)(cos, targets)

    best = cos.detach().view(3, num_classes, k).amax(dim=2)
    expected = best.clone()
    for i, y in enumerate(targets.tolist()):
        expected[i, y] = torch.cos(torch.acos(best[i, y]) + m)
    assert torch.allclose(out, expected * s, atol=1e-4)
    out.sum().backward()
    assert torch.isfinite(cos.grad).all()


@pytest.mark.fastreid
def test_loss_study_baseline_owns_sphere_bias_only_for_sphereface2():
    from fastreid.modeling.meta_arch import build_model

    from reidfm.config import setup_cfg

    for cond, expect_bias in (("B6", True), ("B7", True), ("B3", False)):
        run = expand_loss_study_run(cond, 0)
        cfg = setup_cfg(
            run.config_file,
            opts=list(run.opts) + ["MODEL.DEVICE", "cpu", "MODEL.BACKBONE.PRETRAIN", "False", "MODEL.HEADS.NUM_CLASSES", "12"],
            model_key=run.init_model,
            output_dir="/tmp/loss_study_test",
            prepare_weights=False,
            validate_study=False,
        )
        model = build_model(cfg)
        names = {n for n, _ in model.named_parameters()}
        assert ("sphere_loss.bias" in names) is expect_bias, cond


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
