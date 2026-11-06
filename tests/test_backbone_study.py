"""Cross-backbone study manifest (OSNet-AIN, ConvNeXt-S) and backbones."""

from __future__ import annotations

import pytest
import torch

from reidfm.experiments.contract import BACKBONE_STUDY_CONDITIONS, IMAGENET_INIT_MODELS
from reidfm.experiments.loss_study import (
    BACKBONE_MANIFEST,
    expand_loss_study_run,
    load_loss_study_manifest,
)

R0_CONDITIONS = ("OSNet_R0", "ConvNeXtS_R0")
B2_CONDITIONS = ("OSNet_B2", "ConvNeXtS_B2")
FEAT_DIM = {
    "OSNet_R0": 512,
    "OSNet_B2": 512,
    "ConvNeXtS_R0": 768,
    "ConvNeXtS_B2": 768,
}


def _cfg(condition):
    from reidfm.config import load_cfg

    run = expand_loss_study_run(condition, 0, load_loss_study_manifest(BACKBONE_MANIFEST))
    return run, load_cfg(run.config_file, run.opts)


def test_manifest_conditions_match_contract():
    manifest = load_loss_study_manifest(BACKBONE_MANIFEST)
    assert tuple(manifest["conditions"]) == BACKBONE_STUDY_CONDITIONS


def test_backbone_runs_share_schedule_and_imagenet_init():
    from reidfm.config import load_cfg

    sbs = expand_loss_study_run("R0", 0)
    sbs_cfg = load_cfg(sbs.config_file, sbs.opts)
    for condition in BACKBONE_STUDY_CONDITIONS:
        run, cfg = _cfg(condition)
        assert run.init_model in IMAGENET_INIT_MODELS
        assert run.init_model != sbs.init_model
        assert cfg.SOLVER == sbs_cfg.SOLVER
        assert cfg.INPUT == sbs_cfg.INPUT
        assert cfg.DATALOADER == sbs_cfg.DATALOADER
        assert cfg.MODEL.BACKBONE.WITH_IBN is False and cfg.MODEL.BACKBONE.WITH_NL is False


def test_backbone_r0_matches_sbs_circle_triplet_objective():
    from reidfm.config import load_cfg

    sbs = expand_loss_study_run("R0", 0)
    sbs_cfg = load_cfg(sbs.config_file, sbs.opts)
    for condition in R0_CONDITIONS:
        _, cfg = _cfg(condition)
        assert cfg.MODEL.HEADS == sbs_cfg.MODEL.HEADS
        assert cfg.MODEL.LOSSES == sbs_cfg.MODEL.LOSSES
        assert cfg.MODEL.HEADS.CLS_LAYER == "CircleSoftmax"
        assert list(cfg.MODEL.LOSSES.NAME) == ["CrossEntropyLoss", "TripletLoss"]


def test_backbone_b2_matches_sbs_arcface_objective():
    from reidfm.config import load_cfg

    sbs = expand_loss_study_run("B2", 0)
    sbs_cfg = load_cfg(sbs.config_file, sbs.opts)
    for condition in B2_CONDITIONS:
        _, cfg = _cfg(condition)
        assert cfg.MODEL.HEADS == sbs_cfg.MODEL.HEADS
        assert cfg.MODEL.LOSSES == sbs_cfg.MODEL.LOSSES
        assert cfg.MODEL.HEADS.CLS_LAYER == "ArcSoftmax"
        assert list(cfg.MODEL.LOSSES.NAME) == ["CrossEntropyLoss"]
        assert "TripletLoss" not in cfg.MODEL.LOSSES.NAME


@pytest.mark.fastreid
@pytest.mark.parametrize(
    "condition",
    ["OSNet_R0", "ConvNeXtS_R0", "OSNet_B2", "ConvNeXtS_B2"],
)
def test_backbone_model_builds_and_embeds(condition):
    from reidfm.fastreid_compat import ensure_fastreid

    ensure_fastreid()
    from fastreid.modeling.meta_arch import build_model

    _, cfg = _cfg(condition)
    cfg.defrost()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = 10
    model = build_model(cfg).eval()
    with torch.no_grad():
        feats = model({"images": torch.zeros(2, 3, 384, 128)})
    assert feats.shape == (2, FEAT_DIM[condition])


@pytest.mark.fastreid
def test_convnext_last_stride_one_keeps_resolution():
    from reidfm.models.backbones import ConvNeXtBackbone

    x = torch.zeros(1, 3, 384, 128)
    assert ConvNeXtBackbone(last_stride=2)(x).shape[-2:] == (12, 4)
    assert ConvNeXtBackbone(last_stride=1)(x).shape[-2:] == (23, 7)


@pytest.mark.fastreid
def test_osnet_imagenet_checkpoint_loads_every_trunk_tensor(tmp_path):
    from reidfm.config import get_reidfm_cfg
    from reidfm.models.backbones import OSNetAINBackbone, build_osnet_ain_backbone
    from reidfm.models.backbones.osnet_ain import osnet_ain_x1_0

    reference = osnet_ain_x1_0(num_classes=1000)
    path = tmp_path / "imagenet.pth"
    torch.save({"state_dict": {f"module.{k}": v for k, v in reference.state_dict().items()}}, path)

    cfg = get_reidfm_cfg()
    cfg.MODEL.BACKBONE.PRETRAIN = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = str(path)
    model = build_osnet_ain_backbone(cfg)
    assert isinstance(model, OSNetAINBackbone)
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key]), key
