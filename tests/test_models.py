"""Model registry, checkpoint remapping, and FastReID integration (marked ``fastreid``)."""

from __future__ import annotations

import pytest

from reidfm.models import available_models, get_model_spec


def test_registry_contents():
    keys = set(available_models())
    assert {"sbs_r50_ibn_msmt", "sbs_r101_ibn_msmt", "sbs_r50_ibn_imagenet", "osnet_ain_msdc", "sbs_s50_mot17"} <= keys
    spec = get_model_spec("sbs_r50_ibn_msmt")
    assert spec.config_file.endswith("MSMT17/sbs_R50-ibn.yml")
    assert spec.weights_url.endswith("msmt_sbs_R50-ibn.pth")
    assert get_model_spec("sbs_r50_ibn_imagenet").init == "imagenet"
    assert all(get_model_spec(k).init == "imagenet" for k in available_models(init="imagenet"))
    assert available_models(backend="osnet") == ["osnet_ain_msdc"]
    with pytest.raises(KeyError):
        get_model_spec("does_not_exist")


def test_zoo_checksums_are_hex():
    from reidfm.models.zoo import CHECKSUMS

    assert CHECKSUMS and all(len(v) == 64 and int(v, 16) >= 0 for v in CHECKSUMS.values())


def test_remap_checkpoint_old_zoo_keys():
    import torch

    from reidfm.models.init import is_classifier_key, remap_checkpoint

    model_state = {
        "heads.bottleneck.0.weight": torch.zeros(4),
        "heads.weight": torch.zeros(10, 4),
        "backbone.conv1.weight": torch.zeros(2, 3),
    }
    old = {
        "heads.bnneck.weight": torch.ones(4),
        "heads.classifier.weight": torch.ones(751, 4),  # shape mismatch -> dropped
        "backbone.conv1.weight": torch.ones(2, 3),
        "backbone.unused": torch.ones(1),
    }
    remapped = remap_checkpoint(old, model_state)
    assert set(remapped) == {"heads.bottleneck.0.weight", "backbone.conv1.weight"}
    assert is_classifier_key("heads.weight") and is_classifier_key("b1_head.weight")
    assert not is_classifier_key("heads.bottleneck.0.weight")


@pytest.mark.fastreid
def test_manifest_datasets_registered():
    from fastreid.data.datasets import DATASET_REGISTRY

    from reidfm.data.registry import available_datasets, get_dataset_spec

    for key in available_datasets():
        assert DATASET_REGISTRY.get(get_dataset_spec(key).fastreid_name) is not None


@pytest.mark.fastreid
def test_setup_cfg_applies_pool_and_imagenet_init(tmp_path, monkeypatch):
    monkeypatch.setenv("REIDFM_OUTPUT_ROOT", str(tmp_path))
    from reidfm.config import setup_cfg

    cfg = setup_cfg("configs/scratch_sbs_r50ibn_imagenet.yml", prepare_weights=False)
    assert cfg.MODEL.BACKBONE.PRETRAIN is True and cfg.MODEL.BACKBONE.WITH_IBN is True
    assert "PRID2011M" not in cfg.DATASETS.NAMES and "IUSTM" not in cfg.DATASETS.NAMES
    assert "Market1501M" in cfg.DATASETS.NAMES and cfg.DATASETS.TESTS
    assert cfg.OUTPUT_DIR.startswith(str(tmp_path))
    assert cfg.DATALOADER.SAMPLER_TRAIN == "DomainBalancedIdentitySampler"


@pytest.mark.fastreid
def test_osnet_backbone_builds():
    import torch

    from reidfm.config import get_reidfm_cfg

    cfg = get_reidfm_cfg()
    cfg.MODEL.BACKBONE.NAME = "build_osnet_ain_backbone"
    cfg.MODEL.BACKBONE.PRETRAIN = False
    from fastreid.modeling.backbones import build_backbone

    out = build_backbone(cfg)(torch.zeros(1, 3, 256, 128))
    assert out.shape[1] == 512
