"""Registered starting points (FastReID zoo, DiffMOT, torchreid, ImageNet) + trained models.

Importing this module fills :data:`reidfm.models.registry.MODEL_REGISTRY`.
"""

from __future__ import annotations

from reidfm.models.registry import MODEL_REGISTRY, ModelSpec, register_model
from reidfm.utils.config import load_yaml
from reidfm.utils.paths import REPO_ROOT

FASTREID_RELEASE = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1"
DIFFMOT_RELEASE = "https://github.com/Kroery/DiffMOT/releases/download/v1.0"
OSNET_AIN_MSDC_URL = "https://drive.google.com/uc?id=1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB"
TRAINED_REGISTRY = REPO_ROOT / "configs" / "models" / "trained.yaml"

# sha256 of files we downloaded and verified (Person-ReID-BenchMark/weights).
# MSMT17 checkpoints are not listed yet; add them after the first verified download.
CHECKSUMS: dict[str, str] = {
    "dance_sbs_S50.pth": "004beb12f0dd9dcf355bc57f5b8cc66b6904fc6d5599fb41ee9403c4c659ab29",
    "market_agw_R101-ibn.pth": "852395db3984c5dc54d1ff690aa288139e0cd554b91b8bbe588d1ab6727dca8e",
    "market_agw_R50-ibn.pth": "254823d1d1f803930beb6f1f036634f8b3667bb3a8e0e417f426563915463efb",
    "market_agw_R50.pth": "90b8098c063aa88d3b29d4e5ea8d9cb8557b6e084482d46f0c2cc6814dfc1dbd",
    "market_agw_S50.pth": "df27fca76bceadf63885cde87fe5d3a9cfbbe8c5c1c55eb2bac65b622e455fc8",
    "market_bot_R101-ibn.pth": "3a48b0ecf7cb48c4f52e68d115f83fbb80bc95e4da91069f4838d9815f61f6e2",
    "market_bot_R50-ibn.pth": "68670a3dcfab923eda104515a1f1ad15bf0447427021b3324cbd9712fd31c7d4",
    "market_bot_R50.pth": "764fa8ca18117e2f52102791334c852d1c2e5e0b7c727e3f271c16893f4184aa",
    "market_bot_S50.pth": "77d64b6ebcba5893aec94815d83997835023bb771be0a074cfba0762e581cae6",
    "market_mgn_R50-ibn.pth": "42690c333ebaca987e4081b09451f0426aa5e784b7a1ea658f7d639698c1e96b",
    "market_sbs_R101-ibn.pth": "076945caf799258d71445990109bb705265a6b05eb0549ce7a0a614da889ab6b",
    "market_sbs_R50-ibn.pth": "b02d7d195faa7ed6bec947291b6a11781fac5c77ff548800de62fee2cb5f98c1",
    "market_sbs_R50.pth": "2358c9d668108a54b3a03db5d519851ee4a49e7057c2ab638bd47addf7998263",
    "market_sbs_S50.pth": "989f336d0530fd15f413f9d6ced0e93882808f93520d6a9d4e406eaf62798092",
    "mot17_sbs_S50.pth": "eb2e83afe774c85f20b735a7fabc423659c022387187eba449419c18dd6b4fa9",
    "mot20_sbs_S50.pth": "d57c0bee3b1048aad8619d12f5d10277c8136a78141de09b68e1352e41b4059b",
    "osnet_ain_ms_d_c.pth.tar": "2f38acc25e28cb29407635db2be315edc08d5457a904b72a9a11e427f41f3242",
}

_FAMILIES = (
    ("sbs", "SBS", "sbs", "Stronger BoT: circle softmax, GeM, non-local, AutoAugment"),
    ("bot", "BoT", "bagtricks", "Bag of Tricks baseline (BNNeck, CE + triplet)"),
    ("agw", "AGW", "AGW", "AGW: non-local + GeM + weighted triplet"),
)
_BACKBONES = (
    ("r50", "R50", "R50"),
    ("r50_ibn", "R50-IBN", "R50-ibn"),
    ("s50", "S50", "S50"),
    ("r101_ibn", "R101-IBN", "R101-ibn"),
)
_TRAINSETS = (
    ("market", "Market1501", "Market1501"),
    ("msmt", "MSMT17", "MSMT17"),
    ("duke", "DukeMTMC", "DukeMTMC"),
)


def _register_fastreid_zoo() -> None:
    for fam_key, fam_label, yaml_prefix, notes in _FAMILIES:
        for bb_key, bb_label, yaml_stem in _BACKBONES:
            for ts_key, ts_label, cfg_dir in _TRAINSETS:
                weight = f"{ts_key}_{fam_key}_{yaml_stem}.pth"
                register_model(
                    ModelSpec(
                        key=f"{fam_key}_{bb_key}_{ts_key}",
                        label=f"FastReID {fam_label}-{bb_label} ({ts_label})",
                        family=fam_label,
                        backbone=bb_label,
                        trained_on=ts_label,
                        init="reid_pretrained",
                        config_file=f"fast-reid/configs/{cfg_dir}/{yaml_prefix}_{yaml_stem}.yml",
                        weights_url=f"{FASTREID_RELEASE}/{weight}",
                        weights_file=weight,
                        sha256=CHECKSUMS.get(weight),
                        input_size=(384, 128) if fam_key == "sbs" else (256, 128),
                        source="fastreid_zoo",
                        notes=notes,
                    )
                )
    for ts_key, ts_label, cfg_dir in (("market", "Market1501", "Market1501"), ("duke", "DukeMTMC", "DukeMTMC")):
        weight = f"{ts_key}_mgn_R50-ibn.pth"
        register_model(
            ModelSpec(
                key=f"mgn_r50_ibn_{ts_key}",
                label=f"FastReID MGN-R50-IBN ({ts_label})",
                family="MGN",
                backbone="R50-IBN",
                trained_on=ts_label,
                init="reid_pretrained",
                config_file=f"fast-reid/configs/{cfg_dir}/mgn_R50-ibn.yml",
                weights_url=f"{FASTREID_RELEASE}/{weight}",
                weights_file=weight,
                sha256=CHECKSUMS.get(weight),
                feat_dim=2048 * 8,
                source="fastreid_zoo",
                notes="Multi-granularity network; no MSMT17 zoo checkpoint",
            )
        )


def _register_tracking_models() -> None:
    for key, trained_on, weight in (
        ("sbs_s50_mot17", "MOT17", "mot17_sbs_S50.pth"),
        ("sbs_s50_mot20", "MOT20", "mot20_sbs_S50.pth"),
        ("sbs_s50_dancetrack", "DanceTrack", "dance_sbs_S50.pth"),
    ):
        register_model(
            ModelSpec(
                key=key,
                label=f"FastReID SBS-S50 ({trained_on})",
                family="SBS",
                backbone="S50",
                trained_on=trained_on,
                init="reid_pretrained",
                config_file="fast-reid/configs/Market1501/sbs_S50.yml",
                weights_url=f"{DIFFMOT_RELEASE}/{weight}",
                weights_file=weight,
                sha256=CHECKSUMS.get(weight),
                source="diffmot",
                notes="Deep OC-SORT / DiffMOT tracking ReID weights",
                tags=("tracking",),
            )
        )
    register_model(
        ModelSpec(
            key="osnet_ain_msdc",
            label="OSNet-AIN x1.0 (MSMT17+Duke+CUHK03)",
            family="OSNet",
            backbone="OSNet-AIN",
            trained_on="MSMT17+DukeMTMC+CUHK03",
            init="reid_pretrained",
            backend="osnet",
            config_file="configs/models/osnet_ain_x1_0.yml",
            weights_url=OSNET_AIN_MSDC_URL,
            weights_file="osnet_ain_ms_d_c.pth.tar",
            sha256=CHECKSUMS.get("osnet_ain_ms_d_c.pth.tar"),
            input_size=(256, 128),
            feat_dim=512,
            source="torchreid",
            notes="Zero-shot eval uses the native torchreid head; fine-tuning uses the FastReID backbone wrapper",
            tags=("tracking",),
        )
    )


def _register_imagenet_baselines() -> None:
    for bb_key, bb_label, yaml_stem in (("r50_ibn", "R50-IBN", "R50-ibn"), ("r101_ibn", "R101-IBN", "R101-ibn")):
        register_model(
            ModelSpec(
                key=f"sbs_{bb_key}_imagenet",
                label=f"SBS-{bb_label} (ImageNet init)",
                family="SBS",
                backbone=bb_label,
                trained_on="ImageNet",
                init="imagenet",
                config_file=f"fast-reid/configs/MSMT17/sbs_{yaml_stem}.yml",
                source="imagenet",
                notes="No ReID weights: FastReID downloads the ImageNet IBN backbone (MODEL.BACKBONE.PRETRAIN=True)",
            )
        )
    for fam_key, fam_label, yaml_prefix in (
        ("bot", "BoT", "bagtricks"),
        ("agw", "AGW", "AGW"),
    ):
        register_model(
            ModelSpec(
                key=f"{fam_key}_r50_ibn_imagenet",
                label=f"FastReID {fam_label}-R50-IBN (ImageNet init)",
                family=fam_label,
                backbone="R50-IBN",
                trained_on="ImageNet",
                init="imagenet",
                config_file=f"fast-reid/configs/MSMT17/{yaml_prefix}_R50-ibn.yml",
                input_size=(256, 128),
                source="imagenet",
                notes="Fair appendix baseline: train on pool v2 from ImageNet IBN only",
            )
        )


def _register_trained() -> None:
    if not TRAINED_REGISTRY.is_file():
        return
    for entry in load_yaml(TRAINED_REGISTRY).get("models") or []:
        entry = dict(entry)
        entry["input_size"] = tuple(entry.get("input_size", (384, 128)))
        entry["tags"] = tuple(entry.get("tags", ()))
        entry.setdefault("init", "trained")
        register_model(ModelSpec(**entry), overwrite=True)


if not MODEL_REGISTRY:
    _register_fastreid_zoo()
    _register_tracking_models()
    _register_imagenet_baselines()
    _register_trained()
