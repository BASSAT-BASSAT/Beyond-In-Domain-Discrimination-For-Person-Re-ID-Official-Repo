"""Extra backbones registered into FastReID's ``BACKBONE_REGISTRY``.

FastReID already ships ResNet(-IBN), ResNeSt, OSNet, ViT, ...; we only add what it lacks.
Configs select them with ``MODEL.BACKBONE.NAME``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch
from torch import nn

from reidfm.models.backbones.osnet_ain import osnet_ain_x1_0, pretrained_urls

logger = logging.getLogger(__name__)
_REGISTERED = False

OSNET_AIN_IMAGENET_FILE = "osnet_ain_x1_0_imagenet.pth"
OSNET_AIN_IMAGENET_SHA256 = "b8a578eacc122c16496c8835c96a0157db62963b3de05b5a5a4b1ff5bad7546a"


class OSNetAINBackbone(nn.Module):
    """torchreid OSNet-AIN x1.0 trunk returning the 512-channel conv5 feature map.

    State-dict keys equal torchreid's minus ``fc``/``classifier``, so torchreid
    checkpoints load after :func:`reidfm.models.init.strip_osnet_checkpoint`.
    """

    def __init__(self):
        super().__init__()
        net = osnet_ain_x1_0(num_classes=1, pretrained=False)
        for name in ("conv1", "maxpool", "conv2", "pool2", "conv3", "pool3", "conv4", "conv5"):
            setattr(self, name, getattr(net, name))

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        return self.conv5(self.conv4(x))


def osnet_ain_imagenet_path() -> Path:
    """Download (once) and verify torchreid's ImageNet OSNet-AIN x1.0 checkpoint."""
    from reidfm.utils.paths import weights_root

    path = weights_root() / OSNET_AIN_IMAGENET_FILE
    if not path.is_file():
        import gdown

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        gdown.download(pretrained_urls["osnet_ain_x1_0"], str(tmp), quiet=True)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OSNET_AIN_IMAGENET_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest} != {OSNET_AIN_IMAGENET_SHA256}; delete it and retry")
    return path


def build_osnet_ain_backbone(cfg) -> nn.Module:
    model = OSNetAINBackbone()
    if cfg.MODEL.BACKBONE.PRETRAIN:
        path = Path(cfg.MODEL.BACKBONE.PRETRAIN_PATH) if cfg.MODEL.BACKBONE.PRETRAIN_PATH else osnet_ain_imagenet_path()
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        state = {k: v for k, v in state.items() if not k.startswith(("fc.", "classifier."))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"OSNet-AIN ImageNet load: missing={missing[:5]} unexpected={unexpected[:5]}")
        logger.info("OSNet-AIN x1.0 initialised from ImageNet (%s)", path.name)
    return model


class ConvNeXtBackbone(nn.Module):
    """torchvision ConvNeXt trunk (``features``) returning the last-stage feature map.

    ``last_stride=1`` turns the final 2x2/stride-2 downsampling conv into stride 1, the
    ConvNeXt counterpart of the ResNet last-stride trick used by the Strong Baseline.
    """

    def __init__(self, variant: str = "small", pretrained: bool = False, last_stride: int = 2,
                 drop_path: float = 0.1):
        super().__init__()
        from torchvision import models

        builders = {
            "tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights),
            "small": (models.convnext_small, models.ConvNeXt_Small_Weights),
            "base": (models.convnext_base, models.ConvNeXt_Base_Weights),
        }
        build, weights = builders[variant]
        net = build(
            weights=weights.IMAGENET1K_V1 if pretrained else None,
            stochastic_depth_prob=drop_path,
        )
        self.features = net.features
        if last_stride == 1:
            self.features[6][1].stride = (1, 1)

    def forward(self, x):
        return self.features(x)


def build_convnext_backbone(cfg) -> nn.Module:
    depth = str(cfg.MODEL.BACKBONE.DEPTH).lower()
    variant = {"s": "small", "small": "small", "t": "tiny", "tiny": "tiny", "b": "base", "base": "base"}[depth]
    return ConvNeXtBackbone(
        variant=variant,
        pretrained=bool(cfg.MODEL.BACKBONE.PRETRAIN),
        last_stride=int(cfg.MODEL.BACKBONE.LAST_STRIDE),
        drop_path=float(cfg.MODEL.BACKBONE.DROP_PATH_RATIO),
    )


def prefetch_backbone_weights(cfg) -> None:
    """Download ImageNet weights before DDP workers start so ranks never race on the cache."""
    if not cfg.MODEL.BACKBONE.PRETRAIN:
        return
    name = cfg.MODEL.BACKBONE.NAME
    if name == "build_osnet_ain_backbone" and not cfg.MODEL.BACKBONE.PRETRAIN_PATH:
        osnet_ain_imagenet_path()
    elif name == "build_convnext_backbone":
        build_convnext_backbone(cfg)


def register_backbones() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from fastreid.modeling.backbones import BACKBONE_REGISTRY

    for builder in (build_osnet_ain_backbone, build_convnext_backbone):
        if builder.__name__ not in BACKBONE_REGISTRY._obj_map:
            BACKBONE_REGISTRY.register(builder)
    _REGISTERED = True


__all__ = [
    "ConvNeXtBackbone",
    "OSNetAINBackbone",
    "build_convnext_backbone",
    "build_osnet_ain_backbone",
    "prefetch_backbone_weights",
    "register_backbones",
]
