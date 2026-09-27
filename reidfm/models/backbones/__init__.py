"""Extra backbones registered into FastReID's ``BACKBONE_REGISTRY``.

FastReID already ships ResNet(-IBN), ResNeSt, OSNet, ViT, ...; we only add what it lacks.
Configs select them with ``MODEL.BACKBONE.NAME``.
"""

from __future__ import annotations

from torch import nn

from reidfm.models.backbones.osnet_ain import osnet_ain_x1_0

_REGISTERED = False


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


def build_osnet_ain_backbone(cfg) -> nn.Module:
    return OSNetAINBackbone()


def register_backbones() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from fastreid.modeling.backbones import BACKBONE_REGISTRY

    if "build_osnet_ain_backbone" not in BACKBONE_REGISTRY._obj_map:
        BACKBONE_REGISTRY.register(build_osnet_ain_backbone)
    _REGISTERED = True


__all__ = ["OSNetAINBackbone", "build_osnet_ain_backbone", "register_backbones"]
