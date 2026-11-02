"""Embedding head with optional Sub-center ArcFace logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import get_norm, pooling
from fastreid.layers.weight_init import weights_init_kaiming
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY


class _SubCenterArcSoftmax(nn.Module):
    def __init__(self, num_classes: int, num_subcenters: int, scale: float, margin: float):
        super().__init__()
        self.num_classes = num_classes
        self.k = num_subcenters
        self.s = scale
        self.m = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C*K] cosines after linear on normalized features
        n = logits.size(0)
        cos = logits.float().view(n, self.num_classes, self.k).amax(dim=2)
        index = torch.where(targets != -1)[0]
        if index.numel():
            target_cos = cos[index, targets[index]].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            cos = cos.index_put((index, targets[index]), torch.cos(torch.acos(target_cos) + self.m))
        return cos * self.s


@REID_HEADS_REGISTRY.register()
class AngularEmbeddingHead(nn.Module):
    """Same pooling/BN-neck path as FastReID ``EmbeddingHead``; supports Sub-center ArcFace."""

    @configurable
    def __init__(
        self,
        *,
        feat_dim,
        embedding_dim,
        num_classes,
        neck_feat,
        pool_type,
        cls_type,
        scale,
        margin,
        with_bnneck,
        norm_type,
        num_subcenters: int = 1,
    ):
        super().__init__()
        assert hasattr(pooling, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()
        self.neck_feat = neck_feat
        self.cls_type = cls_type
        self.num_subcenters = max(1, int(num_subcenters))

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim
        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
        self.bottleneck = nn.Sequential(*neck)

        out_classes = num_classes * self.num_subcenters if cls_type == "SubCenterArcSoftmax" else num_classes
        self.weight = nn.Parameter(torch.Tensor(out_classes, feat_dim))
        if cls_type == "SubCenterArcSoftmax":
            self.cls_layer = _SubCenterArcSoftmax(num_classes, self.num_subcenters, scale, margin)
        elif cls_type == "ArcSoftmax":
            from fastreid.layers.any_softmax import ArcSoftmax

            self.cls_layer = ArcSoftmax(num_classes, scale, margin)
        elif cls_type == "Linear":
            from fastreid.layers.any_softmax import Linear

            self.cls_layer = Linear(num_classes, scale, margin)
        else:
            raise ValueError(f"AngularEmbeddingHead unsupported CLS_LAYER={cls_type!r}")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        return {
            "feat_dim": cfg.MODEL.BACKBONE.FEAT_DIM,
            "embedding_dim": cfg.MODEL.HEADS.EMBEDDING_DIM,
            "num_classes": cfg.MODEL.HEADS.NUM_CLASSES,
            "neck_feat": cfg.MODEL.HEADS.NECK_FEAT,
            "pool_type": cfg.MODEL.HEADS.POOL_LAYER,
            "cls_type": cfg.MODEL.HEADS.CLS_LAYER,
            "scale": cfg.MODEL.HEADS.SCALE,
            "margin": cfg.MODEL.HEADS.MARGIN,
            "with_bnneck": cfg.MODEL.HEADS.WITH_BNNECK,
            "norm_type": cfg.MODEL.HEADS.NORM,
            "num_subcenters": int(getattr(cfg.MODEL.HEADS, "NUM_SUBCENTERS", 1)),
        }

    def forward(self, features, targets=None):
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]

        if not self.training:
            return neck_feat

        if self.cls_layer.__class__.__name__ == "Linear":
            logits = F.linear(neck_feat, self.weight)
            raw_cos = F.linear(F.normalize(neck_feat), F.normalize(self.weight[: self.cls_layer.num_classes]))
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))
            raw_cos = logits

        cls_outputs = self.cls_layer(logits.clone(), targets)

        if self.neck_feat == "before":
            feat = pool_feat[..., 0, 0]
        elif self.neck_feat == "after":
            feat = neck_feat
        else:
            raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": cls_outputs,
            "features": feat,
            "raw_cosine_logits": raw_cos,
        }
