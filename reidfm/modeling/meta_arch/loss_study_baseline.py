"""Baseline meta-architecture with SphereFace2 loss support."""

from __future__ import annotations

from fastreid.modeling.losses import cross_entropy_loss, triplet_loss
from fastreid.modeling.meta_arch.baseline import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY

from reidfm.modeling.losses.sphereface2 import sphereface2_loss


@META_ARCH_REGISTRY.register()
class LossStudyBaseline(Baseline):
    @classmethod
    def from_config(cls, cfg):
        out = super().from_config(cfg)
        sphere = cfg.MODEL.LOSSES.SPHERE
        out["loss_kwargs"]["sphere"] = {
            "scale": float(sphere.SCALE),
            "margin": float(sphere.MARGIN),
            "lambda": float(sphere.LAMBDA),
            "gamma": float(sphere.GAMMA),
        }
        return out

    def losses(self, outputs, gt_labels):
        loss_dict = super().losses(outputs, gt_labels)
        loss_names = self.loss_kwargs["loss_names"]
        if "SphereFace2Loss" not in loss_names:
            return loss_dict

        sph = self.loss_kwargs.get("sphere") or {}
        raw = outputs.get("raw_cosine_logits")
        if raw is None:
            raise KeyError("SphereFace2Loss requires AngularEmbeddingHead raw_cosine_logits")
        loss_dict["loss_sphere"] = sphereface2_loss(
            raw,
            gt_labels,
            scale=float(sph.get("scale", 64.0)),
            margin=float(sph.get("margin", 0.2)),
            gamma=float(sph.get("gamma", 1.0)),
        ) * float(sph.get("lambda", 1.0))
        if "CrossEntropyLoss" not in loss_names and "loss_cls" in loss_dict:
            del loss_dict["loss_cls"]
        return loss_dict
