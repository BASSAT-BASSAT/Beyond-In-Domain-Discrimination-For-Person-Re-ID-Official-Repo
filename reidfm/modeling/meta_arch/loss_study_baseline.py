"""Baseline meta-architecture with SphereFace2 loss support."""

from __future__ import annotations

from fastreid.config import configurable
from fastreid.modeling.meta_arch.baseline import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY

from reidfm.modeling.losses.sphereface2 import SphereFace2Loss


@META_ARCH_REGISTRY.register()
class LossStudyBaseline(Baseline):
    @configurable
    def __init__(self, *, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.sphere_loss = None
        if "SphereFace2Loss" in self.loss_kwargs["loss_names"]:
            sph = self.loss_kwargs["sphere"]
            self.sphere_loss = SphereFace2Loss(
                num_classes,
                alpha=sph["lambda"],
                scale=sph["scale"],
                margin=sph["margin"],
                t=sph["t"],
                loss_weight=sph["loss_weight"],
            )

    @classmethod
    def from_config(cls, cfg):
        out = super().from_config(cfg)
        sphere = cfg.MODEL.LOSSES.SPHERE
        out["num_classes"] = int(cfg.MODEL.HEADS.NUM_CLASSES)
        out["loss_kwargs"]["sphere"] = {
            "scale": float(sphere.SCALE),
            "margin": float(sphere.MARGIN),
            "lambda": float(sphere.LAMBDA),
            "t": float(sphere.T),
            "loss_weight": float(sphere.LOSS_WEIGHT),
        }
        return out

    def losses(self, outputs, gt_labels):
        loss_dict = super().losses(outputs, gt_labels)
        if self.sphere_loss is None:
            return loss_dict

        raw = outputs.get("raw_cosine_logits")
        if raw is None:
            raise KeyError("SphereFace2Loss requires AngularEmbeddingHead raw_cosine_logits")
        loss_dict["loss_sphere"] = self.sphere_loss(raw, gt_labels)
        if "CrossEntropyLoss" not in self.loss_kwargs["loss_names"] and "loss_cls" in loss_dict:
            del loss_dict["loss_cls"]
        return loss_dict
