"""SphereFace2-C loss (Wen et al., ICLR 2022): one-vs-all binary classification on the hypersphere.

Mirrors the reference implementation in OpenSphere (``opensphere/module/head/sphereface2.py``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sphereface2_init_bias(num_classes: int, *, alpha: float, scale: float, margin: float, t: float) -> float:
    """Initial bias that balances positive/negative pair losses at the start of training."""
    z = alpha / ((1.0 - alpha) * (num_classes - 1.0))
    ay = scale * (2.0 * 0.5**t - 1.0 - margin)
    ai = scale * (2.0 * 0.5**t - 1.0 + margin)
    temp = (1.0 - z) ** 2 + 4.0 * z * math.exp(ay - ai)
    return math.log(2.0 * z) - ai - math.log(1.0 - z + math.sqrt(temp))


def sphereface2_loss(
    cosine: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor,
    *,
    alpha: float,
    scale: float,
    margin: float,
    t: float,
    loss_weight: float,
) -> torch.Tensor:
    """``cosine`` is ``[N, C]`` cosine similarity between L2-normalized features and class weights.

    ``alpha`` is the positive/negative balance (lambda in Eq. 5 of the paper), ``scale`` the
    hypersphere radius r, ``margin`` the additive margin m, and ``t`` the similarity-adjustment
    exponent of g(z) = 2((z + 1) / 2)^t - 1.
    """
    valid = targets >= 0
    if not valid.any():
        return cosine.sum() * 0.0

    cos = cosine[valid].float()
    y = targets[valid]
    num_classes = cos.size(1)
    one_hot = F.one_hot(y, num_classes=num_classes).to(dtype=cos.dtype)

    with torch.no_grad():
        g_cos = 2.0 * ((cos + 1.0) / 2.0).pow(t) - 1.0
        g_cos = g_cos - margin * (2.0 * one_hot - 1.0)
        d_theta = g_cos - cos

    logits = scale * (cos + d_theta) + bias.float()
    weight = alpha * one_hot + (1.0 - alpha) * (1.0 - one_hot)
    weight = loss_weight * num_classes / scale * weight
    return F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)


class SphereFace2Loss(nn.Module):
    """Holds the learnable bias b; class weights live in the embedding head."""

    def __init__(
        self,
        num_classes: int,
        *,
        alpha: float,
        scale: float,
        margin: float,
        t: float,
        loss_weight: float,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("SphereFace2Loss requires at least two classes")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"SphereFace2 alpha must be in (0, 1), got {alpha}")
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.margin = float(margin)
        self.t = float(t)
        self.loss_weight = float(loss_weight)
        init = sphereface2_init_bias(self.num_classes, alpha=self.alpha, scale=self.scale, margin=self.margin, t=self.t)
        self.bias = nn.Parameter(torch.full((1,), init))

    def forward(self, cosine: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sphereface2_loss(
            cosine,
            targets,
            self.bias,
            alpha=self.alpha,
            scale=self.scale,
            margin=self.margin,
            t=self.t,
            loss_weight=self.loss_weight,
        )
