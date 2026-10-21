"""SphereFace2-style multi-class angular margin (one-vs-all hyperspherical)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sphereface2_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: float,
    margin: float,
    gamma: float = 0.0,
) -> torch.Tensor:
    """``logits`` are cosine similarities (unnormalized dot products of unit vectors).

    Positive class: push cos(theta) below cos(theta + m).
    Negative classes: push cos(theta) above cos(theta - m) when gamma > 0 (asymmetric).
    """
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0.0

    idx = torch.where(valid)[0]
    t = targets[idx]
    cos = logits[idx].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos)
    one_hot = F.one_hot(t, num_classes=cos.size(1)).to(dtype=cos.dtype)

    pos_cos = (cos * one_hot).sum(dim=1)
    pos_theta = torch.acos(pos_cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    pos_target = torch.cos(pos_theta + margin)

    neg_mask = 1.0 - one_hot
    neg_cos = cos * neg_mask + one_hot * (-1.0)
    neg_theta = torch.acos(neg_cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    neg_target = torch.cos(neg_theta - margin * gamma)

    target_cos = one_hot * pos_target.unsqueeze(1) + neg_mask * neg_target
    scaled = target_cos * scale
    log_probs = F.log_softmax(scaled, dim=1)
    loss = F.nll_loss(log_probs, t)
    return loss
