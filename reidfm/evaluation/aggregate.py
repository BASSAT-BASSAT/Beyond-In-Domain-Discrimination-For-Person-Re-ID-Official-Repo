"""Tracklet aggregation, identical to benchmark ``reid/aggregate.py``: mean of frame features, then L2."""

from __future__ import annotations

import numpy as np


def mean_pool_l2(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("expected a non-empty [N, D] feature array")
    pooled = features.mean(axis=0)
    norm = np.linalg.norm(pooled)
    return (pooled if norm == 0 else pooled / norm).astype(np.float32)
