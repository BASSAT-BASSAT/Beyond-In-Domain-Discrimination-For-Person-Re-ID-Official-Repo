"""Frame sampling for tracklets (benchmark ``reid/sampling.py``)."""

from __future__ import annotations

import numpy as np


def sample_indices(num_items: int, num_frames: int = 8) -> list[int]:
    """Uniformly pick up to ``num_frames`` indices; all of them if the tracklet is shorter."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if num_items <= num_frames:
        return list(range(num_items))
    return [int(i) for i in np.unique(np.linspace(0, num_items - 1, num=num_frames, dtype=np.int64))]
