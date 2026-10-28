"""V3-only freshness-aware MeanMetricCheckpointer.

The Legacy MeanMetricCheckpointer is intentionally unchanged.
Evaluation metrics are valid only at the exact EventStorage iteration
in which the EvalHook recorded them. A stale/missing metric is -1.0,
which means 'not evaluated this checkpoint epoch'.
"""
from __future__ import annotations

from math import isfinite

from fastreid.utils.events import get_event_storage

from reidfm.engine.hooks import MeanMetricCheckpointer


class V3FreshMeanMetricCheckpointer(MeanMetricCheckpointer):
    def _metric(self) -> float:
        storage = get_event_storage()
        recorded = storage.latest().get(self.metric_name)
        if recorded is None:
            return -1.0

        value, metric_iteration = recorded
        if int(metric_iteration) != int(storage.iter):
            return -1.0

        score = float(value)
        if not isfinite(score):
            raise ValueError("Non-finite fresh V3 mean/metric")
        return score
