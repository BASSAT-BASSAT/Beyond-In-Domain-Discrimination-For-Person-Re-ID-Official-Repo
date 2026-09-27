"""Benchmark-compatible evaluation: metrics, tracklet pooling, extractors, manifest evaluator."""

from reidfm.evaluation.aggregate import mean_pool_l2
from reidfm.evaluation.metrics import cosine_distance, evaluate_rank, evaluate_rank_from_features, format_metrics
from reidfm.evaluation.sampling import sample_indices

__all__ = [
    "cosine_distance",
    "evaluate_rank",
    "evaluate_rank_from_features",
    "format_metrics",
    "mean_pool_l2",
    "sample_indices",
]
