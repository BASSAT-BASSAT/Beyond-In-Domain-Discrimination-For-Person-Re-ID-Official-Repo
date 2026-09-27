"""Market-1501-style CMC / mAP / mINP, ported from Person-ReID-BenchMark ``reid/metrics.py``.

Gallery entries with the same pid *and* camid as the query are discarded (junk). Results
are percentages. Only change vs the benchmark: AP is vectorized (identical values) and a
query chunk without valid queries no longer aborts the chunked evaluation.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
    return features / norms


def cosine_distance(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    return (1.0 - l2_normalize(query_features) @ l2_normalize(gallery_features).T).astype(np.float32)


def _eval_market1501(distmat, query_pids, gallery_pids, query_camids, gallery_camids, max_rank):
    num_query, num_gallery = distmat.shape
    max_rank = min(max_rank, num_gallery)
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    all_cmc: list[np.ndarray] = []
    all_ap: list[float] = []
    all_inp: list[float] = []
    for q_idx in range(num_query):
        order = indices[q_idx]
        remove = (gallery_pids[order] == query_pids[q_idx]) & (gallery_camids[order] == query_camids[q_idx])
        raw_cmc = matches[q_idx][~remove]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        max_pos_idx = int(np.max(np.where(raw_cmc == 1)[0]))
        all_inp.append(float(cmc[max_pos_idx] / (max_pos_idx + 1.0)))
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        precision = raw_cmc.cumsum().astype(np.float64) / np.arange(1, len(raw_cmc) + 1, dtype=np.float64)
        all_ap.append(float((precision * raw_cmc).sum() / raw_cmc.sum()))
    return all_cmc, all_ap, all_inp


def _summarize_rank(all_cmc, all_ap, all_inp) -> dict[str, float]:
    if not all_ap:
        raise RuntimeError("All query identities are missing from the gallery.")
    cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / float(len(all_ap))
    return {
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "Rank-10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "Rank-20": float(cmc[min(19, len(cmc) - 1)] * 100.0),
        "mAP": float(np.mean(all_ap) * 100.0),
        "mINP": float(np.mean(all_inp) * 100.0),
        "num_valid_queries": float(len(all_ap)),
    }


def evaluate_rank(distmat, query_pids, gallery_pids, query_camids, gallery_camids, max_rank: int = 50) -> dict[str, float]:
    return _summarize_rank(*_eval_market1501(distmat, query_pids, gallery_pids, query_camids, gallery_camids, max_rank))


def evaluate_rank_from_features(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int = 50,
    query_chunk: int = 256,
) -> dict[str, float]:
    """Same metrics as :func:`evaluate_rank` without a dense Q x G distance matrix."""
    query = l2_normalize(query_features)
    gallery = l2_normalize(gallery_features)
    all_cmc, all_ap, all_inp = [], [], []
    chunk = max(int(query_chunk), 1)
    for start in range(0, len(query), chunk):
        end = min(start + chunk, len(query))
        distmat = (1.0 - query[start:end] @ gallery.T).astype(np.float32)
        cmc, ap, inp = _eval_market1501(
            distmat, query_pids[start:end], gallery_pids, query_camids[start:end], gallery_camids, max_rank
        )
        all_cmc += cmc
        all_ap += ap
        all_inp += inp
    return _summarize_rank(all_cmc, all_ap, all_inp)


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"R1 {metrics['Rank-1']:.2f} | R5 {metrics['Rank-5']:.2f} | R10 {metrics['Rank-10']:.2f} | "
        f"mAP {metrics['mAP']:.2f} | mINP {metrics['mINP']:.2f} | valid Q {int(metrics['num_valid_queries'])}"
    )
