"""Tracklet-level (MARS) pooling for held-out V3 evaluation."""

from __future__ import annotations

import numpy as np

from reidfm.evaluation.v3_evaluator import mean_pool_tracklets, sample_tracklet_records
from reidfm.evaluation.v3_metrics import evaluate_v3_rank_from_features


def _row(role, pid, cam, track, frame, tracklet_id=""):
    name = f"{pid:04d}C{cam}T{track:04d}F{frame:03d}.jpg"
    return {
        "sample_uid": f"MARS::{role}::{name}",
        "dataset": "MARS",
        "dataset_root_id": "MARS",
        "relative_path": f"bbox_test/{pid:04d}/{name}",
        "identity_key": f"test{pid}",
        "validation_protocol_id": "mars_official_v1",
        "retrieval_role": role,
        "raw_camera_or_view": str(cam),
        "camera_context_id": "",
        "tracklet_id": tracklet_id,
    }


def test_frames_group_into_tracklets_from_filenames():
    rows = [_row("query", 1, 1, 1, f) for f in range(1, 21)] + [_row("query", 2, 3, 7, f) for f in range(1, 4)]
    sampled, owners, tracklets = sample_tracklet_records(rows, num_frames=8)
    assert len(tracklets) == 2
    assert np.bincount(owners).tolist() == [8, 3]
    assert len(sampled) == 11
    assert tracklets[0]["identity_key"] == "test1"
    assert tracklets[0]["sample_uid"] != tracklets[1]["sample_uid"]


def test_explicit_tracklet_id_wins_over_filename():
    rows = [_row("gallery", 5, 2, 1, f, tracklet_id="gallery000010") for f in range(1, 5)]
    rows += [_row("gallery", 5, 2, 1, f, tracklet_id="gallery000011") for f in range(5, 9)]
    _, _, tracklets = sample_tracklet_records(rows, num_frames=8)
    assert len(tracklets) == 2


def test_mean_pool_is_l2_normalised_per_tracklet():
    feats = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0]], dtype=np.float32)
    pooled = mean_pool_tracklets(feats, np.array([0, 0, 1]), 2)
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)
    assert np.allclose(pooled[1], [1.0, 0.0])


def test_tracklet_rows_rank_cross_camera():
    query = [_row("query", 1, 1, 1, f) for f in range(1, 5)]
    gallery = [_row("gallery", 1, 2, 3, f) for f in range(1, 5)] + [_row("gallery", 9, 2, 4, f) for f in range(1, 5)]
    _, q_owner, q_tracks = sample_tracklet_records(query, 8)
    _, g_owner, g_tracks = sample_tracklet_records(gallery, 8)
    qf = mean_pool_tracklets(np.tile([[1.0, 0.0]], (len(q_owner), 1)), q_owner, len(q_tracks))
    g_frames = np.array([[1.0, 0.1]] * 4 + [[0.0, 1.0]] * 4, dtype=np.float32)
    gf = mean_pool_tracklets(g_frames, g_owner, len(g_tracks))
    metrics = evaluate_v3_rank_from_features(qf, gf, q_tracks, g_tracks, "mars_official_v1", max_rank=2)
    assert metrics["num_query"] == 1 and metrics["num_gallery"] == 2
    assert metrics["Rank-1"] == 100.0
