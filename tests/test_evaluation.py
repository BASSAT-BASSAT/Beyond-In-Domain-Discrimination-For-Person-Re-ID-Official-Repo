"""Metrics and the manifest evaluator, with a fake extractor (no model, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reidfm.data.manifest import Manifest, ManifestRow
from reidfm.evaluation.aggregate import mean_pool_l2
from reidfm.evaluation.evaluator import build_items, evaluate_manifest, evaluate_sets, results_markdown
from reidfm.evaluation.metrics import evaluate_rank, evaluate_rank_from_features
from reidfm.evaluation.sampling import sample_indices


def test_perfect_ranking():
    distmat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    m = evaluate_rank(distmat, np.array([1, 2]), np.array([1, 2, 3]), np.array([0, 0]), np.array([1, 1, 1]))
    assert m["Rank-1"] == pytest.approx(100.0) and m["mAP"] == pytest.approx(100.0)
    assert m["cmc"][0] == pytest.approx(100.0)


def test_same_camera_match_is_junk():
    # Query pid 1 cam 0: gallery[0] is the same pid+cam (junk), gallery[2] the valid match at rank 2.
    distmat = np.array([[0.0, 0.5, 0.9]], dtype=np.float32)
    m = evaluate_rank(distmat, np.array([1]), np.array([1, 2, 1]), np.array([0]), np.array([0, 1, 1]))
    assert m["Rank-1"] == pytest.approx(0.0)
    assert m["mAP"] == pytest.approx(50.0)


def test_chunked_matches_dense():
    rng = np.random.default_rng(0)
    q, g = rng.normal(size=(30, 16)), rng.normal(size=(80, 16))
    qp, gp = rng.integers(0, 10, 30), rng.integers(0, 10, 80)
    qc, gc = rng.integers(0, 3, 30), rng.integers(0, 3, 80)
    qn, gn = q / np.linalg.norm(q, axis=1, keepdims=True), g / np.linalg.norm(g, axis=1, keepdims=True)
    dense = evaluate_rank((1 - qn @ gn.T).astype(np.float32), qp, gp, qc, gc)
    chunked = evaluate_rank_from_features(q, g, qp, gp, qc, gc)
    for key in ("Rank-1", "mAP", "mINP"):
        assert chunked[key] == pytest.approx(dense[key], abs=1e-4)


def test_mean_pool_l2_and_sampling():
    pooled = mean_pool_l2(np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(pooled, [1.0, 0.0])
    assert sample_indices(3, 8) == [0, 1, 2]
    assert len(sample_indices(100, 8)) == 8


class FakeExtractor:
    """Feature = one-hot of the pid encoded in the file name (``<pid>_*.jpg``) + small noise."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.calls = 0

    def extract(self, paths, desc=None):
        self.calls += 1
        rng = np.random.default_rng(0)
        feats = rng.normal(scale=0.01, size=(len(paths), self.dim)).astype(np.float32)
        for i, path in enumerate(paths):
            feats[i, int(Path(path).name.split("_")[0]) % self.dim] += 1.0
        return feats


def _video_manifest(tmp_path: Path) -> Manifest:
    rows = []
    for pid in range(4):
        for split, cam in (("query", "0"), ("gallery", "1")):
            for frame in range(20):
                rows.append(ManifestRow(f"{pid}_{split}_{frame}.jpg", f"test{pid}", cam, "demo", split, f"{split}{pid}", str(frame)))
    return Manifest("demo", tmp_path, rows, {"kind": "video"})


def test_evaluate_manifest_video(tmp_path: Path):
    manifest = _video_manifest(tmp_path)
    items = build_items(manifest, "query", num_frames=8)
    assert len(items) == 4 and all(len(item.paths) == 8 for item in items)
    metrics = evaluate_manifest(manifest, FakeExtractor(), num_frames=8)
    assert metrics["Rank-1"] == pytest.approx(100.0)
    assert metrics["kind"] == "video" and metrics["num_query"] == 4 and metrics["num_gallery"] == 4


def test_evaluate_sets_adds_mean_and_markdown(tmp_path: Path, monkeypatch):
    import reidfm.evaluation.evaluator as evaluator

    monkeypatch.setattr(evaluator, "load_manifest", lambda key, roots=None: _video_manifest(tmp_path))
    results = evaluate_sets(["a", "b"], FakeExtractor())
    assert list(results) == ["a", "b", "mean"]
    assert results["mean"]["mAP"] == pytest.approx(100.0)
    assert "| **mean** |" in results_markdown(results, "demo")
