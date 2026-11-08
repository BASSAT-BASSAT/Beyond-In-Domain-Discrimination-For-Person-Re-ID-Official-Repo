"""Manifest format, converters (synthetic Market layout), subsampling, pool validation, audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from reidfm.data.audit import audit_manifest, check_pool_roles
from reidfm.data.converters import build_dataset, collect_rows
from reidfm.data.converters.base import SourceRow, cap_per_identity, linspace_indices, sample_eval_tracklets
from reidfm.data.manifest import ManifestRow, find_manifest, load_manifest, read_manifest, validate_rows, write_manifest
from reidfm.data.pool import PoolConfig, load_pool
from reidfm.utils.config import expand_env


def test_expand_env(monkeypatch):
    monkeypatch.setenv("REIDFM_TEST_A", "/data")
    monkeypatch.delenv("REIDFM_TEST_B", raising=False)
    value = {"a": "${REIDFM_TEST_A}/x", "b": ["${REIDFM_TEST_B:-fallback}"], "c": "${REIDFM_TEST_B:-${REIDFM_TEST_A}}"}
    assert expand_env(value) == {"a": "/data/x", "b": ["fallback"], "c": "/data"}


def test_manifest_roundtrip_and_validation(tmp_path: Path):
    rows = [
        ManifestRow("images/a.jpg", "1", "1", "demo", "train"),
        ManifestRow("images/b.jpg", "test1", "2", "demo", "query", "t1", "3"),
    ]
    path = write_manifest(tmp_path / "demo" / "manifest.csv", rows)
    assert read_manifest(path) == rows
    assert validate_rows(rows) == []
    bad = rows + [ManifestRow("images/a.jpg", "1", "1", "demo", "train"), ManifestRow("x.jpg", "1", "1", "de_mo", "val")]
    problems = " ".join(validate_rows(bad))
    assert "bad split" in problems and "must not contain '_'" in problems


def test_find_manifest_one_level_deeper(tmp_path: Path):
    write_manifest(tmp_path / "kaggle-slug" / "demo" / "manifest.csv", [ManifestRow("a.jpg", "1", "1", "demo", "train")])
    assert find_manifest("demo", [tmp_path]) == tmp_path / "kaggle-slug" / "demo" / "manifest.csv"
    with pytest.raises(FileNotFoundError):
        find_manifest("missing", [tmp_path])


def test_market_converter_and_build(market_like: Path, tmp_path: Path):
    _, rows = collect_rows("market1501", {}, raw_root=market_like)
    by_split = {s: [r for r in rows if r.split == s] for s in ("train", "query", "gallery")}
    assert len(by_split["train"]) == 3 * 2 * 2
    assert len(by_split["query"]) == 2 * 2
    assert {r.pid for r in by_split["gallery"]} == {"0", "10", "11"}  # -1 junk dropped, 0 distractor kept

    meta = build_dataset("market1501", tmp_path / "pool", raw_root=market_like, mode="copy", workers=2)
    manifest = load_manifest("market1501", [tmp_path / "pool"])
    assert meta["dataset"] == "market1501" and manifest.has_train and manifest.has_test
    assert all(manifest.resolve(r).is_file() for r in manifest.rows)
    report = audit_manifest(manifest)
    assert report.errors == []

    eval_only = build_dataset("market1501", tmp_path / "pool2", raw_root=market_like, mode="link", include_train=False)
    assert eval_only["include_train"] is False
    assert not load_manifest("market1501", [tmp_path / "pool2"]).has_train


def test_linspace_indices_matches_benchmark_rule():
    assert linspace_indices(5, 8) == [0, 1, 2, 3, 4]
    assert linspace_indices(100, 8) == [0, 14, 28, 42, 56, 70, 84, 99]


def _video_rows(n_frames: int, split: str, pid: str = "1", cam: str = "1", tracklet: str = "t0"):
    return [SourceRow(Path(f"{tracklet}_{i}.jpg"), pid, cam, split, tracklet, i) for i in range(n_frames)]


def test_sample_eval_tracklets_keeps_train():
    rows = _video_rows(50, "query") + _video_rows(50, "train", tracklet="t1")
    kept = sample_eval_tracklets(rows, 8)
    assert sum(r.split == "query" for r in kept) == 8
    assert sum(r.split == "train" for r in kept) == 50


def test_cap_per_identity_spreads_over_cameras():
    rows = _video_rows(40, "train", cam="1", tracklet="a") + _video_rows(4, "train", cam="2", tracklet="b")
    kept = cap_per_identity(rows, 10)
    assert len(kept) == 10
    assert sum(r.camid == "2" for r in kept) == 4  # the small camera is kept whole


def test_pool_validation_rejects_ood_sources():
    pool = PoolConfig("x", sources=["market1501", "prid2011"], ood=["prid2011"], val_sets=[], test_sets=[])
    with pytest.raises(ValueError):
        pool.validate()
    pool = PoolConfig("x", sources=["market1501"], ood=[], val_sets=["mot17"], test_sets=[])
    with pytest.raises(ValueError, match="without query/gallery"):
        pool.validate()


def test_shipped_pools_are_valid():
    pool_v2 = load_pool("configs/datasets/pool_v2.yaml")
    assert "prw" not in pool_v2.sources
    assert "market1501" not in pool_v2.sources
    assert {"prid2011", "iust"} <= set(pool_v2.sources)
    assert pool_v2.ood == []
    assert set(pool_v2.test_sets) == {"market1501", "msmt17", "dukemtmc", "occludedduke"}
    assert "mot17" not in pool_v2.sources and "mot20" not in pool_v2.sources
    load_pool("configs/datasets/pool_smoke.yaml")


def test_train_split_filter():
    from reidfm.data.split_filter import train_pid_allowed

    allowed = {"demo": {"1", "2"}}
    assert train_pid_allowed("demo", "1", allowed)
    assert not train_pid_allowed("demo", "9", allowed)
    assert not train_pid_allowed("other", "1", allowed)
    assert train_pid_allowed("demo", "9", {})


def test_check_pool_roles_flags_missing_splits(market_like: Path, tmp_path: Path):
    build_dataset("market1501", tmp_path / "pool", raw_root=market_like, include_train=False)
    manifests = {"market1501": load_manifest("market1501", [tmp_path / "pool"])}
    pool = PoolConfig("x", sources=["market1501"], ood=[], val_sets=["market1501"], test_sets=[])
    assert any("no train rows" in e for e in check_pool_roles(pool, manifests))
