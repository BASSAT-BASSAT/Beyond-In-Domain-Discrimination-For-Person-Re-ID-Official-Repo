"""Train/val split policies and ORBench converter."""

from __future__ import annotations

import json
from pathlib import Path

from reidfm.data.converters.orbench_rgb import convert as convert_orbench
from reidfm.data.converters.prid2011 import convert as convert_prid
from reidfm.data.manifest import Manifest, ManifestRow
from reidfm.data.split_policies import (
    CUSTOM_VAL_LABEL,
    get_split_policy,
    identity_disjoint_train_val,
    train_pids_from_manifest,
)


def test_orbench_policy_modes():
    assert get_split_policy("orbenchrgb", "cluster").kind == "identity_disjoint"
    assert get_split_policy("orbenchrgb", "official_native").kind == "official_native"


def test_official_native_uses_all_train_pids():
    from reidfm.data.leakage_split import _split_official_native_domain

    rows = [
        ManifestRow("a.jpg", "1", "1", "demo", "train"),
        ManifestRow("b.jpg", "2", "1", "demo", "train"),
        ManifestRow("q.jpg", "t9", "1", "demo", "query"),
        ManifestRow("g.jpg", "t9", "2", "demo", "gallery"),
    ]
    m = Manifest("demo", Path("/tmp"), rows)
    train, val, _, meta = _split_official_native_domain("demo", m, in_training_val=True)
    assert train == ["1", "2"]
    assert val == []
    assert meta["in_training_eval"] is True


def test_identity_disjoint_no_overlap():
    pids = [f"id{i}" for i in range(20)]
    train, val, meta = identity_disjoint_train_val(pids, train_fraction=0.85, seed=0)
    assert set(train) & set(val) == set()
    assert set(train) | set(val) == set(pids)
    assert meta["custom_val"] == CUSTOM_VAL_LABEL


def test_orbench_converter_official_files(tmp_path: Path):
    root = tmp_path / "ORBench"
    vis = root / "vis" / "0001"
    vis.mkdir(parents=True)
    img = vis / "a.jpg"
    img.write_bytes(b"x")
    train = [{"file_name": "vis/0001/a.jpg", "pid": "1", "modality": "vis"}]
    (root / "train_annos.json").write_text(json.dumps(train), encoding="utf-8")
    test = {
        "query": [{"file_name": "vis/0001/a.jpg", "pid": "99", "modality": "nir"}],
        "gallery": [{"file_name": "vis/0001/a.jpg", "pid": "99", "modality": "vis"}],
    }
    (root / "test_gallery_and_queries.json").write_text(json.dumps(test), encoding="utf-8")
    rows = convert_orbench(root, {"rgb_only": True})
    train_pids = {r.pid for r in rows if r.split == "train"}
    test_pids = {r.pid for r in rows if r.split != "train"}
    assert "1" in train_pids
    assert all(p.startswith("test") for p in test_pids)


def test_last_train_only_in_train_eval_in_query_gallery(tmp_path: Path):
    from reidfm.data.converters.last import convert as convert_last

    root = tmp_path / "last_ds"
    (root / "train" / "1").mkdir(parents=True)
    (root / "train" / "1" / "1_0_0_0_0_1.jpg").write_bytes(b"x")
    for split in ("val", "test"):
        (root / split / "query").mkdir(parents=True, exist_ok=True)
        (root / split / "gallery" / "9").mkdir(parents=True, exist_ok=True)
        (root / split / "query" / "9_0_1.jpg").write_bytes(b"x")
        (root / split / "gallery" / "9" / "9_0_2.jpg").write_bytes(b"x")
    rows = convert_last(root, {"eval_splits": ["val", "test"]})
    assert sum(1 for r in rows if r.split == "train") == 1
    eval_rows = [r for r in rows if r.split != "train"]
    assert len(eval_rows) == 4
    assert {r.pid for r in eval_rows} == {"val_9", "test_9"}


def test_prid_train_pids_disjoint_from_test(tmp_path: Path):
    shots = tmp_path / "multi_shot"
    for cam in ("cam_a", "cam_b"):
        for pid in (1, 2):
            folder = shots / cam / f"person_{pid:04d}"
            folder.mkdir(parents=True)
            (folder / "f.png").write_bytes(b"x")
    rows = convert_prid(tmp_path, {"include_train": True, "num_ids": 2})
    train_pids = {r.pid for r in rows if r.split == "train"}
    eval_pids = {r.pid for r in rows if r.split != "train"}
    assert train_pids == {"tr1", "tr2"}
    assert eval_pids == {"1", "2"}
    assert train_pids.isdisjoint(eval_pids)


def test_train_pids_from_manifest_only_train_split():
    rows = [
        ManifestRow("a.jpg", "1", "1", "demo", "train"),
        ManifestRow("b.jpg", "2", "1", "demo", "query"),
    ]
    m = Manifest("demo", Path("/tmp"), rows)
    assert train_pids_from_manifest(m) == ["1"]
