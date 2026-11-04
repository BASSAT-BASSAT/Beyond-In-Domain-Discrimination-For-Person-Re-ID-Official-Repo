"""Tests for Kaggle reidraw eval row loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reidfm.data.reid_target_reader import normalize_eval_row
from reidfm.data.unseen_targets import iter_target_records, load_target_specs


def test_normalize_eval_row_maps_role_aliases():
    row = normalize_eval_row(
        {"path": "a.jpg", "pid": "7", "role": "Q", "protocol": "market1501_official_v1"},
        dataset="Market1501",
        dataset_root_id="Market1501",
    )
    assert row["retrieval_role"] == "query"
    assert row["relative_path"] == "a.jpg"
    assert row["identity_key"] == "7"


def test_iter_target_records_uses_bundle_reader(tmp_path: Path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "Market1501.reidraw").write_bytes(b"placeholder")
    (root / "MSMT17_V2.reidraw").write_bytes(b"x")
    (root / "MARS.reidraw").write_bytes(b"x")
    (root / "DukeMTMC.reidraw").write_bytes(b"x")
    (root / "OccludedDukeMTMC.reidraw").write_bytes(b"x")
    rows = [
        {
            "sample_uid": "q1",
            "dataset": "Market1501",
            "dataset_root_id": "Market1501",
            "relative_path": "query/1.jpg",
            "identity_key": "1",
            "validation_protocol_id": "market1501_official_v1",
            "retrieval_role": "query",
            "raw_camera_or_view": "1",
        },
        {
            "sample_uid": "g1",
            "dataset": "Market1501",
            "dataset_root_id": "Market1501",
            "relative_path": "gallery/1.jpg",
            "identity_key": "1",
            "validation_protocol_id": "market1501_official_v1",
            "retrieval_role": "gallery",
            "raw_camera_or_view": "2",
        },
    ]
    for stem in ("MSMT17_V2", "MARS", "DukeMTMC", "OccludedDukeMTMC"):
        (root / f"{stem}.reidraw").write_bytes(b"x")
    (root / "TeamRecorded_manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    (root / "reid_target_reader.py").write_text(
        "def iter_eval_manifest_rows(reidraw_path, dataset=None):\n"
        "    return " + repr(rows) + "\n",
        encoding="utf-8",
    )
    (root / "reid_targets_index.json").write_text(
        json.dumps(
            {
                "targets": {
                    "Market1501": {"dataset": "Market1501", "kaggle_file": "Market1501.reidraw"},
                }
            }
        ),
        encoding="utf-8",
    )
    specs = load_target_specs(root)
    market_rows = list(iter_target_records(specs["market1501"], root))
    assert len(market_rows) == 2
    assert {r["retrieval_role"] for r in market_rows} == {"query", "gallery"}
