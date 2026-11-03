"""Tests for unseen target index parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reidfm.data.unseen_targets import load_target_specs, _skip_target


def test_skip_teamrecorded():
    assert _skip_target("TeamRecorded")
    assert _skip_target("team_recorded")
    assert not _skip_target("Market1501")


def test_load_target_specs_from_index(tmp_path: Path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "Market1501.reidraw").write_bytes(b"x")
    (root / "MSMT17_V2.reidraw").write_bytes(b"x")
    (root / "MARS.reidraw").write_bytes(b"x")
    (root / "DukeMTMC.reidraw").write_bytes(b"x")
    (root / "OccludedDukeMTMC.reidraw").write_bytes(b"x")
    (root / "TeamRecorded.reidraw").write_bytes(b"x")
    manifest = {
        "sample_uid": "a",
        "dataset": "Market1501",
        "dataset_root_id": "Market1501",
        "relative_path": "img.jpg",
        "identity_key": "1",
        "validation_protocol_id": "market1501_official_v1",
        "retrieval_role": "query",
    }
    (root / "Market1501_manifest.json").write_text(json.dumps([manifest]), encoding="utf-8")
    for stem in ("MSMT17_V2", "MARS", "DukeMTMC", "OccludedDukeMTMC"):
        row = dict(manifest)
        row["dataset_root_id"] = stem
        row["dataset"] = stem
        (root / f"{stem}_manifest.json").write_text(json.dumps([row]), encoding="utf-8")
    (root / "reid_targets_index.json").write_text(
        json.dumps(
            {
                "targets": {
                    "market1501": {"reidraw_file": "Market1501.reidraw"},
                    "TeamRecorded": {"reidraw_file": "TeamRecorded.reidraw"},
                }
            }
        ),
        encoding="utf-8",
    )
    specs = load_target_specs(root)
    assert "market1501" in specs
    assert "teamrecorded" not in specs
    assert len(specs) == 5
