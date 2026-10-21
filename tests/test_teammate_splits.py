from __future__ import annotations

import json
from pathlib import Path

from reidfm.data.leakage_split import _split_teammate_or_dhash_domain
from reidfm.data.manifest import Manifest, ManifestRow
from reidfm.data.teammate_splits import validate_teammate_split


def test_validate_teammate_split(tmp_path, monkeypatch):
    monkeypatch.setenv("REIDFM_SPLITS_ROOT", str(tmp_path))
    rows = [ManifestRow("a.jpg", "1", "1", "demo", "train"), ManifestRow("b.jpg", "2", "1", "demo", "train")]
    manifest = Manifest("demo", Path("/tmp"), rows)
    (tmp_path / "demo.json").write_text(json.dumps({"train": ["1"], "val": ["2"], "label": "t"}), encoding="utf-8")
    from reidfm.data.teammate_splits import load_teammate_split

    data = load_teammate_split("demo")
    train, val = validate_teammate_split("demo", manifest, data)
    assert train == ["1"] and val == ["2"]
