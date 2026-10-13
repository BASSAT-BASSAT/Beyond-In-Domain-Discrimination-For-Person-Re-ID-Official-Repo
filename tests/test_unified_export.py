"""Unified v1 CSV -> reidfm manifest export."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.unified.schema import UNIFIED_COLUMNS

from reidfm.data.manifest import read_manifest
from reidfm.data.unified_export import export_unified_csv


def _write_minimal_unified(path: Path, rel: str, image: Path) -> None:
    row = {col: "" for col in UNIFIED_COLUMNS}
    row.update(
        {
            "sample_uid": "abc",
            "dataset": "IUST_PersonReID",
            "dataset_root_id": "IUST_PersonReID",
            "relative_path": rel,
            "source_split": "train",
            "raw_pid": "7",
            "identity_key": "IUST_PersonReID::train::7",
            "camera_context_id": "cam1",
            "clothes_id": "0",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg


def test_unified_export_roundtrip(tmp_path: Path):
    root = tmp_path / "IUST_PersonReID"
    img = root / "p" / "a.jpg"
    unified = tmp_path / "u.csv"
    _write_minimal_unified(unified, "p/a.jpg", img)
    out = tmp_path / "pool" / "iust"
    meta = export_unified_csv(unified, out, dataset_key="iust", raw_roots=tmp_path, path_mode="relative")
    rows = read_manifest(out / "manifest.csv")
    assert meta["dataset"] == "iust"
    assert len(rows) == 1 and rows[0].pid == "7" and rows[0].split == "train"
