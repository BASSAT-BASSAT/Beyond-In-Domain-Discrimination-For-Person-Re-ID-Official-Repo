"""Tests for REID_RAW_ARCHIVE_MIRROR zip/tar eval targets."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from reidfm.data.raw_archive_mirror import (
    RawArchiveImageReader,
    detect_archive_kind,
    iter_eval_rows_from_raw_mirror,
    iter_manifest_rows_from_archive,
)
from reidfm.data.reid_target_reader import iter_eval_rows


def _write_market_zip(path: Path) -> None:
    manifest = (
        "sample_uid,dataset,dataset_root_id,relative_path,identity_key,"
        "validation_protocol_id,retrieval_role,raw_camera_or_view\n"
        "Market1501::q1,Market1501,Market1501,images/q1.jpg,1,market1501_official_v1,query,1\n"
        "Market1501::g1,Market1501,Market1501,images/g1.jpg,1,market1501_official_v1,gallery,2\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("validation_manifest.csv", manifest)
        archive.writestr("images/q1.jpg", b"\xff\xd8\xff\xd9")
        archive.writestr("images/g1.jpg", b"\xff\xd8\xff\xd9")
    path.write_bytes(buf.getvalue())


def test_detect_zip_mirror(tmp_path: Path):
    reidraw = tmp_path / "Market1501.reidraw"
    _write_market_zip(reidraw)
    assert detect_archive_kind(reidraw) == "zip"
    rows = iter_manifest_rows_from_archive(reidraw)
    assert len(rows) == 2


def test_iter_eval_rows_from_zip_mirror(tmp_path: Path):
    reidraw = tmp_path / "Market1501.reidraw"
    _write_market_zip(reidraw)
    rows = list(
        iter_eval_rows_from_raw_mirror(
            reidraw,
            dataset="Market1501",
            dataset_root_id="Market1501",
        )
    )
    assert len(rows) == 2
    assert {r["retrieval_role"] for r in rows} == {"query", "gallery"}


def test_raw_archive_image_reader_suffix_match(tmp_path: Path):
    reidraw = tmp_path / "Market1501.reidraw"
    _write_market_zip(reidraw)
    with RawArchiveImageReader(reidraw) as reader:
        assert reader.read_bytes("images/q1.jpg") == b"\xff\xd8\xff\xd9"
        assert reader.read_bytes("nested/deep/q1.jpg") == b"\xff\xd8\xff\xd9"


def test_iter_eval_rows_prefers_bundle_script(tmp_path: Path):
    root = tmp_path / "bundle"
    root.mkdir()
    reidraw = root / "Market1501.reidraw"
    _write_market_zip(reidraw)
    rows = [
        {
            "sample_uid": "u1",
            "relative_path": "images/q1.jpg",
            "identity_key": "9",
            "validation_protocol_id": "market1501_official_v1",
            "retrieval_role": "query",
            "raw_camera_or_view": "1",
        }
    ]
    (root / "reid_target_reader.py").write_text(
        "def read_eval_rows(reidraw_path, dataset=None, bundle_root=None):\n"
        "    return " + json.dumps(rows) + "\n",
        encoding="utf-8",
    )
    out = list(
        iter_eval_rows(
            reidraw,
            bundle_root=root,
            dataset="Market1501",
            dataset_root_id="Market1501",
        )
    )
    assert len(out) == 1
    assert out[0]["identity_key"] == "9"
