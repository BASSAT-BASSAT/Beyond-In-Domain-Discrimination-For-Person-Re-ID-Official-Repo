#!/usr/bin/env python3
"""Export only header + N data rows from verified Global P3/P5 manifests.

Read-only for full manifests/other artifacts; writes exactly two small preview CSVs.
Run from repository root: python scripts/dg_reid_data_pipeline/create_global_previews.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from pathlib import Path

TRAIN_FIELDS = (
    "sample_uid,dataset,dataset_root_id,relative_path,identity_key,class_id,"
    "raw_camera_or_view,camera_semantics,camera_context_id,video_id,tracklet_id,"
    "frame_id,clothes_id,candidate_pool,source_row_number"
).split(",")
VAL_FIELDS = (
    "sample_uid,dataset,dataset_root_id,relative_path,identity_key,"
    "validation_protocol_id,retrieval_role,raw_camera_or_view,camera_semantics,"
    "camera_context_id,video_id,tracklet_id,frame_id,clothes_id,candidate_pool,"
    "source_row_number"
).split(",")
FILES = (
    ("train_manifest.csv", "train_manifest_preview.csv", TRAIN_FIELDS),
    ("validation_manifest.csv", "validation_manifest_preview.csv", VAL_FIELDS),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Missing Global checksums: {path}")
    expected = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        pieces = line.strip().split(maxsplit=1)
        if len(pieces) != 2:
            raise ValueError(f"Invalid checksum line: {line!r}")
        hash_value, file_name = pieces
        file_name = file_name.lstrip("* ").replace("\\", "/")
        if file_name in expected:
            raise ValueError(f"Duplicate checksum entry: {file_name}")
        expected[file_name] = hash_value.lower()
    return expected


def get_preview_rows(src: Path, columns: list[str], count: int) -> list[list[str]]:
    with src.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        head = next(reader, None)
        if head != columns:
            raise ValueError(f"Unexpected schema in {src}: {head!r}")
        rows = []
        for i in range(count):
            row = next(reader, None)
            if row is None:
                raise ValueError(f"Only {i} data rows available in {src}, expected {count}")
            if len(row) != len(head):
                raise ValueError(f"Invalid column count for preview row {i + 1} in {src}")
            rows.append(row)
        return rows


def write_csv_atomic(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    file_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=".preview_", suffix=".tmp", delete=False) as file:
            file_name = file.name
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
        os.replace(file_name, path)
    finally:
        if file_name and os.path.exists(file_name):
            os.unlink(file_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root; default current working directory")
    parser.add_argument("--rows", type=int, default=10, help="Number of data rows, default 10")
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("--rows must be > 0")
    out = args.repo.resolve() / "unified" / "final_dataset" / "Global"
    hashes = load_checksums(out / "checksums.sha256")
    prepared = []
    for src_name, preview_name, cols in FILES:
        src = out / src_name
        if not src.is_file():
            raise ValueError(f"Missing full Global manifest: {src}")
        expected = hashes.get(src_name)
        if expected is None:
            raise ValueError(f"No checksum for {src_name} in Global checksums.sha256")
        actual = sha256(src)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {src_name}: expected={expected}, actual={actual}")
        rows = get_preview_rows(src, cols, args.rows)
        prepared.append((preview_name, cols, rows))
        print(f"[PASS] {src_name}: hash and schema; extracted {len(rows)} samples")

    preview_dir = out / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for filename, cols, rows in prepared:
        dest = preview_dir / filename
        write_csv_atomic(dest, cols, rows)
        print(f"[OK] {dest} ({len(rows)} data rows + header)")
    print("[DONE] Both full Global manifests and original checksums are unchanged.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as error:
        raise SystemExit(f"[STOP] {error}") from error
