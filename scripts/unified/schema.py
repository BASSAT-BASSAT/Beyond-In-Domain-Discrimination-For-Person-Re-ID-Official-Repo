
"""
Unified Dataset Schema — Version 1

Defines the common column structure used by all
Person ReID source candidate manifests.

This module does not modify images or CSV files.
"""

import csv
from pathlib import Path

SCHEMA_VERSION = "v1"

UNIFIED_COLUMNS = (
    "sample_uid",
    "dataset",
    "candidate_pool",
    "dataset_root_id",
    "relative_path",
    "source_split",
    "raw_pid",
    "identity_namespace",
    "identity_key",
    "raw_camera_or_view",
    "camera_semantics",
    "camera_context_id",
    "video_id",
    "tracklet_id",
    "frame_id",
    "clothes_id",
    "source_row_number",
    "source_image_index",
    "source_metadata_json",
)


def validate_header(columns):
    """Ensure a CSV follows the unified v1 schema."""

    if columns is None:
        raise ValueError("CSV header is missing.")

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate column names detected.")

    if tuple(columns) != UNIFIED_COLUMNS:
        raise ValueError(
            "CSV columns do not match Unified Schema v1.\n"
            f"Expected: {UNIFIED_COLUMNS}\n"
            f"Received: {columns}"
        )

    return True


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    reference = (
        project_root
        / "unified"
        / "v1"
        / "cuhk03_np_unified_v1.csv"
    )

    with reference.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        validate_header(reader.fieldnames)

    print("Schema version:", SCHEMA_VERSION)
    print("Expected columns:", len(UNIFIED_COLUMNS))
    print("Reference:", reference)
    print("CUHK03 header validation: PASS")
