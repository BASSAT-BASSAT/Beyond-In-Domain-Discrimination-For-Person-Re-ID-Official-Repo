
"""
RPIfield Known Actors Adapter — Unified Schema v1.

Converts the validated Known-Actors candidate pool
into unified metadata rows.

Important:
- No train/validation protocol assigned yet.
- Appearance labels are not assumed clothes IDs.
- Sequence IDs are not assumed tracklet IDs.
- Direct execution is read-only.
"""

import csv
import json
import hashlib

from pathlib import Path

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_NAME = "RPIfield"

DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "RPIfield"
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "RPIfield"
    / "rpifield_known_actors_candidates_v0.csv"
)

EXCLUSIONS_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "RPIfield"
    / "rpifield_hard_exclusions_v0.csv"
)

IDENTITY_NAMESPACE = "RPIfield::known_actor"


def adapt_row(source_row, row_number):
    """Convert one Known Actor candidate to Unified v1."""

    # 1. Validate identity category
    assert source_row["dataset"] == DATASET_NAME
    assert source_row["identity_type"] == "known_actor"
    assert source_row["is_known_actor"] == "True"
    assert source_row["is_distractor"] == "False"
    assert source_row["exists"] == "True"

    pid = source_row["person_id"]
    camera = source_row["camera_id"]
    frame = source_row["frame_id"]
    appearance = source_row["appearance_label"]

    assert pid.isdigit()
    assert camera.isdigit()
    assert 1 <= int(camera) <= 12
    assert frame.isdigit()
    assert appearance

    # 2. Validate camera and sequence metadata
    assert source_row["camera"] == f"Cam_{camera}"

    expected_sequence = (
        f"cam{int(camera):02d}:{appearance}"
    )

    assert source_row["sequence_id"] == expected_sequence

    # Preserve original frame indexing.
    assert source_row["timestamp"] == frame

    expected_filename = f"Cam_{camera}f_{frame}.png"

    assert source_row["filename"] == expected_filename

    # 3. Validate original relative path
    original_relative = Path(source_row["relative_path"])

    assert not original_relative.is_absolute()
    assert ".." not in original_relative.parts

    expected_relative = (
        Path("Data")
        / f"Cam_{camera}"
        / appearance
        / expected_filename
    )

    assert original_relative == expected_relative

    # 4. Resolve actual image
    root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        source_row["path"]
    ).resolve(strict=True)

    assert image_path.is_file()

    expected_path = (
        root
        / "extracted"
        / "RPIfield_v0"
        / original_relative
    ).resolve(strict=True)

    assert image_path == expected_path

    relative_path = image_path.relative_to(root).as_posix()

    # 5. Deterministic sample UID
    uid_input = DATASET_NAME + "\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 6. Preserve all original metadata
    # except machine-specific absolute path.
    metadata = {
        key: value
        for key, value in source_row.items()
        if key != "path"
    }

    # 7. Build unified row
    unified = {
        column: ""
        for column in UNIFIED_COLUMNS
    }

    unified.update({
        "sample_uid": sample_uid,
        "dataset": DATASET_NAME,
        "candidate_pool": (
            "rpifield_known_actors_candidates_v0"
        ),
        "dataset_root_id": DATASET_NAME,
        "relative_path": relative_path,
        "source_split": "unassigned",
        "raw_pid": pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": f"{IDENTITY_NAMESPACE}::{pid}",
        "raw_camera_or_view": camera,
        "camera_semantics": "physical_camera_code",
        "camera_context_id": (
            f"RPIfield::camera::{camera}"
        ),
        "frame_id": frame,
        "source_row_number": str(row_number),
        "source_metadata_json": json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")
        ),
    })

    return unified


def validate_adapter():
    """Validate RPIfield candidates without writing files."""

    # 1. Load documented hard exclusions
    excluded_paths = set()
    excluded_known = 0

    with EXCLUSIONS_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            assert row["hard_exclude"] == "true"

            path = row["path"]

            assert path not in excluded_paths
            excluded_paths.add(path)

            if row["identity_type"] == "known_actor":
                excluded_known += 1

    assert len(excluded_paths) == 1263
    assert excluded_known == 160

    # 2. Validate cleaned candidate pool
    count = 0

    identities = set()
    cameras = set()
    sample_uids = set()
    relative_paths = set()

    first_example = None

    with SOURCE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "dataset", "camera_id", "camera",
            "person_id", "identity_type",
            "is_known_actor", "is_distractor",
            "appearance_label", "reappearance_index",
            "sequence_id", "frame_id", "timestamp",
            "filename", "relative_path", "path", "exists"
        }

        assert required.issubset(reader.fieldnames or [])

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            # Confirm no excluded image was retained
            assert source_row["path"] not in excluded_paths

            unified = adapt_row(
                source_row, row_number
            )

            # Common schema
            assert tuple(unified) == UNIFIED_COLUMNS

            # Metadata integrity
            metadata = json.loads(
                unified["source_metadata_json"]
            )

            assert metadata == {
                key: value
                for key, value in source_row.items()
                if key != "path"
            }

            assert unified["raw_pid"] == source_row["person_id"]
            assert unified["frame_id"] == source_row["frame_id"]
            assert unified["source_split"] == "unassigned"

            # No unsupported metadata inference
            assert unified["video_id"] == ""
            assert unified["tracklet_id"] == ""
            assert unified["clothes_id"] == ""
            assert unified["source_image_index"] == ""

            # Unique sample checks
            uid = unified["sample_uid"]
            relative = unified["relative_path"]

            assert uid not in sample_uids
            assert relative not in relative_paths

            sample_uids.add(uid)
            relative_paths.add(relative)

            identities.add(unified["identity_key"])
            cameras.add(unified["raw_camera_or_view"])

            if first_example is None:
                first_example = unified

            count += 1

    # 3. Final audit-based checks
    assert count == 211484
    assert len(identities) == 112
    assert len(cameras) == 12

    assert len(sample_uids) == count
    assert len(relative_paths) == count

    print("RPIFIELD ADAPTER VALIDATION")
    print("-" * 50)

    print("Known actor candidates:", count)
    print("Known actor identities:", len(identities))
    print("Camera codes:", len(cameras))
    print("Known actor hard exclusions:", excluded_known)
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")

    for field in (
        "relative_path",
        "identity_key",
        "source_split",
        "raw_camera_or_view",
        "camera_context_id",
        "frame_id",
        "tracklet_id",
        "clothes_id"
    ):
        print(f"{field}: {first_example[field]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nRPIFIELD ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
