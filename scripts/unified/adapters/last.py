
"""
LaST Person ReID Adapter — Unified Schema v1.

Converts LaST training candidates into unified rows.

Read-only validation when executed directly.
No CSV files are created or modified.
"""

import csv
import json
import hashlib

from collections import defaultdict
from pathlib import Path

from reidfm.data.identity import visual_train_pid

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "LaST"

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "LaST"
    / "last_clean_train_candidates_v0.csv"
)

IDENTITY_NAMESPACE = "LaST::train"


def adapt_row(source_row, row_number):
    """Convert one LaST candidate to Unified Schema v1."""

    # 1. Validate training metadata
    assert source_row["split"] == "train"
    assert source_row["is_distractor"] == "False"
    assert source_row["filename_anomaly"] == "False"
    assert source_row["folder_pid_mismatch"] == "False"

    original_fields = (
        "pid", "image_id", "video_id",
        "frame_id", "bbox_id", "clothes_id"
    )

    assert all(
        source_row[field].isdigit()
        for field in original_fields
    )

    # 2. Check metadata against filename
    filename = source_row["filename"]
    parts = Path(filename).stem.split("_")

    assert len(parts) == 6
    assert tuple(map(int, parts)) == tuple(
        int(source_row[field])
        for field in original_fields
    )

    # 3. Resolve portable path
    dataset_root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        source_row["path"]
    ).resolve(strict=True)

    assert image_path.is_file()

    relative_path = image_path.relative_to(
        dataset_root
    ).as_posix()

    assert Path(relative_path).name == filename

    # 4. Generate deterministic sample UID
    uid_input = "LaST\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 5. Build identity namespace
    pid = source_row["pid"]
    visual_pid = visual_train_pid(pid, source_row["clothes_id"])
    identity_key = f"{IDENTITY_NAMESPACE}::{visual_pid}"

    # 6. Preserve dataset-specific metadata
    source_metadata = {
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
        "dataset": "LaST",
        "candidate_pool": "last_clean_train_candidates_v0",
        "dataset_root_id": "LaST",
        "relative_path": relative_path,
        "source_split": "train",
        "raw_pid": pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": identity_key,
        "camera_semantics": "not_available",
        "video_id": source_row["video_id"],
        "frame_id": source_row["frame_id"],
        "clothes_id": source_row["clothes_id"],
        "source_image_index": source_row["image_id"],
        "source_row_number": str(row_number),
        "source_metadata_json": json.dumps(
            source_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")
        ),
    })

    return unified


def validate_adapter():
    """Validate LaST adapter without writing any files."""

    count = 0

    identities = set()
    sample_uids = set()
    relative_paths = set()
    clothes_by_pid = defaultdict(set)

    first_example = None

    with SOURCE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "filename", "split", "pid", "image_id",
            "video_id", "frame_id", "bbox_id",
            "clothes_id", "folder_pid", "is_distractor",
            "filename_anomaly", "folder_pid_mismatch",
            "path"
        }

        assert required.issubset(reader.fieldnames or [])

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            unified = adapt_row(source_row, row_number)

            # Common schema contract
            assert tuple(unified) == UNIFIED_COLUMNS

            # Source metadata integrity
            metadata = json.loads(
                unified["source_metadata_json"]
            )

            assert metadata == {
                key: value
                for key, value in source_row.items()
                if key != "path"
            }

            assert unified["raw_pid"] == source_row["pid"]
            assert unified["video_id"] == source_row["video_id"]
            assert unified["frame_id"] == source_row["frame_id"]
            assert unified["clothes_id"] == source_row["clothes_id"]
            assert unified["source_image_index"] == source_row["image_id"]

            # Never invent camera or tracklet IDs
            assert unified["raw_camera_or_view"] == ""
            assert unified["camera_context_id"] == ""
            assert unified["tracklet_id"] == ""

            # Unique image identifiers
            uid = unified["sample_uid"]
            path = unified["relative_path"]

            assert uid not in sample_uids
            assert path not in relative_paths

            sample_uids.add(uid)
            relative_paths.add(path)

            identities.add(unified["identity_key"])

            clothes_by_pid[source_row["pid"]].add(
                source_row["clothes_id"]
            )

            if first_example is None:
                first_example = unified

            count += 1

    # Final validation against audit findings
    multi_clothes = sum(
        len(clothes) >= 2
        for clothes in clothes_by_pid.values()
    )

    assert count == 71248
    assert len(identities) == 5000
    assert len(sample_uids) == count
    assert len(relative_paths) == count
    assert multi_clothes == 3180

    print("LaST ADAPTER VALIDATION")
    print("-" * 50)
    print("Source rows:", count)
    print("Valid unified rows:", count)
    print("Unique identities:", len(identities))
    print("Multi-clothes identities:", multi_clothes)
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")
    for field in (
        "relative_path", "identity_key",
        "video_id", "frame_id", "clothes_id",
        "camera_semantics"
    ):
        print(f"{field}: {first_example[field]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nLaST ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
