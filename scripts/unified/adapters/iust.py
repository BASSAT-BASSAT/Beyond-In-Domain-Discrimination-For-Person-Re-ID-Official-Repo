
"""
IUST Person ReID Adapter — Unified Schema v1

Converts IUST training candidate rows to unified rows.

Execution mode: read-only validation.
No CSV files are created or modified.
"""

import csv
import json
import hashlib

from pathlib import Path

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "IUST_PersonReID"
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "IUST_PersonReID"
    / "iust_clean_train_candidates_v0.csv"
)

IDENTITY_NAMESPACE = "IUST_PersonReID::train"


def adapt_row(source_row, row_number):
    """Convert one IUST row into Unified Schema v1."""

    # 1. Validate source metadata
    assert source_row["split"] == "train"
    assert source_row["pid"]
    assert source_row["camera"]
    assert source_row["frame"].isdigit()

    # 2. Resolve portable image path
    dataset_root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        source_row["path"]
    ).resolve(strict=True)

    assert image_path.is_file()

    relative_path = image_path.relative_to(
        dataset_root
    ).as_posix()

    assert Path(relative_path).name == source_row["filename"]

    # 3. Generate deterministic image UID
    uid_input = "IUST_PersonReID\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 4. Create source-scoped identity
    pid = source_row["pid"]
    camera = source_row["camera"]

    # 5. Preserve original metadata
    source_metadata = {
        key: value
        for key, value in source_row.items()
        if key != "path"
    }

    # 6. Build unified row
    unified = {
        column: ""
        for column in UNIFIED_COLUMNS
    }

    unified.update({
        "sample_uid": sample_uid,
        "dataset": "IUST_PersonReID",
        "candidate_pool": "iust_clean_train_candidates_v0",
        "dataset_root_id": "IUST_PersonReID",
        "relative_path": relative_path,
        "source_split": source_row["split"],
        "raw_pid": pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": f"{IDENTITY_NAMESPACE}::{pid}",
        "raw_camera_or_view": camera,
        "camera_semantics": "physical_camera_code",
        "camera_context_id": (
            f"IUST_PersonReID::camera::{camera}"
        ),
        "frame_id": source_row["frame"],
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
    """Validate all IUST candidates without writing files."""

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
            "filename", "split", "pid",
            "camera", "sequence", "frame",
            "suffix", "path"
        }

        assert required.issubset(
            reader.fieldnames or []
        )

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            unified = adapt_row(
                source_row, row_number
            )

            # Validate schema consistency
            assert tuple(unified) == UNIFIED_COLUMNS

            # Validate preserved metadata
            original = json.loads(
                unified["source_metadata_json"]
            )

            assert original["sequence"] == source_row["sequence"]
            assert unified["raw_pid"] == source_row["pid"]
            assert unified["frame_id"] == source_row["frame"]

            # Ensure unique samples
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

    # Final checks
    assert count == 72393
    assert len(identities) == 1193
    assert len(cameras) == 17
    assert len(sample_uids) == count

    print("IUST ADAPTER VALIDATION")
    print("-" * 50)

    print("Source rows:", count)
    print("Valid unified rows:", count)
    print("Unique identities:", len(identities))
    print("Camera codes:", len(cameras))
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")
    for key in [
        "relative_path", "identity_key",
        "raw_camera_or_view", "frame_id"
    ]:
        print(f"{key}: {first_example[key]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nIUST ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
