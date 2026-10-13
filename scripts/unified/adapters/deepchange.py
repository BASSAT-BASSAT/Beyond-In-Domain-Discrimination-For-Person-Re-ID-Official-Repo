
"""
DeepChange Adapter — Unified Schema v1

Resolves paths using validated audit evidence.

When executed directly:
- Converts and validates rows in memory.
- Does not create or modify any files.
"""

import csv
import json
import hashlib

from functools import lru_cache
from pathlib import Path

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "DeepChange"

SOURCE_FILE = (
    PROJECT_ROOT / "manifests" / "DeepChange"
    / "deepchange_clean_train_candidates_v0.csv"
)

HASH_FILE = (
    PROJECT_ROOT / "reports" / "DeepChange"
    / "image_hashes.csv"
)

IDENTITY_NAMESPACE = "DeepChange::train"

KEY_FIELDS = (
    "filename", "pid", "camera_id", "tracklet_id"
)


def make_key(row):
    """Create an exact metadata join key."""
    return tuple(row[field] for field in KEY_FIELDS)


@lru_cache(maxsize=1)
def load_path_lookup():
    """
    Load verified training image paths from
    DeepChange audit evidence only once.
    """

    lookup = {}

    with HASH_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)

        assert {
            "split", "path", *KEY_FIELDS
        }.issubset(reader.fieldnames or [])

        for row in reader:

            if row["split"] != "train":
                continue

            key = make_key(row)

            assert key not in lookup, (
                f"Duplicate audit key: {key}"
            )

            lookup[key] = row["path"]

    assert len(lookup) == 75083

    return lookup


def adapt_row(source_row, row_number):
    """Convert one DeepChange row to Unified v1."""

    # 1. Validate original metadata
    assert source_row["split"] == "train"
    assert source_row["pid"].isdigit()
    assert source_row["camera_id"].isdigit()
    assert source_row["tracklet_id"].isdigit()
    assert source_row["frame_id"].isdigit()

    # 2. Resolve image path from audit evidence
    lookup = load_path_lookup()
    key = make_key(source_row)

    assert key in lookup, (
        f"Missing audit path for {key}"
    )

    dataset_root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        lookup[key]
    ).resolve(strict=True)

    assert image_path.is_file()

    relative_path = image_path.relative_to(
        dataset_root
    ).as_posix()

    assert Path(relative_path).name == source_row["filename"]

    # 3. Create stable sample UID
    uid_input = "DeepChange\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 4. Preserve original identity
    pid = source_row["pid"]
    camera = source_row["camera_id"]

    # 5. Preserve all source-specific metadata
    source_metadata = dict(source_row)

    # 6. Build unified row
    unified = {
        column: ""
        for column in UNIFIED_COLUMNS
    }

    unified.update({
        "sample_uid": sample_uid,
        "dataset": "DeepChange",
        "candidate_pool": "deepchange_clean_train_candidates_v0",
        "dataset_root_id": "DeepChange",
        "relative_path": relative_path,
        "source_split": "train",
        "raw_pid": pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": f"{IDENTITY_NAMESPACE}::{pid}",
        "raw_camera_or_view": camera,
        "camera_semantics": "physical_camera_code",
        "camera_context_id": f"DeepChange::camera::{camera}",
        "tracklet_id": source_row["tracklet_id"],
        "frame_id": source_row["frame_id"],
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
    """Validate all candidates without writing CSVs."""

    count = 0
    identities = set()
    cameras = set()
    sample_uids = set()
    relative_paths = set()
    matched_keys = set()
    first_example = None

    lookup = load_path_lookup()

    with SOURCE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "filename", "split", "pid",
            "camera_id", "date", "time",
            "timestamp", "frame_id", "tracklet_id"
        }

        assert required.issubset(reader.fieldnames or [])

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            key = make_key(source_row)

            assert key not in matched_keys
            matched_keys.add(key)

            unified = adapt_row(
                source_row, row_number
            )

            # Check common schema
            assert tuple(unified) == UNIFIED_COLUMNS

            # Verify original metadata preservation
            metadata = json.loads(
                unified["source_metadata_json"]
            )

            assert metadata == source_row

            assert unified["raw_pid"] == source_row["pid"]
            assert unified["tracklet_id"] == source_row["tracklet_id"]
            assert unified["frame_id"] == source_row["frame_id"]

            # Do not invent video/clothes metadata
            assert unified["video_id"] == ""
            assert unified["clothes_id"] == ""
            assert unified["source_image_index"] == ""

            uid = unified["sample_uid"]
            path = unified["relative_path"]

            assert uid not in sample_uids
            assert path not in relative_paths

            sample_uids.add(uid)
            relative_paths.add(path)

            identities.add(unified["identity_key"])
            cameras.add(unified["raw_camera_or_view"])

            if first_example is None:
                first_example = unified

            count += 1

    # Validate cleaning and join integrity
    assert count == 75009
    assert len(identities) == 450
    assert len(cameras) == 16
    assert len(sample_uids) == count
    assert len(relative_paths) == count
    assert len(matched_keys) == count

    assert matched_keys.issubset(lookup.keys())
    assert len(set(lookup) - matched_keys) == 74

    print("DEEPCHANGE ADAPTER VALIDATION")
    print("-" * 50)

    print("Original train images:", len(lookup))
    print("Clean candidate images:", count)
    print("Local cleaning exclusions:", 74)
    print("Matched paths:", len(matched_keys))
    print("Unique identities:", len(identities))
    print("Camera codes:", len(cameras))
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")
    for field in (
        "relative_path", "identity_key",
        "raw_camera_or_view", "tracklet_id",
        "frame_id", "camera_semantics"
    ):
        print(f"{field}: {first_example[field]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nDEEPCHANGE ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
