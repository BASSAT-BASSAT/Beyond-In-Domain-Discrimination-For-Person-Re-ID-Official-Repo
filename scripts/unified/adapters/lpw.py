
"""
LPW Unified Adapter — Schema v1.

Converts cleaned LPW training candidates to
unified rows while preserving scene, view,
identity and tracklet semantics.

Running directly performs read-only validation.
"""

import csv
import json
import hashlib

from pathlib import Path

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "LPW"

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "LPW"
    / "lpw_clean_train_candidates_v0.csv"
)

IDENTITY_NAMESPACE = "LPW::train"

# Conflicting tracklets excluded during Local Cleaning v0.
HARD_EXCLUDED_TRACKLETS = {
    "scen2-view2-819",
    "scen2-view2-826",
}


def adapt_row(source_row, row_number):
    """Convert one cleaned LPW row to Unified Schema v1."""

    # 1. Validate official training role
    assert source_row["dataset"] == "LPW"
    assert source_row["official_split"] == "train"
    assert source_row["official_role"] == "train"

    scene = source_row["scene"]
    view = source_row["view"]
    local_pid = source_row["person_local_id"]

    assert scene in {"scen2", "scen3"}
    assert view in {"view1", "view2", "view3", "view4"}
    assert local_pid.isdigit()

    # 2. Validate identity mapping
    expected_pid = f"{scene}-{local_pid}"

    assert source_row["global_person_id"] == expected_pid

    # 3. Validate tracklet mapping
    expected_tracklet = f"{scene}-{view}-{local_pid}"

    assert source_row["tracklet_id"] == expected_tracklet

    assert expected_tracklet not in HARD_EXCLUDED_TRACKLETS

    # 4. Preserve image-number semantics
    image_number = source_row["image_number"]

    assert image_number.isdigit()
    assert source_row["filename"] == f"{image_number}.jpg"

    # 5. Resolve portable image path
    root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        source_row["path"]
    ).resolve(strict=True)

    assert image_path.is_file()

    relative_path = image_path.relative_to(root).as_posix()

    assert Path(relative_path).name == source_row["filename"]

    # 6. Generate deterministic sample UID
    uid_input = "LPW\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 7. Preserve source-specific metadata
    metadata = {
        key: value
        for key, value in source_row.items()
        if key != "path"
    }

    # 8. Build unified row
    unified = {
        column: ""
        for column in UNIFIED_COLUMNS
    }

    unified.update({
        "sample_uid": sample_uid,
        "dataset": "LPW",
        "candidate_pool": "lpw_clean_train_candidates_v0",
        "dataset_root_id": "LPW",
        "relative_path": relative_path,
        "source_split": "train",
        "raw_pid": expected_pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": (
            f"{IDENTITY_NAMESPACE}::{expected_pid}"
        ),
        "raw_camera_or_view": view,
        "camera_semantics": "scene_local_view",
        "camera_context_id": f"LPW::{scene}::{view}",
        "tracklet_id": expected_tracklet,
        "source_image_index": image_number,
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
    """Validate all LPW candidates without writing files."""

    count = 0
    identities = set()
    tracklets = set()
    scenes = set()
    sample_uids = set()
    relative_paths = set()

    first_example = None

    with SOURCE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "dataset", "scene", "view",
            "person_local_id", "global_person_id",
            "tracklet_id", "image_number",
            "official_split", "official_role",
            "filename", "path"
        }

        assert required.issubset(reader.fieldnames or [])

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            unified = adapt_row(source_row, row_number)

            # Common schema integrity
            assert tuple(unified) == UNIFIED_COLUMNS

            # Original metadata integrity
            metadata = json.loads(
                unified["source_metadata_json"]
            )

            assert metadata == {
                key: value
                for key, value in source_row.items()
                if key != "path"
            }

            assert unified["raw_pid"] == (
                source_row["global_person_id"]
            )

            assert unified["tracklet_id"] == (
                source_row["tracklet_id"]
            )

            assert unified["source_image_index"] == (
                source_row["image_number"]
            )

            # Do not invent unsupported metadata
            assert unified["frame_id"] == ""
            assert unified["video_id"] == ""
            assert unified["clothes_id"] == ""

            # Unique image verification
            uid = unified["sample_uid"]
            path = unified["relative_path"]

            assert uid not in sample_uids
            assert path not in relative_paths

            sample_uids.add(uid)
            relative_paths.add(path)

            identities.add(unified["identity_key"])
            tracklets.add(unified["tracklet_id"])
            scenes.add(source_row["scene"])

            if first_example is None:
                first_example = unified

            count += 1

    # Final audit-based checks
    assert count == 418447
    assert len(identities) == 1975
    assert len(tracklets) == 5864
    assert scenes == {"scen2", "scen3"}

    assert tracklets.isdisjoint(HARD_EXCLUDED_TRACKLETS)

    assert len(sample_uids) == count
    assert len(relative_paths) == count

    print("LPW ADAPTER VALIDATION")
    print("-" * 50)
    print("Clean candidate images:", count)
    print("Unique identities:", len(identities))
    print("Retained tracklets:", len(tracklets))
    print("Scenes:", sorted(scenes))
    print("Excluded conflicting tracklets:", 2)
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")

    for field in (
        "relative_path",
        "identity_key",
        "raw_camera_or_view",
        "camera_context_id",
        "tracklet_id",
        "source_image_index",
        "frame_id"
    ):
        print(f"{field}: {first_example[field]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nLPW ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
