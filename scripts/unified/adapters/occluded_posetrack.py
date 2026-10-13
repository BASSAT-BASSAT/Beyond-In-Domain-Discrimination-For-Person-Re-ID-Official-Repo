
"""
Occluded-PoseTrack-ReID Adapter — Unified Schema v1.

Converts clean training candidates to unified rows.

Important:
- Logical video IDs are NOT physical camera IDs.
- PoseTrack image IDs are NOT verified temporal frame IDs.
- Person IDs are split-scoped.

Direct execution performs read-only validation.
"""

import csv
import json
import hashlib

from pathlib import Path

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "Occluded-PoseTrack-ReID"
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "Occluded-PoseTrack-ReID"
    / "occluded_posetrack_clean_train_candidates_v0.csv"
)

DATASET_NAME = "Occluded-PoseTrack-ReID"

IDENTITY_NAMESPACE = (
    "Occluded-PoseTrack-ReID::train"
)


def adapt_row(source_row, row_number):
    """Convert one source candidate to Unified Schema v1."""

    # 1. Validate training partition
    assert source_row["dataset"] == DATASET_NAME
    assert source_row["source_partition"] == "train"
    assert source_row["split"] == "train"

    pid = source_row["person_id"]
    video = source_row["video_id"]
    logical_camera = source_row["logical_camid"]
    image_id = source_row["posetrack_image_id"]

    assert pid.isdigit()
    assert video.isdigit()
    assert logical_camera == video
    assert image_id.isdigit()
    assert source_row["annotation_id"].isdigit()

    assert int(source_row["crop_width"]) > 0
    assert int(source_row["crop_height"]) > 0

    assert source_row["folder_matches_video"] == "True"

    # 2. Verify filename metadata
    expected_filename = (
        f"{pid}_{video}_{image_id}.jpg"
    )

    assert source_row["filename"] == expected_filename

    # 3. Resolve portable image path
    dataset_root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        source_row["path"]
    ).resolve(strict=True)

    assert image_path.is_file()

    relative_path = image_path.relative_to(
        dataset_root
    ).as_posix()

    assert image_path.name == expected_filename
    assert image_path.parent.name == video

    assert (
        relative_path.startswith(
            "extracted/occluded_posetrack_reid/images/train/"
        )
    )

    # 4. Generate stable sample UID
    uid_input = DATASET_NAME + "\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

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
        "dataset": DATASET_NAME,
        "candidate_pool": (
            "occluded_posetrack_clean_train_candidates_v0"
        ),
        "dataset_root_id": DATASET_NAME,
        "relative_path": relative_path,
        "source_split": "train",
        "raw_pid": pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": f"{IDENTITY_NAMESPACE}::{pid}",
        "raw_camera_or_view": logical_camera,
        "camera_semantics": "logical_video_proxy",
        "camera_context_id": (
            f"{DATASET_NAME}::train::video::{video}"
        ),
        "video_id": video,
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
    """Validate all train candidates without writing files."""

    count = 0
    identities = set()
    videos = set()
    sample_uids = set()
    relative_paths = set()

    first_example = None

    with SOURCE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "dataset", "source_partition", "split",
            "filename", "path", "person_id",
            "video_id", "logical_camid",
            "posetrack_image_id", "annotation_id",
            "crop_width", "crop_height",
            "negative_kps_available",
            "folder_matches_video"
        }

        assert required.issubset(
            reader.fieldnames or []
        )

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            unified = adapt_row(source_row, row_number)

            # 1. Common schema validation
            assert tuple(unified) == UNIFIED_COLUMNS

            # 2. Source metadata preservation
            metadata = json.loads(
                unified["source_metadata_json"]
            )

            assert metadata == {
                key: value
                for key, value in source_row.items()
                if key != "path"
            }

            # 3. Identity and video integrity
            assert unified["raw_pid"] == (
                source_row["person_id"]
            )

            assert unified["video_id"] == (
                source_row["video_id"]
            )

            assert unified["raw_camera_or_view"] == (
                source_row["logical_camid"]
            )

            assert (
                unified["camera_semantics"]
                == "logical_video_proxy"
            )

            # 4. Do not invent unsupported metadata
            assert unified["tracklet_id"] == ""
            assert unified["frame_id"] == ""
            assert unified["clothes_id"] == ""
            assert unified["source_image_index"] == ""

            # 5. Verify sample uniqueness
            uid = unified["sample_uid"]
            relative = unified["relative_path"]

            assert uid not in sample_uids
            assert relative not in relative_paths

            sample_uids.add(uid)
            relative_paths.add(relative)

            identities.add(unified["identity_key"])
            videos.add(unified["video_id"])

            if first_example is None:
                first_example = unified

            count += 1

    # 6. Audit-based validations
    assert count == 17898
    assert len(identities) == 1000
    assert len(videos) == 474

    assert len(sample_uids) == count
    assert len(relative_paths) == count

    print("OCCLUDED-POSETRACK ADAPTER VALIDATION")
    print("-" * 50)

    print("Clean candidate images:", count)
    print("Unique identities:", len(identities))
    print("Logical videos:", len(videos))
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")

    for field in (
        "relative_path",
        "identity_key",
        "video_id",
        "raw_camera_or_view",
        "camera_semantics",
        "tracklet_id",
        "frame_id"
    ):
        print(f"{field}: {first_example[field]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nOCCLUDED-POSETRACK ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
