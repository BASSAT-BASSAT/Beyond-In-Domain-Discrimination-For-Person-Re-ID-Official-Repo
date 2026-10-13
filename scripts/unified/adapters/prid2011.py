
"""
PRID2011 Unified Adapter — Schema v1.

Converts matched multi-shot candidate images
into unified rows.

No official supervised train/validation split
is assigned at this stage.

Direct execution is read-only.
"""

import csv
import json
import hashlib

from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "PRID2011"

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "PRID2011"
    / "prid2011_matched_multishot_candidates_v0.csv"
)

DATASET_NAME = "PRID2011"

IDENTITY_NAMESPACE = "PRID2011::matched_multishot"


def adapt_row(source_row, row_number):
    """Convert one PRID2011 candidate into Unified v1."""

    # 1. Validate candidate selection
    assert source_row["dataset"] == DATASET_NAME
    assert source_row["version"] == "multi_shot"
    assert source_row["cross_camera_matched"] == "True"

    camera = source_row["camera"]
    assert camera in {"cam_a", "cam_b"}

    person_number = int(source_row["person_number"])
    assert 1 <= person_number <= 200

    local_pid = f"person_{person_number:04d}"
    global_pid = f"shared-{person_number:04d}"

    assert source_row["person_local_id"] == local_pid
    assert source_row["global_person_id"] == global_pid

    # 2. Validate original trajectory
    expected_trajectory = f"{camera}-{local_pid}"

    assert source_row["trajectory_id"] == expected_trajectory

    # 3. Validate sequence index
    # Original CSV stores values such as "1.0".
    sequence = Decimal(source_row["sequence_index"])

    assert sequence == int(sequence)
    assert sequence >= 1

    sequence_index = str(int(sequence))

    filename = source_row["filename"]

    assert Path(filename).suffix.lower() == ".png"
    assert Path(filename).stem.isdigit()
    assert int(Path(filename).stem) == int(sequence)

    # 4. Resolve portable image path
    root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        source_row["path"]
    ).resolve(strict=True)

    assert image_path.is_file()

    relative = image_path.relative_to(root)
    relative_path = relative.as_posix()

    assert relative.parts[:3] == (
        "extracted", "prid_2011", "multi_shot"
    )

    assert relative.parts[3] == camera
    assert relative.parts[4] == local_pid
    assert relative.name == filename

    # 5. Generate deterministic image UID
    uid_input = DATASET_NAME + "\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 6. Preserve original metadata
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
            "prid2011_matched_multishot_candidates_v0"
        ),
        "dataset_root_id": DATASET_NAME,
        "relative_path": relative_path,
        "source_split": "unassigned",
        "raw_pid": global_pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": (
            f"{IDENTITY_NAMESPACE}::{global_pid}"
        ),
        "raw_camera_or_view": camera,
        "camera_semantics": "physical_camera_code",
        "camera_context_id": (
            f"PRID2011::camera::{camera}"
        ),
        "tracklet_id": expected_trajectory,
        "source_image_index": sequence_index,
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
    """Validate candidates without creating any CSV."""

    count = 0
    identities = set()
    trajectories = set()
    sample_uids = set()
    relative_paths = set()

    camera_counts = Counter()
    cameras_per_identity = defaultdict(set)
    trajectory_indices = defaultdict(set)

    first_example = None

    with SOURCE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "dataset", "version", "camera",
            "person_local_id", "person_number",
            "global_person_id", "cross_camera_matched",
            "trajectory_id", "sequence_index",
            "filename", "path"
        }

        assert required.issubset(reader.fieldnames or [])

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            unified = adapt_row(
                source_row, row_number
            )

            # Common schema
            assert tuple(unified) == UNIFIED_COLUMNS

            # Preserve original metadata
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
                source_row["trajectory_id"]
            )

            assert unified["source_split"] == "unassigned"

            # No unverified temporal frame numbers
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

            # Verify trajectory sequence uniqueness
            trajectory = unified["tracklet_id"]
            sequence_index = unified["source_image_index"]

            assert sequence_index not in trajectory_indices[trajectory]
            trajectory_indices[trajectory].add(sequence_index)

            pid = unified["identity_key"]
            camera = unified["raw_camera_or_view"]

            identities.add(pid)
            trajectories.add(trajectory)
            camera_counts[camera] += 1
            cameras_per_identity[pid].add(camera)

            if first_example is None:
                first_example = unified

            count += 1

    # Audit-based final checks
    assert count == 40033
    assert len(identities) == 200
    assert len(trajectories) == 400

    assert camera_counts == {
        "cam_a": 23330,
        "cam_b": 16703
    }

    assert all(
        cameras == {"cam_a", "cam_b"}
        for cameras in cameras_per_identity.values()
    )

    assert len(sample_uids) == count
    assert len(relative_paths) == count

    print("PRID2011 ADAPTER VALIDATION")
    print("-" * 50)
    print("Matched candidate images:", count)
    print("Matched identities:", len(identities))
    print("Cam A images:", camera_counts["cam_a"])
    print("Cam B images:", camera_counts["cam_b"])
    print("Retained trajectories:", len(trajectories))
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")

    for field in (
        "relative_path",
        "identity_key",
        "source_split",
        "raw_camera_or_view",
        "camera_semantics",
        "tracklet_id",
        "source_image_index",
        "frame_id"
    ):
        print(f"{field}: {first_example[field]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nPRID2011 ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
