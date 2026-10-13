
"""
ORBench RGB Adapter — Unified Schema v1.

Uses cleaned RGB/VIS training candidates only.

Preserves the separate LLCM and SYSU camera
namespaces and original identity metadata.

Direct execution is read-only.
"""

import csv
import json
import hashlib
import re

from decimal import Decimal
from pathlib import Path
from collections import Counter, defaultdict

from scripts.unified.schema import UNIFIED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "ORBench"

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "ORBench"
    / "orbench_clean_train_candidates_v0.csv"
)

DATASET_NAME = "ORBench_RGB"
IDENTITY_NAMESPACE = "ORBench_RGB::train"


def adapt_row(source_row, row_number):
    """Convert one ORBench RGB training candidate."""

    # 1. Validate clean RGB training selection
    assert source_row["split"] == "train"
    assert source_row["mode"] == "RGB"

    assert (
        source_row["recommended_training_exclusion"]
        == "False"
    )

    origin = source_row["source_dataset"]
    assert origin in {"llcm", "sysu"}

    pid = source_row["orbench_id"]
    source_pid = source_row["source_pid"]
    camera = source_row["camera_id"]

    assert pid.isdigit()
    assert source_pid.isdigit()
    assert camera.isdigit()

    assert int(source_row["width"]) > 0
    assert int(source_row["height"]) > 0

    # 2. Validate relative image path
    original_relative = Path(source_row["file_path"])

    assert not original_relative.is_absolute()
    assert ".." not in original_relative.parts
    assert original_relative.parts[0] == "vis"

    root = DATASET_ROOT.resolve(strict=True)

    image_path = (
        root / "extracted" / "ORBench"
        / original_relative
    ).resolve(strict=True)

    assert image_path.is_file()

    relative_path = image_path.relative_to(root).as_posix()

    # 3. Validate source-specific temporal metadata
    frame_id = ""
    filename = image_path.name

    if origin == "llcm":

        pattern = (
            r"_c(\d+)_s(\d+)_f(\d+)_vis\."
            r"(?:jpg|jpeg|png)$"
        )
        match = re.search(pattern, filename)

        assert match is not None

        sequence = Decimal(source_row["sequence_id"])
        frame = Decimal(source_row["frame_number"])

        assert sequence == int(sequence)
        assert frame == int(frame)

        assert int(match.group(1)) == int(camera)
        assert int(match.group(2)) == int(sequence)
        assert int(match.group(3)) == int(frame)

        frame_id = str(int(frame))

    else:
        # SYSU has no equivalent sequence/frame metadata
        assert source_row["sequence_id"] == ""
        assert source_row["frame_number"] == ""

        match = re.search(
            r"_cam(\d+)_\d+_vis\.(?:jpg|jpeg|png)$",
            filename
        )

        assert match is not None
        assert int(match.group(1)) == int(camera)

    # 4. Generate stable sample identifier
    uid_input = "ORBench_RGB\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 5. Preserve all original source metadata
    metadata = dict(source_row)

    # 6. Build unified row
    unified = {
        column: ""
        for column in UNIFIED_COLUMNS
    }

    unified.update({
        "sample_uid": sample_uid,
        "dataset": DATASET_NAME,
        "candidate_pool": (
            "orbench_clean_train_candidates_v0"
        ),
        "dataset_root_id": "ORBench",
        "relative_path": relative_path,
        "source_split": "train",
        "raw_pid": pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": (
            f"{IDENTITY_NAMESPACE}::{pid}"
        ),
        "raw_camera_or_view": camera,
        "camera_semantics": (
            "source_scoped_camera_code"
        ),
        "camera_context_id": (
            f"ORBench::{origin}::camera::{camera}"
        ),
        "frame_id": frame_id,
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
    """Validate clean candidates without writing CSVs."""

    count = 0
    origins = Counter()

    identities = set()
    sample_uids = set()
    relative_paths = set()
    camera_contexts = set()

    # Validate one-to-one source identity mapping
    source_to_orbench = defaultdict(set)
    orbench_to_source = defaultdict(set)

    llcm_frames = 0
    first_example = None

    with SOURCE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "file_path", "split", "json_label",
            "orbench_id", "source_dataset",
            "source_pid", "camera_id",
            "sequence_id", "frame_number",
            "width", "height", "aspect_ratio",
            "mode", "pixel_sha256", "file_sha256",
            "recommended_training_exclusion"
        }

        assert required.issubset(reader.fieldnames or [])

        for row_number, source_row in enumerate(
            reader, start=2
        ):
            unified = adapt_row(source_row, row_number)

            assert tuple(unified) == UNIFIED_COLUMNS

            metadata = json.loads(
                unified["source_metadata_json"]
            )
            assert metadata == source_row

            origin = source_row["source_dataset"]
            origin_pid = source_row["source_pid"]
            orbench_id = source_row["orbench_id"]

            source_identity = (origin, origin_pid)

            source_to_orbench[source_identity].add(orbench_id)
            orbench_to_source[orbench_id].add(source_identity)

            assert unified["raw_pid"] == orbench_id

            assert unified["camera_context_id"] == (
                f"ORBench::{origin}::camera::"
                f"{source_row['camera_id']}"
            )

            if origin == "llcm":
                assert unified["frame_id"].isdigit()
                llcm_frames += 1
            else:
                assert unified["frame_id"] == ""

            # No invented tracklet or video IDs
            assert unified["tracklet_id"] == ""
            assert unified["video_id"] == ""
            assert unified["source_image_index"] == ""

            uid = unified["sample_uid"]
            relative = unified["relative_path"]

            assert uid not in sample_uids
            assert relative not in relative_paths

            sample_uids.add(uid)
            relative_paths.add(relative)

            identities.add(unified["identity_key"])
            camera_contexts.add(
                unified["camera_context_id"]
            )
            origins[origin] += 1

            if first_example is None:
                first_example = unified

            count += 1

    # Final audit-based checks
    assert count == 26932
    assert origins == {
        "llcm": 9643,
        "sysu": 17289
    }

    assert len(identities) == 600
    assert len(source_to_orbench) == 600
    assert len(orbench_to_source) == 600

    assert all(
        len(mapped) == 1
        for mapped in source_to_orbench.values()
    )
    assert all(
        len(mapped) == 1
        for mapped in orbench_to_source.values()
    )

    assert llcm_frames == 9643
    assert len(sample_uids) == count
    assert len(relative_paths) == count

    print("ORBENCH RGB ADAPTER VALIDATION")
    print("-" * 50)
    print("Clean candidate images:", count)
    print("LLCM images:", origins["llcm"])
    print("SYSU images:", origins["sysu"])
    print("Unique identities:", len(identities))
    print("Source identity mapping:", "ONE-TO-ONE")
    print("Camera contexts:", len(camera_contexts))
    print("LLCM frame IDs preserved:", llcm_frames)
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))

    print("\nFirst converted row:")
    for field in (
        "relative_path",
        "identity_key",
        "raw_camera_or_view",
        "camera_context_id",
        "frame_id",
    ):
        print(f"{field}: {first_example[field]}")

    print("\nCSV files created: NO")
    print("Source files modified: NO")
    print("\nORBENCH RGB ADAPTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_adapter()
