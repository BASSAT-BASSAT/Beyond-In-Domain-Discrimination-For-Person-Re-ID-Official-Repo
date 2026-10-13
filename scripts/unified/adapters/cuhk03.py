
"""
CUHK03-NP Unified Adapter — v1

Convert CUHK03 detected training candidates into
Unified Schema v1.

Current execution mode:
Verify reproducibility against the existing v1 CSV.

No files are created or overwritten.
"""

import csv
import json
import hashlib

from pathlib import Path
from itertools import zip_longest

from scripts.unified.schema import (
    UNIFIED_COLUMNS,
    validate_header,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "CUHK03_NP"
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "CUHK03_NP"
    / "detected_clean_train_candidates_v0.csv"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "unified"
    / "v1"
    / "cuhk03_np_unified_v1.csv"
)

IDENTITY_NAMESPACE = "CUHK03_NP::detected::train"


def adapt_row(source_row, row_number):
    """
    Convert one CUHK03 source row into
    Unified Schema v1.
    """

    # 1. Validate original metadata
    assert source_row["split"] == "train"
    assert source_row["variant"] == "detected"
    assert source_row["pid"]
    assert source_row["view"] in {"1", "2"}

    # 2. Convert absolute path to relative path
    dataset_root = DATASET_ROOT.resolve(strict=True)

    image_path = Path(
        source_row["path"]
    ).resolve(strict=True)

    assert image_path.is_file()

    relative_path = image_path.relative_to(
        dataset_root
    ).as_posix()

    assert Path(relative_path).name == source_row["filename"]

    # 3. Generate deterministic sample ID
    uid_input = "CUHK03_NP\0" + relative_path

    sample_uid = hashlib.sha256(
        uid_input.encode("utf-8")
    ).hexdigest()

    # 4. Create dataset-specific identity key
    pid = source_row["pid"]

    identity_key = (
        f"{IDENTITY_NAMESPACE}::{pid}"
    )

    # 5. Preserve original source metadata
    original_metadata = {
        key: value
        for key, value in source_row.items()
        if key != "path"
    }

    # 6. Create unified row
    unified = {
        column: ""
        for column in UNIFIED_COLUMNS
    }

    unified.update({
        "sample_uid": sample_uid,
        "dataset": "CUHK03_NP",
        "candidate_pool": "detected_clean_train_candidates_v0",
        "dataset_root_id": "CUHK03_NP",
        "relative_path": relative_path,
        "source_split": source_row["split"],
        "raw_pid": pid,
        "identity_namespace": IDENTITY_NAMESPACE,
        "identity_key": identity_key,
        "raw_camera_or_view": source_row["view"],
        "camera_semantics": "pair_local_view",
        "source_row_number": str(row_number),
        "source_image_index": source_row["image_index"],
        "source_metadata_json": json.dumps(
            original_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")
        ),
    })

    return unified


def verify_reproducibility():
    """
    Recompute all unified rows in memory and
    compare them against the existing v1 CSV.
    """

    count = 0
    identities = set()
    sample_uids = set()

    with (
        SOURCE_FILE.open(
            newline="", encoding="utf-8-sig"
        ) as source_file,

        REFERENCE_FILE.open(
            newline="", encoding="utf-8-sig"
        ) as reference_file,
    ):

        source_reader = csv.DictReader(source_file)
        reference_reader = csv.DictReader(reference_file)

        validate_header(reference_reader.fieldnames)

        for row_number, (source, reference) in enumerate(
            zip_longest(source_reader, reference_reader),
            start=2
        ):

            if source is None or reference is None:
                raise ValueError(
                    f"Row count mismatch at line {row_number}"
                )

            generated = adapt_row(source, row_number)

            if generated != reference:
                different_columns = [
                    col for col in UNIFIED_COLUMNS
                    if generated[col] != reference[col]
                ]

                raise ValueError(
                    f"Row {row_number} mismatch: "
                    f"{different_columns}"
                )

            uid = generated["sample_uid"]

            assert uid not in sample_uids

            sample_uids.add(uid)
            identities.add(generated["identity_key"])
            count += 1

    assert count == 7365
    assert len(sample_uids) == 7365
    assert len(identities) == 767

    print("CUHK03 ADAPTER REPRODUCIBILITY")
    print("-" * 50)
    print("Source rows:", count)
    print("Matching unified rows:", count)
    print("Unique sample UIDs:", len(sample_uids))
    print("Unique identities:", len(identities))
    print("Schema columns:", len(UNIFIED_COLUMNS))
    print("Row mismatches:", 0)
    print("Files modified: NO")
    print()
    print("CUHK03 ADAPTER REPRODUCIBILITY: PASS")


if __name__ == "__main__":
    verify_reproducibility()
