
"""
Unified Manifest Builder — Schema v1

Builds unified CSV manifests using dataset adapters.

- Does not modify original manifests or images.
- Does not overwrite existing unified files.
- Validates image counts, identities, and UIDs.
"""

import argparse
import csv
import hashlib
import os
import tempfile

from pathlib import Path

from scripts.unified.schema import (
    UNIFIED_COLUMNS,
    validate_header,
)

from scripts.unified.adapters import (
    cuhk03,
    iust,
    last,
    deepchange,
    lpw,
    occluded_posetrack,
    orbench,
    prid2011,
    rpifield,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "unified" / "v1"


# Registered datasets and their expected results.
DATASETS = {
    "cuhk03": {
        "adapter": cuhk03,
        "output": "cuhk03_np_unified_v1.csv",
        "images": 7365,
        "identities": 767,
    },

    "iust": {
        "adapter": iust,
        "output": "iust_unified_v1.csv",
        "images": 72393,
        "identities": 1193,
    },

    "last": {
        "adapter": last,
        "output": "last_unified_v1.csv",
        "images": 71248,
        "identities": 5000,
    },

    "deepchange": {
        "adapter": deepchange,
        "output": "deepchange_unified_v1.csv",
        "images": 75009,
        "identities": 450,
        },

    "lpw": {
        "adapter": lpw,
        "output": "lpw_unified_v1.csv",
        "images": 418447,
        "identities": 1975,
    },

    "occluded_posetrack": {
        "adapter": occluded_posetrack,
        "output": "occluded_posetrack_unified_v1.csv",
        "images": 17898,
        "identities": 1000,
    },

    "orbench": {
        "adapter": orbench,
        "output": "orbench_rgb_unified_v1.csv",
        "images": 26932,
        "identities": 600,
    },

    "prid2011": {
        "adapter": prid2011,
        "output": "prid2011_unified_v1.csv",
        "images": 40033,
        "identities": 200,
    },

    "rpifield": {
        "adapter": rpifield,
        "output": "rpifield_unified_v1.csv",
        "images": 211484,
        "identities": 112,
    },
}

def calculate_sha256(path):
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_dataset(dataset_name):
    """Build and validate one unified dataset."""

    config = DATASETS[dataset_name]

    adapter = config["adapter"]
    output = OUTPUT_DIR / config["output"]

    # Never overwrite existing results.
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    identities = set()
    sample_uids = set()
    relative_paths = set()

    temporary_path = None

    try:
        # Build into a temporary file first.
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=OUTPUT_DIR,
            prefix=".unified_build_",
            suffix=".tmp",
            delete=False,
        ) as temp:

            temporary_path = Path(temp.name)

            writer = csv.DictWriter(
                temp,
                fieldnames=UNIFIED_COLUMNS,
            )
            writer.writeheader()

            with adapter.SOURCE_FILE.open(
                newline="",
                encoding="utf-8-sig",
            ) as source_file:

                reader = csv.DictReader(source_file)

                for row_number, source_row in enumerate(
                    reader, start=2
                ):
                    # Reuse the existing adapter.
                    unified = adapter.adapt_row(
                        source_row, row_number
                    )

                    assert tuple(unified) == UNIFIED_COLUMNS

                    uid = unified["sample_uid"]
                    path = unified["relative_path"]
                    identity = unified["identity_key"]

                    assert uid not in sample_uids
                    assert path not in relative_paths

                    sample_uids.add(uid)
                    relative_paths.add(path)
                    identities.add(identity)

                    writer.writerow(unified)
                    count += 1

        # Validate before publishing the CSV.
        assert count == config["images"]
        assert len(identities) == config["identities"]
        assert len(sample_uids) == count
        assert len(relative_paths) == count

        with temporary_path.open(
            newline="", encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)
            validate_header(reader.fieldnames)
            assert sum(1 for _ in reader) == count

        checksum = calculate_sha256(temporary_path)

        # Publish only after all validations pass.
        # os.link fails if the output already exists.
        os.link(temporary_path, output)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print("\nUNIFIED DATASET BUILD")
    print("-" * 50)
    print("Dataset:", dataset_name)
    print("Images:", count)
    print("Unique identities:", len(identities))
    print("Unique sample UIDs:", len(sample_uids))
    print("Schema columns:", len(UNIFIED_COLUMNS))
    print("Output:", output)
    print("SHA-256:", checksum)
    print("Original files modified: NO")
    print("\nUNIFIED BUILD: PASS")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    args = parser.parse_args()

    build_dataset(args.dataset)
