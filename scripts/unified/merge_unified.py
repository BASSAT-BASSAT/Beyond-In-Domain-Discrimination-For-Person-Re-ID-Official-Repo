
"""
Master Unified Candidate Manifest — v1.

Merges the 9 validated source manifests.

- Preserves all 19 columns.
- Checks sample and identity integrity.
- Keeps unassigned source splits unchanged.
- Never overwrites existing outputs.
- Does not modify input files.
"""

import csv
import hashlib
import json
import os
import tempfile

from pathlib import Path

from scripts.unified.schema import (
    UNIFIED_COLUMNS,
    validate_header,
)
from scripts.unified.build_unified import (
    DATASETS,
    OUTPUT_DIR,
)


# Fixed source order for reproducible output.
SOURCES = {
    "cuhk03": "CUHK03_NP",
    "iust": "IUST_PersonReID",
    "last": "LaST",
    "deepchange": "DeepChange",
    "lpw": "LPW",
    "occluded_posetrack": "Occluded-PoseTrack-ReID",
    "orbench": "ORBench_RGB",
    "prid2011": "PRID2011",
    "rpifield": "RPIfield",
}

MASTER_FILE = OUTPUT_DIR / "all_sources_unified_v1.csv"

EXPECTED_TOTAL_IMAGES = 940809
EXPECTED_TOTAL_IDENTITIES = 11297


def sha256_file(path):
    """Calculate a reproducible file checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def merge_all_sources():
    """Merge and validate the nine unified manifests."""

    if set(SOURCES) != set(DATASETS):
        raise ValueError(
            "The registry must contain exactly 9 sources."
        )

    if MASTER_FILE.exists():
        raise FileExistsError(
            f"Master file already exists: {MASTER_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_uids = set()
    identities = set()
    total_rows = 0
    temporary_path = None

    try:
        # First write to a temporary file.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=OUTPUT_DIR,
            prefix=".master_build_",
            suffix=".tmp",
            delete=False,
        ) as temporary:

            temporary_path = Path(temporary.name)

            writer = csv.DictWriter(
                temporary,
                fieldnames=UNIFIED_COLUMNS,
            )
            writer.writeheader()

            # Read each source independently.
            for name, dataset_name in SOURCES.items():
                config = DATASETS[name]
                adapter = config["adapter"]

                source_file = (
                    OUTPUT_DIR / config["output"]
                )

                source_count = 0
                source_identities = set()

                expected_split = (
                    "unassigned"
                    if name in {"prid2011", "rpifield"}
                    else "train"
                )

                with source_file.open(
                    newline="", encoding="utf-8-sig"
                ) as file:

                    reader = csv.DictReader(file)
                    validate_header(reader.fieldnames)

                    for row in reader:
                        # Validate dataset provenance.
                        assert row["dataset"] == dataset_name
                        assert (
                            row["dataset_root_id"]
                            == adapter.DATASET_ROOT.name
                        )
                        assert (
                            row["candidate_pool"]
                            == adapter.SOURCE_FILE.stem
                        )
                        assert row["source_split"] == expected_split

                        # Validate identity namespace.
                        namespace = adapter.IDENTITY_NAMESPACE
                        pid = row["raw_pid"]

                        assert pid
                        assert row["identity_namespace"] == namespace
                        assert row["identity_key"] == (
                            f"{namespace}::{pid}"
                        )

                        # Validate portable path.
                        relative = row["relative_path"]

                        assert relative
                        assert not Path(relative).is_absolute()
                        assert ".." not in Path(relative).parts
                        assert "\\" not in relative

                        # Recalculate sample UID.
                        expected_uid = hashlib.sha256(
                            (
                                dataset_name
                                + "\0"
                                + relative
                            ).encode("utf-8")
                        ).hexdigest()

                        uid = row["sample_uid"]

                        assert uid == expected_uid

                        if uid in sample_uids:
                            raise ValueError(
                                f"Duplicate sample UID: {uid}"
                            )

                        sample_uids.add(uid)

                        # Ensure metadata JSON is valid.
                        metadata = json.loads(
                            row["source_metadata_json"]
                        )
                        assert isinstance(metadata, dict)

                        identity = row["identity_key"]

                        source_identities.add(identity)
                        identities.add(identity)

                        writer.writerow(row)

                        source_count += 1
                        total_rows += 1

                # Source-level integrity checks.
                assert source_count == config["images"]
                assert (
                    len(source_identities)
                    == config["identities"]
                )

                print(
                    f"[PASS] {dataset_name}: "
                    f"{source_count:,} images, "
                    f"{len(source_identities):,} IDs",
                    flush=True,
                )

        # Global integrity checks.
        assert total_rows == EXPECTED_TOTAL_IMAGES
        assert len(sample_uids) == total_rows
        assert len(identities) == EXPECTED_TOTAL_IDENTITIES

        # Verify the written temporary CSV.
        with temporary_path.open(
            newline="", encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)
            validate_header(reader.fieldnames)

            saved_count = sum(1 for _ in reader)

        assert saved_count == total_rows

        checksum = sha256_file(temporary_path)

        # Publish without overwriting an existing file.
        os.link(temporary_path, MASTER_FILE)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print("\nMASTER UNIFIED MANIFEST v1")
    print("-" * 55)
    print("Datasets:", len(SOURCES))
    print("Total images:", total_rows)
    print("Unique sample UIDs:", len(sample_uids))
    print("Source-scoped identities:", len(identities))
    print("Schema columns:", len(UNIFIED_COLUMNS))
    print("PRID2011/RPIfield split: unassigned")
    print("Output:", MASTER_FILE)
    print("SHA-256:", checksum)
    print("Input files modified: NO")
    print("\nMASTER UNIFIED MERGE: PASS")


if __name__ == "__main__":
    merge_all_sources()
