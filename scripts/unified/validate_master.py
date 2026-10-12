
"""
Independent Master Manifest Validator — v1.

Verifies the published Master Candidate Manifest
against all nine individual unified manifests.

Read-only. No files are created or modified.
"""

import csv
import json
import hashlib

from pathlib import PurePosixPath

from scripts.unified.schema import (
    UNIFIED_COLUMNS,
    validate_header,
)

from scripts.unified.build_unified import DATASETS
from scripts.unified.merge_unified import (
    SOURCES,
    MASTER_FILE,
    OUTPUT_DIR,
    EXPECTED_TOTAL_IMAGES,
    EXPECTED_TOTAL_IDENTITIES,
)


EXPECTED_SHA256 = (
    "4fba9e5578b19679275f061dc97740518d300830a0306b725c39e0dfc15e34ce"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_master():

    require(
        set(SOURCES) == set(DATASETS),
        "Source registry mismatch",
    )

    # Verify the exact published master version.
    checksum = file_sha256(MASTER_FILE)

    require(
        checksum == EXPECTED_SHA256,
        "Master SHA-256 mismatch",
    )

    total = 0
    all_uids = set()
    all_paths = set()
    all_identities = set()

    # Read master sequentially, without loading
    # all 940K rows into memory.
    with MASTER_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as master_file:

        master_reader = csv.DictReader(master_file)
        validate_header(master_reader.fieldnames)

        for name, dataset_name in SOURCES.items():

            config = DATASETS[name]
            adapter = config["adapter"]

            source_file = (
                OUTPUT_DIR / config["output"]
            )

            count = 0
            identities = set()

            expected_split = (
                "unassigned"
                if name in {"prid2011", "rpifield"}
                else "train"
            )

            with source_file.open(
                newline="", encoding="utf-8-sig"
            ) as file:

                source_reader = csv.DictReader(file)
                validate_header(source_reader.fieldnames)

                for line_number, source_row in enumerate(
                    source_reader, start=2
                ):

                    master_row = next(master_reader, None)

                    require(
                        master_row is not None,
                        f"Master ended early: {dataset_name}",
                    )

                    # Exact comparison of all 19 columns.
                    require(
                        source_row == master_row,
                        f"Row mismatch: {dataset_name}, "
                        f"source line {line_number}",
                    )

                    row = master_row

                    require(
                        row["dataset"] == dataset_name,
                        "Dataset provenance mismatch",
                    )

                    require(
                        row["candidate_pool"]
                        == adapter.SOURCE_FILE.stem,
                        "Candidate pool mismatch",
                    )

                    require(
                        row["dataset_root_id"]
                        == adapter.DATASET_ROOT.name,
                        "Dataset root mismatch",
                    )

                    require(
                        row["source_split"] == expected_split,
                        "Unexpected source split",
                    )

                    require(
                        row["source_row_number"]
                        == str(line_number),
                        "Source row lineage mismatch",
                    )

                    pid = row["raw_pid"]
                    namespace = adapter.IDENTITY_NAMESPACE

                    require(
                        bool(pid)
                        and row["identity_namespace"] == namespace
                        and row["identity_key"]
                        == f"{namespace}::{pid}",
                        "Identity mapping mismatch",
                    )

                    relative = row["relative_path"]

                    parsed = PurePosixPath(relative)

                    require(
                        bool(relative)
                        and not parsed.is_absolute()
                        and ".." not in parsed.parts
                        and "\\" not in relative
                        and parsed.as_posix() == relative,
                        "Invalid portable path",
                    )

                    # Verify deterministic image UID.
                    expected_uid = hashlib.sha256(
                        (
                            dataset_name + "\0" + relative
                        ).encode("utf-8")
                    ).hexdigest()

                    uid = row["sample_uid"]

                    require(
                        uid == expected_uid,
                        "Sample UID mismatch",
                    )

                    require(
                        uid not in all_uids,
                        "Duplicate sample UID",
                    )

                    all_uids.add(uid)

                    path_key = (
                        row["dataset_root_id"], relative
                    )

                    require(
                        path_key not in all_paths,
                        "Duplicate dataset-relative path",
                    )

                    all_paths.add(path_key)

                    metadata = json.loads(
                        row["source_metadata_json"]
                    )

                    require(
                        isinstance(metadata, dict),
                        "Invalid source metadata",
                    )

                    identities.add(row["identity_key"])
                    all_identities.add(row["identity_key"])

                    count += 1
                    total += 1

            require(
                count == config["images"],
                f"{dataset_name}: row count mismatch",
            )

            require(
                len(identities) == config["identities"],
                f"{dataset_name}: identity count mismatch",
            )

            print(
                f"[PASS] {dataset_name}: {count:,} rows",
                flush=True,
            )

        # Ensure there are no extra master rows.
        require(
            next(master_reader, None) is None,
            "Master contains unexpected extra rows",
        )

    require(
        total == EXPECTED_TOTAL_IMAGES,
        "Global image count mismatch",
    )

    require(
        len(all_uids) == total,
        "Global UID uniqueness failure",
    )

    require(
        len(all_paths) == total,
        "Global path uniqueness failure",
    )

    require(
        len(all_identities) == EXPECTED_TOTAL_IDENTITIES,
        "Global identity count mismatch",
    )

    print("\nINDEPENDENT MASTER VALIDATION")
    print("-" * 50)
    print("Datasets verified:", len(SOURCES))
    print("Rows compared:", total)
    print("Unique sample UIDs:", len(all_uids))
    print("Unique relative paths:", len(all_paths))
    print("Source-scoped identities:", len(all_identities))
    print("Schema columns:", len(UNIFIED_COLUMNS))
    print("Master SHA-256:", checksum)
    print("All row comparisons: PASS")
    print("Files modified: NO")
    print("\nMASTER VALIDATION: PASS")


if __name__ == "__main__":
    validate_master()
