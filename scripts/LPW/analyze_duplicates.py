from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter
import hashlib

import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/LPW"
    / "lpw_manifest.csv"
)

REPORT_DIR = (
    BASE
    / "reports/LPW"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HASHED_OUT = (
    REPORT_DIR
    / "manifest_with_sha256.csv"
)

GROUPS_OUT = (
    REPORT_DIR
    / "exact_duplicate_groups.csv"
)

ROWS_OUT = (
    REPORT_DIR
    / "exact_duplicate_rows.csv"
)


# ============================================================
# Hash worker
# ============================================================

def sha256_file(item):

    idx, path = item

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return (
        idx,
        h.hexdigest()
    )


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(
    MANIFEST,
    dtype={
        "person_local_id": str,
        "global_person_id": str,
        "tracklet_id": str,
    },
)


print("=" * 90)
print("LPW EXACT DUPLICATE AUDIT")
print("=" * 90)

print(
    "\nImages:",
    len(df)
)


# ============================================================
# File-size prefilter
#
# Exact duplicates MUST have the same byte size.
# Files with a unique size cannot have an exact duplicate.
# ============================================================

print("\nReading file sizes...")


df["file_size"] = [
    Path(path).stat().st_size
    for path in df["path"]
]


size_counts = (
    df["file_size"]
    .value_counts()
)


candidate_mask = (
    df["file_size"]
    .map(size_counts)
    > 1
)


candidate_indices = (
    df.index[
        candidate_mask
    ]
    .tolist()
)


print(
    "Images in repeated-size groups:",
    len(candidate_indices)
)

print(
    "Images with unique byte size:",
    int(
        (~candidate_mask).sum()
    )
)


# ============================================================
# SHA256
# ============================================================

workers = min(
    8,
    max(
        1,
        cpu_count() - 1
    )
)


print(
    "Workers:",
    workers
)

print(
    "\nComputing SHA256 hashes..."
)


items = [
    (
        idx,
        df.at[
            idx,
            "path"
        ],
    )
    for idx in candidate_indices
]


hash_map = {}


with Pool(
    processes=workers
) as pool:

    for n, (
        idx,
        digest,
    ) in enumerate(
        pool.imap_unordered(
            sha256_file,
            items,
            chunksize=500,
        ),
        start=1,
    ):

        hash_map[
            idx
        ] = digest

        if (
            n % 50000 == 0
            or n == len(items)
        ):

            print(
                f"  processed "
                f"{n:,} / {len(items):,}"
            )


df["sha256"] = None

for idx, digest in hash_map.items():

    df.at[
        idx,
        "sha256"
    ] = digest


# ============================================================
# Duplicate groups
# ============================================================

hashed = df[
    df["sha256"]
    .notna()
].copy()


hash_counts = (
    hashed[
        "sha256"
    ]
    .value_counts()
)


duplicate_hashes = set(
    hash_counts[
        hash_counts > 1
    ].index
)


dup = hashed[
    hashed[
        "sha256"
    ].isin(
        duplicate_hashes
    )
].copy()


# ============================================================
# Assign duplicate group IDs
# ============================================================

hash_to_group = {
    digest:
        group_id

    for group_id, digest
    in enumerate(
        sorted(
            duplicate_hashes
        ),
        start=1,
    )
}


if len(dup) > 0:

    dup[
        "duplicate_group"
    ] = (
        dup[
            "sha256"
        ]
        .map(
            hash_to_group
        )
    )

else:

    dup[
        "duplicate_group"
    ] = pd.Series(
        dtype=int
    )


# ============================================================
# Group characteristics
# ============================================================

group_rows = []


for digest in sorted(
    duplicate_hashes
):

    x = dup[
        dup[
            "sha256"
        ]
        == digest
    ]

    group_id = (
        hash_to_group[
            digest
        ]
    )

    persons = (
        x[
            "global_person_id"
        ]
        .nunique()
    )

    tracklets = (
        x[
            "tracklet_id"
        ]
        .nunique()
    )

    scenes = (
        x[
            "scene"
        ]
        .nunique()
    )

    views = (
        x[
            "view"
        ]
        .nunique()
    )

    roles = set(
        x[
            "official_role"
        ]
        .tolist()
    )


    cross_train_eval = (
        "train" in roles
        and (
            "query" in roles
            or "gallery" in roles
        )
    )


    cross_query_gallery = (
        "query" in roles
        and "gallery" in roles
    )


    group_rows.append(
        {
            "duplicate_group":
                group_id,

            "sha256":
                digest,

            "images":
                len(x),

            "persons":
                persons,

            "tracklets":
                tracklets,

            "scenes":
                scenes,

            "views":
                views,

            "cross_person":
                persons > 1,

            "cross_tracklet":
                tracklets > 1,

            "cross_scene":
                scenes > 1,

            "cross_view":
                views > 1,

            "cross_train_eval":
                cross_train_eval,

            "cross_query_gallery":
                cross_query_gallery,
        }
    )


groups_df = pd.DataFrame(
    group_rows
)


# ============================================================
# Summary
# ============================================================

duplicate_groups = len(
    groups_df
)

images_in_groups = len(
    dup
)

extra_copies = (
    images_in_groups
    - duplicate_groups
)


print("\n" + "=" * 90)
print("EXACT FILE-BYTE DUPLICATES")
print("=" * 90)

print(
    "Duplicate groups       :",
    duplicate_groups
)

print(
    "Images in dup groups   :",
    images_in_groups
)

print(
    "Extra duplicate copies :",
    extra_copies
)


if duplicate_groups > 0:

    print("\nGROUP CHARACTERISTICS")
    print("-" * 90)

    print(
        "Same-person groups      :",
        int(
            (
                ~groups_df[
                    "cross_person"
                ]
            ).sum()
        )
    )

    print(
        "Cross-person groups     :",
        int(
            groups_df[
                "cross_person"
            ].sum()
        )
    )

    print(
        "Same-tracklet groups    :",
        int(
            (
                ~groups_df[
                    "cross_tracklet"
                ]
            ).sum()
        )
    )

    print(
        "Cross-tracklet groups   :",
        int(
            groups_df[
                "cross_tracklet"
            ].sum()
        )
    )

    print(
        "Cross-scene groups      :",
        int(
            groups_df[
                "cross_scene"
            ].sum()
        )
    )

    print(
        "Cross-view groups       :",
        int(
            groups_df[
                "cross_view"
            ].sum()
        )
    )

    print(
        "Train ↔ Eval groups     :",
        int(
            groups_df[
                "cross_train_eval"
            ].sum()
        )
    )

    print(
        "Query ↔ Gallery groups  :",
        int(
            groups_df[
                "cross_query_gallery"
            ].sum()
        )
    )


# ============================================================
# Examples
# ============================================================

if duplicate_groups > 0:

    print("\n" + "=" * 90)
    print("DUPLICATE GROUP EXAMPLES")
    print("=" * 90)


    for group_id in sorted(
        dup[
            "duplicate_group"
        ]
        .unique()
    )[:20]:

        x = dup[
            dup[
                "duplicate_group"
            ]
            == group_id
        ]

        print(
            f"\nGroup {group_id}"
        )

        print(
            "-" * 90
        )

        print(
            x[
                [
                    "official_role",
                    "scene",
                    "view",
                    "global_person_id",
                    "tracklet_id",
                    "image_number",
                    "filename",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# Save
# ============================================================

df.to_csv(
    HASHED_OUT,
    index=False
)

groups_df.to_csv(
    GROUPS_OUT,
    index=False
)

dup.to_csv(
    ROWS_OUT,
    index=False
)


print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(
    "Hashed manifest:",
    HASHED_OUT
)

print(
    "Duplicate groups:",
    GROUPS_OUT
)

print(
    "Duplicate rows:",
    ROWS_OUT
)

print("\nDONE")
