from pathlib import Path
from multiprocessing import Pool, cpu_count
import hashlib
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRID2011"
    / "prid2011_manifest.csv"
)

OUT_DIR = (
    BASE
    / "reports/PRID2011"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ROWS_OUT = (
    OUT_DIR
    / "exact_duplicate_rows.csv"
)

GROUPS_OUT = (
    OUT_DIR
    / "exact_duplicate_groups.csv"
)


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


df = pd.read_csv(
    MANIFEST,
    dtype={
        "global_person_id": str,
        "trajectory_id": str,
    },
)

multi = df[
    df["version"] == "multi_shot"
].copy()

multi = multi.reset_index(
    drop=True
)


print("=" * 90)
print("PRID2011 MULTI-SHOT EXACT DUPLICATE AUDIT")
print("=" * 90)

print(
    "\nImages:",
    len(multi)
)


# ============================================================
# File-size prefilter
# ============================================================

print("\nReading file sizes...")

multi["file_size"] = [
    Path(path).stat().st_size
    for path in multi["path"]
]

size_counts = (
    multi["file_size"]
    .value_counts()
)

candidate_mask = (
    multi["file_size"]
    .map(size_counts)
    > 1
)

candidate_indices = (
    multi.index[
        candidate_mask
    ]
    .tolist()
)

print(
    "Images in repeated-size groups:",
    len(candidate_indices)
)

print(
    "Unique-size images:",
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
    "\nComputing SHA256..."
)


items = [
    (
        idx,
        multi.at[
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
            chunksize=200,
        ),
        start=1,
    ):

        hash_map[idx] = digest

        if (
            n % 10000 == 0
            or n == len(items)
        ):

            print(
                f"processed "
                f"{n:,} / {len(items):,}"
            )


multi["sha256"] = None

for idx, digest in hash_map.items():

    multi.at[
        idx,
        "sha256"
    ] = digest


hashed = multi[
    multi["sha256"].notna()
].copy()


counts = (
    hashed["sha256"]
    .value_counts()
)

duplicate_hashes = set(
    counts[
        counts > 1
    ].index
)


dup = hashed[
    hashed["sha256"].isin(
        duplicate_hashes
    )
].copy()


hash_to_group = {
    digest: i
    for i, digest in enumerate(
        sorted(
            duplicate_hashes
        ),
        start=1,
    )
}


if len(dup) > 0:

    dup["duplicate_group"] = (
        dup["sha256"]
        .map(
            hash_to_group
        )
    )


# ============================================================
# Group analysis
# ============================================================

group_rows = []


for digest in sorted(
    duplicate_hashes
):

    x = dup[
        dup["sha256"] == digest
    ]

    persons = (
        x["global_person_id"]
        .nunique()
    )

    trajectories = (
        x["trajectory_id"]
        .nunique()
    )

    cameras = (
        x["camera"]
        .nunique()
    )


    group_rows.append(
        {
            "duplicate_group":
                hash_to_group[
                    digest
                ],

            "sha256":
                digest,

            "images":
                len(x),

            "persons":
                persons,

            "trajectories":
                trajectories,

            "cameras":
                cameras,

            "cross_person":
                persons > 1,

            "cross_trajectory":
                trajectories > 1,

            "cross_camera":
                cameras > 1,
        }
    )


groups = pd.DataFrame(
    group_rows
)


duplicate_groups = len(
    groups
)

images_in_groups = len(
    dup
)

extra_copies = (
    images_in_groups
    - duplicate_groups
)


print("\n" + "=" * 90)
print("EXACT DUPLICATE SUMMARY")
print("=" * 90)

print(
    "Duplicate groups:",
    duplicate_groups
)

print(
    "Images in duplicate groups:",
    images_in_groups
)

print(
    "Extra duplicate copies:",
    extra_copies
)


if duplicate_groups > 0:

    print("\nGROUP CHARACTERISTICS")
    print("-" * 90)

    print(
        "Same-person groups:",
        int(
            (
                ~groups[
                    "cross_person"
                ]
            ).sum()
        )
    )

    print(
        "Cross-person groups:",
        int(
            groups[
                "cross_person"
            ].sum()
        )
    )

    print(
        "Same-trajectory groups:",
        int(
            (
                ~groups[
                    "cross_trajectory"
                ]
            ).sum()
        )
    )

    print(
        "Cross-trajectory groups:",
        int(
            groups[
                "cross_trajectory"
            ].sum()
        )
    )

    print(
        "Cross-camera groups:",
        int(
            groups[
                "cross_camera"
            ].sum()
        )
    )


# ============================================================
# Examples
# ============================================================

if duplicate_groups > 0:

    print("\n" + "=" * 90)
    print("EXAMPLES")
    print("=" * 90)

    for group_id in sorted(
        dup[
            "duplicate_group"
        ].unique()
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
            x[
                [
                    "camera",
                    "global_person_id",
                    "trajectory_id",
                    "sequence_index",
                    "filename",
                ]
            ]
            .to_string(
                index=False
            )
        )


dup.to_csv(
    ROWS_OUT,
    index=False
)

groups.to_csv(
    GROUPS_OUT,
    index=False
)


print("\nSaved:")
print(ROWS_OUT)
print(GROUPS_OUT)

print("\nDONE")
