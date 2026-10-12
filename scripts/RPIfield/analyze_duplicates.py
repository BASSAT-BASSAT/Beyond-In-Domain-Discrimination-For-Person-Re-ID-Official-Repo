from pathlib import Path
from multiprocessing import Pool, cpu_count
import hashlib

import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
)

OUT_DIR = (
    BASE
    / "reports/RPIfield"
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

    return idx, h.hexdigest()


df = pd.read_csv(
    MANIFEST,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
        "identity_type": str,
    },
    low_memory=False,
).reset_index(
    drop=True
)


print("=" * 90)
print("RPIFIELD EXACT DUPLICATE AUDIT")
print("=" * 90)

print(
    "\nImages:",
    len(df)
)


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


items = [
    (
        idx,
        path
    )
    for idx, path in enumerate(
        df["path"]
    )
]


hashes = {}


print("\nComputing SHA256...")


with Pool(
    processes=workers
) as pool:

    for n, (
        idx,
        digest
    ) in enumerate(
        pool.imap_unordered(
            sha256_file,
            items,
            chunksize=300,
        ),
        start=1,
    ):

        hashes[idx] = digest

        if (
            n % 25000 == 0
            or n == len(items)
        ):

            print(
                f"processed "
                f"{n:,} / {len(items):,}"
            )


df["sha256"] = [
    hashes[i]
    for i in range(
        len(df)
    )
]


counts = (
    df["sha256"]
    .value_counts()
)

duplicate_hashes = set(
    counts[
        counts > 1
    ].index
)


dup = df[
    df["sha256"].isin(
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


group_rows = []


for digest in sorted(
    duplicate_hashes
):

    x = dup[
        dup["sha256"] == digest
    ]


    persons = (
        x["person_id"]
        .nunique()
    )

    sequences = (
        x["sequence_id"]
        .nunique()
    )

    cameras = (
        x["camera_id"]
        .nunique()
    )

    frames = (
        x["frame_id"]
        .nunique()
    )

    identity_types = (
        x["identity_type"]
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

            "sequences":
                sequences,

            "cameras":
                cameras,

            "frames":
                frames,

            "identity_types":
                identity_types,

            "cross_person":
                persons > 1,

            "cross_sequence":
                sequences > 1,

            "cross_camera":
                cameras > 1,

            "cross_frame":
                frames > 1,

            "mixed_identity_type":
                identity_types > 1,
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

    for column, label in [
        (
            "cross_person",
            "Cross-person groups",
        ),
        (
            "cross_sequence",
            "Cross-sequence groups",
        ),
        (
            "cross_camera",
            "Cross-camera groups",
        ),
        (
            "cross_frame",
            "Cross-frame groups",
        ),
        (
            "mixed_identity_type",
            "Known/distractor mixed groups",
        ),
    ]:

        print(
            f"{label}:",
            int(
                groups[
                    column
                ].sum()
            )
        )


    print(
        "\nLargest duplicate group:",
        int(
            groups[
                "images"
            ].max()
        )
    )


    print("\n" + "=" * 90)
    print("TOP DUPLICATE GROUPS")
    print("=" * 90)

    print(
        groups
        .sort_values(
            [
                "cross_person",
                "images",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(30)
        .to_string(
            index=False
        )
    )


    print("\n" + "=" * 90)
    print("CROSS-PERSON EXAMPLES")
    print("=" * 90)


    cross_ids = (
        groups.loc[
            groups[
                "cross_person"
            ],
            "duplicate_group",
        ]
        .head(20)
        .tolist()
    )


    if not cross_ids:

        print("None")

    else:

        for group_id in cross_ids:

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
                        "camera_id",
                        "person_id",
                        "identity_type",
                        "appearance_label",
                        "sequence_id",
                        "frame_id",
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
