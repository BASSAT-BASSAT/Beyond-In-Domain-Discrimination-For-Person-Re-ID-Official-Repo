from pathlib import Path
from multiprocessing import Pool, cpu_count
import hashlib
import struct

import pandas as pd
from PIL import Image


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
)

BYTE_DUP_ROWS = (
    BASE
    / "reports/RPIfield"
    / "exact_duplicate_rows.csv"
)

OUT_DIR = (
    BASE
    / "reports/RPIfield"
)

PIXEL_ROWS_OUT = (
    OUT_DIR
    / "pixel_duplicate_rows.csv"
)

PIXEL_GROUPS_OUT = (
    OUT_DIR
    / "pixel_duplicate_groups.csv"
)


def pixel_hash(item):

    idx, path = item

    with Image.open(path) as img:

        img = img.convert("RGB")
        img.load()

        width, height = img.size

        h = hashlib.sha256()

        h.update(
            struct.pack(
                ">II",
                width,
                height,
            )
        )

        h.update(
            img.tobytes()
        )

    return (
        idx,
        h.hexdigest(),
    )


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
print("RPIFIELD DECODED-PIXEL DUPLICATE AUDIT")
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
        path,
    )
    for idx, path in enumerate(
        df["path"]
    )
]


hashes = {}


print(
    "\nDecoding and hashing pixels..."
)


with Pool(
    processes=workers
) as pool:

    for n, (
        idx,
        digest,
    ) in enumerate(
        pool.imap_unordered(
            pixel_hash,
            items,
            chunksize=200,
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


df["pixel_sha256"] = [
    hashes[i]
    for i in range(
        len(df)
    )
]


counts = (
    df["pixel_sha256"]
    .value_counts()
)

duplicate_hashes = set(
    counts[
        counts > 1
    ].index
)


dup = df[
    df[
        "pixel_sha256"
    ].isin(
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

    dup[
        "pixel_duplicate_group"
    ] = (
        dup[
            "pixel_sha256"
        ]
        .map(
            hash_to_group
        )
    )


# ============================================================
# Existing byte-duplicate paths
# ============================================================

if BYTE_DUP_ROWS.exists():

    byte_dup = pd.read_csv(
        BYTE_DUP_ROWS,
        low_memory=False,
    )

    byte_duplicate_paths = set(
        byte_dup[
            "path"
        ].astype(str)
    )

else:

    byte_duplicate_paths = set()


# ============================================================
# Group analysis
# ============================================================

group_rows = []


for digest in sorted(
    duplicate_hashes
):

    x = dup[
        dup[
            "pixel_sha256"
        ]
        == digest
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


    paths = set(
        x[
            "path"
        ].astype(str)
    )

    overlaps_byte_duplicates = (
        len(
            paths
            & byte_duplicate_paths
        )
        > 0
    )


    group_rows.append(
        {
            "pixel_duplicate_group":
                hash_to_group[
                    digest
                ],

            "pixel_sha256":
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

            "cross_person":
                persons > 1,

            "cross_sequence":
                sequences > 1,

            "cross_camera":
                cameras > 1,

            "cross_frame":
                frames > 1,

            "overlaps_byte_duplicates":
                overlaps_byte_duplicates,
        }
    )


groups = pd.DataFrame(
    group_rows
)


print("\n" + "=" * 90)
print("PIXEL DUPLICATE SUMMARY")
print("=" * 90)


print(
    "Duplicate pixel groups:",
    len(groups)
)

print(
    "Images in pixel duplicate groups:",
    len(dup)
)

print(
    "Extra pixel copies:",
    (
        len(dup)
        - len(groups)
    )
)


if len(groups) > 0:

    print("\nGROUP CHARACTERISTICS")
    print("-" * 90)

    print(
        "Cross-person groups:",
        int(
            groups[
                "cross_person"
            ].sum()
        )
    )

    print(
        "Cross-sequence groups:",
        int(
            groups[
                "cross_sequence"
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

    print(
        "Cross-frame groups:",
        int(
            groups[
                "cross_frame"
            ].sum()
        )
    )

    print(
        "Groups overlapping byte duplicates:",
        int(
            groups[
                "overlaps_byte_duplicates"
            ].sum()
        )
    )

    print(
        "Pixel-only groups:",
        int(
            (
                ~groups[
                    "overlaps_byte_duplicates"
                ]
            ).sum()
        )
    )


    print("\n" + "=" * 90)
    print("PIXEL-ONLY CROSS-PERSON GROUPS")
    print("=" * 90)


    pixel_only_cross = groups[
        (
            groups[
                "cross_person"
            ]
        )
        &
        (
            ~groups[
                "overlaps_byte_duplicates"
            ]
        )
    ]


    if len(
        pixel_only_cross
    ) == 0:

        print("None")

    else:

        for group_id in (
            pixel_only_cross[
                "pixel_duplicate_group"
            ]
            .head(30)
        ):

            x = dup[
                dup[
                    "pixel_duplicate_group"
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
    PIXEL_ROWS_OUT,
    index=False
)

groups.to_csv(
    PIXEL_GROUPS_OUT,
    index=False
)


print("\nSaved:")
print(PIXEL_ROWS_OUT)
print(PIXEL_GROUPS_OUT)

print("\nDONE")
