from pathlib import Path
from collections import Counter
import hashlib

import numpy as np
import pandas as pd
from PIL import Image


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/ORBench/extracted/ORBench"
)

MANIFEST = (
    BASE
    / "manifests/ORBench/orbench_rgb_manifest.csv"
)

REPORT_DIR = (
    BASE
    / "reports/ORBench"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)


print("=" * 105)
print("ORBENCH RGB QUALITY / DUPLICATE / TEMPORAL AUDIT")
print("=" * 105)

print("Rows:", len(df))
print("Train:", int((df["split"] == "train").sum()))
print("Test gallery:", int((df["split"] == "test_gallery").sum()))


# ============================================================
# IMAGE INTEGRITY + GEOMETRY + HASHES
# ============================================================

records = []

errors = []


for n, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    path = Path(
        row.absolute_path
    )

    try:

        with Image.open(path) as im:

            im.load()

            mode = im.mode

            rgb = im.convert(
                "RGB"
            )

            width, height = rgb.size

            arr = np.asarray(
                rgb
            )


            # Pixel hash includes dimensions.
            h = hashlib.sha256()

            h.update(
                f"{width}x{height}x3|".encode()
            )

            h.update(
                arr.tobytes()
            )

            pixel_sha256 = h.hexdigest()


        file_sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


        records.append(
            {
                "file_path":
                    row.file_path,

                "split":
                    row.split,

                "json_label":
                    int(
                        row.json_label
                    ),

                "orbench_id":
                    int(
                        row.orbench_id
                    ),

                "source_dataset":
                    row.source_dataset,

                "source_pid":
                    int(
                        row.source_pid
                    ),

                "camera_id":
                    (
                        int(row.camera_id)
                        if pd.notna(row.camera_id)
                        else None
                    ),

                "sequence_id":
                    (
                        int(row.sequence_id)
                        if pd.notna(row.sequence_id)
                        else None
                    ),

                "frame_number":
                    (
                        int(row.frame_number)
                        if pd.notna(row.frame_number)
                        else None
                    ),

                "width":
                    width,

                "height":
                    height,

                "aspect_ratio":
                    width / height,

                "mode":
                    mode,

                "pixel_sha256":
                    pixel_sha256,

                "file_sha256":
                    file_sha256,
            }
        )


    except Exception as e:

        errors.append(
            {
                "file_path":
                    row.file_path,

                "error":
                    repr(e),
            }
        )


    if (
        n % 5000 == 0
        or n == len(df)
    ):

        print(
            f"processed "
            f"{n:,} / {len(df):,}"
        )


audit = pd.DataFrame(
    records
)


print("\n" + "=" * 105)
print("IMAGE INTEGRITY")
print("=" * 105)


print(
    "Readable images:",
    len(audit)
)

print(
    "Unreadable/corrupt images:",
    len(errors)
)


if errors:

    print("\nFirst errors:")

    print(
        pd.DataFrame(
            errors
        )
        .head(20)
        .to_string(index=False)
    )


print(
    "\nOriginal PIL modes:"
)

print(
    audit[
        "mode"
    ]
    .value_counts()
)


# ============================================================
# GEOMETRY
# ============================================================

print("\n" + "=" * 105)
print("RGB GEOMETRY")
print("=" * 105)


for col in [
    "width",
    "height",
    "aspect_ratio",
]:

    print(f"\n{col}:")

    print(
        audit[
            col
        ]
        .describe(
            percentiles=[
                .01,
                .05,
                .25,
                .50,
                .75,
                .95,
                .99,
            ]
        )
        .round(3)
    )


print(
    "\nUnique (width, height) pairs:",
    audit[
        [
            "width",
            "height",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


print(
    "\nMost common resolutions:"
)

print(
    audit.groupby(
        [
            "width",
            "height",
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
    .head(20)
)


landscape = audit[
    audit[
        "width"
    ]
    >
    audit[
        "height"
    ]
]


square = audit[
    audit[
        "width"
    ]
    ==
    audit[
        "height"
    ]
]


print(
    "\nLandscape images:",
    len(landscape)
)

print(
    "Square images:",
    len(square)
)


# ============================================================
# METADATA COVERAGE
# ============================================================

print("\n" + "=" * 105)
print("TEMPORAL METADATA COVERAGE")
print("=" * 105)


for source in sorted(
    audit[
        "source_dataset"
    ].unique()
):

    g = audit[
        audit[
            "source_dataset"
        ]
        == source
    ]


    print(f"\n{source.upper()}")

    for col in [
        "camera_id",
        "sequence_id",
        "frame_number",
    ]:

        count = int(
            g[
                col
            ]
            .notna()
            .sum()
        )

        print(
            f"{col:<15}: "
            f"{count:,}/{len(g):,} "
            f"({100 * count / len(g):.2f}%)"
        )


# ============================================================
# PIXEL DUPLICATES
# ============================================================

pixel_group_sizes = (
    audit.groupby(
        "pixel_sha256"
    )
    .size()
)


pixel_dup_hashes = set(
    pixel_group_sizes[
        pixel_group_sizes > 1
    ]
    .index
)


pixel_dup = audit[
    audit[
        "pixel_sha256"
    ]
    .isin(
        pixel_dup_hashes
    )
].copy()


print("\n" + "=" * 105)
print("EXACT DECODED PIXEL DUPLICATES")
print("=" * 105)


print(
    "Duplicate groups:",
    len(
        pixel_dup_hashes
    )
)

print(
    "Images in duplicate groups:",
    len(
        pixel_dup
    )
)

print(
    "Extra pixel copies:",
    int(
        (
            pixel_group_sizes[
                pixel_group_sizes > 1
            ]
            - 1
        )
        .sum()
    )
)


pixel_group_records = []


for hash_value, g in pixel_dup.groupby(
    "pixel_sha256"
):

    splits = sorted(
        g[
            "split"
        ]
        .unique()
        .tolist()
    )

    labels = sorted(
        g[
            "json_label"
        ]
        .unique()
        .tolist()
    )

    orbench_ids = sorted(
        g[
            "orbench_id"
        ]
        .unique()
        .tolist()
    )

    source_keys = sorted(
        {
            (
                str(r.source_dataset),
                int(r.source_pid),
            )
            for r in g.itertuples()
        }
    )


    pixel_group_records.append(
        {
            "pixel_sha256":
                hash_value,

            "group_size":
                len(g),

            "cross_split":
                len(splits) > 1,

            "cross_json_label":
                len(labels) > 1,

            "cross_orbench_id":
                len(orbench_ids) > 1,

            "cross_original_identity":
                len(source_keys) > 1,

            "splits":
                "|".join(
                    splits
                ),

            "json_labels":
                ",".join(
                    map(
                        str,
                        labels
                    )
                ),

            "orbench_ids":
                ",".join(
                    map(
                        str,
                        orbench_ids
                    )
                ),

            "original_identities":
                "|".join(
                    f"{a}:{b}"
                    for a, b in source_keys
                ),
        }
    )


pixel_groups = pd.DataFrame(
    pixel_group_records
)


if len(
    pixel_groups
):

    print(
        "Cross-split groups:",
        int(
            pixel_groups[
                "cross_split"
            ].sum()
        )
    )

    print(
        "Cross-ORBench-ID groups:",
        int(
            pixel_groups[
                "cross_orbench_id"
            ].sum()
        )
    )

    print(
        "Cross-original-identity groups:",
        int(
            pixel_groups[
                "cross_original_identity"
            ].sum()
        )
    )


    print(
        "\nDuplicate group size distribution:"
    )

    print(
        pixel_groups[
            "group_size"
        ]
        .value_counts()
        .sort_index()
    )


    bad = pixel_groups[
        pixel_groups[
            [
                "cross_split",
                "cross_orbench_id",
                "cross_original_identity",
            ]
        ]
        .any(
            axis=1
        )
    ]


    print(
        "\nPotentially dangerous duplicate groups:",
        len(bad)
    )

    if len(bad):

        print(
            bad.head(
                30
            )
            .to_string(
                index=False
            )
        )


# ============================================================
# BYTE DUPLICATES
# ============================================================

file_group_sizes = (
    audit.groupby(
        "file_sha256"
    )
    .size()
)


file_dups = file_group_sizes[
    file_group_sizes > 1
]


print("\n" + "=" * 105)
print("BYTE-IDENTICAL DUPLICATES")
print("=" * 105)


print(
    "Duplicate groups:",
    len(
        file_dups
    )
)

print(
    "Extra byte copies:",
    int(
        (
            file_dups
            - 1
        )
        .sum()
    )
)


# ============================================================
# TRAIN-ONLY IDENTITY IMBALANCE
# ============================================================

train = audit[
    audit[
        "split"
    ]
    == "train"
].copy()


print("\n" + "=" * 105)
print("TRAIN SOURCE BALANCE")
print("=" * 105)


train_counts = (
    train.groupby(
        "json_label"
    )
    .size()
)


print(
    train_counts.describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .95,
            .99,
        ]
    )
    .round(2)
)


print(
    "\nTrain by origin:"
)

print(
    train.groupby(
        "source_dataset"
    )
    .agg(
        images=(
            "file_path",
            "size"
        ),
        identities=(
            "json_label",
            "nunique"
        ),
    )
)


# ============================================================
# TRAIN CAMERA COVERAGE
#
# Camera namespace includes source dataset.
# ============================================================

train["camera_key"] = (
    train[
        "source_dataset"
    ].astype(str)
    + ":c"
    + train[
        "camera_id"
    ]
    .astype("Int64")
    .astype(str)
)


camera_counts = (
    train.groupby(
        "json_label"
    )[
        "camera_key"
    ]
    .nunique()
)


print(
    "\nTrain camera coverage / ID:"
)

print(
    camera_counts.describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .95,
            .99,
        ]
    )
    .round(2)
)


print(
    "\nCamera coverage distribution:"
)

print(
    camera_counts
    .value_counts()
    .sort_index()
)


# ============================================================
# TEMPORAL REDUNDANCY
#
# Only rows where frame metadata is available.
# Group by:
# source dataset
# original source PID
# camera
# sequence
# ============================================================

temporal = train[
    train[
        [
            "camera_id",
            "sequence_id",
            "frame_number",
        ]
    ]
    .notna()
    .all(
        axis=1
    )
].copy()


gap_records = []


for key, g in temporal.groupby(
    [
        "source_dataset",
        "source_pid",
        "camera_id",
        "sequence_id",
    ]
):

    frames = sorted(
        g[
            "frame_number"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if len(frames) < 2:
        continue


    gaps = np.diff(
        frames
    )


    for gap in gaps:

        gap_records.append(
            {
                "source_dataset":
                    key[0],

                "gap":
                    int(gap),
            }
        )


gaps = pd.DataFrame(
    gap_records
)


print("\n" + "=" * 105)
print("TRAIN TEMPORAL REDUNDANCY")
print("=" * 105)


print(
    "Rows with complete temporal metadata:",
    len(
        temporal
    )
)

print(
    "Adjacent temporal gaps:",
    len(
        gaps
    )
)


if len(
    gaps
):

    for source in sorted(
        gaps[
            "source_dataset"
        ].unique()
    ):

        g = gaps[
            gaps[
                "source_dataset"
            ]
            == source
        ]


        print(
            f"\n{source.upper()} gaps:"
        )


        print(
            g[
                "gap"
            ]
            .describe(
                percentiles=[
                    .01,
                    .05,
                    .25,
                    .50,
                    .75,
                    .90,
                    .95,
                    .99,
                ]
            )
            .round(2)
        )


        for threshold in [
            1,
            5,
            10,
            20,
            50,
            100,
            500,
            1000,
        ]:

            count = int(
                (
                    g[
                        "gap"
                    ]
                    <= threshold
                )
                .sum()
            )

            pct = (
                100
                * count
                / len(g)
            )


            print(
                f"gap <= {threshold:4d}: "
                f"{count:6d} "
                f"({pct:6.2f}%)"
            )


# ============================================================
# SAVE
# ============================================================

audit_out = (
    BASE
    / "manifests/ORBench"
    / "orbench_rgb_quality_manifest.csv"
)

pixel_rows_out = (
    REPORT_DIR
    / "orbench_rgb_pixel_duplicate_rows.csv"
)

pixel_groups_out = (
    REPORT_DIR
    / "orbench_rgb_pixel_duplicate_groups.csv"
)

errors_out = (
    REPORT_DIR
    / "orbench_rgb_decode_errors.csv"
)

gaps_out = (
    REPORT_DIR
    / "orbench_rgb_temporal_gaps.csv"
)


audit.to_csv(
    audit_out,
    index=False
)

pixel_dup.to_csv(
    pixel_rows_out,
    index=False
)

pixel_groups.to_csv(
    pixel_groups_out,
    index=False
)

pd.DataFrame(
    errors
).to_csv(
    errors_out,
    index=False
)

gaps.to_csv(
    gaps_out,
    index=False
)


print("\n" + "=" * 105)
print("SAVED")
print("=" * 105)

print(audit_out)
print(pixel_rows_out)
print(pixel_groups_out)
print(errors_out)
print(gaps_out)

print("\nDONE")
