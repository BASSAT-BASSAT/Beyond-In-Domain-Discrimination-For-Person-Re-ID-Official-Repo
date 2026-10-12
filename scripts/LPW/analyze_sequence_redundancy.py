from pathlib import Path
import numpy as np
import pandas as pd


MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/LPW"
    / "lpw_manifest.csv"
)

OUT = (
    Path.home()
    / "DG_ReID_Audit/manifests/LPW"
    / "lpw_sequence_stats.csv"
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "person_local_id": str,
        "global_person_id": str,
        "tracklet_id": str,
    },
)


print("=" * 90)
print("LPW NUMERIC-SEQUENCE REDUNDANCY AUDIT")
print("=" * 90)

print("\nIMPORTANT:")
print(
    "image_number is treated only as a numeric image identifier."
)
print(
    "This audit does NOT assume that it is an officially documented frame_id."
)


# ============================================================
# Tracklet-level analysis
# ============================================================

tracklet_rows = []
all_gaps = []


for tracklet_id, group in df.groupby(
    "tracklet_id",
    sort=False,
):

    numbers = np.sort(
        group[
            "image_number"
        ]
        .astype(np.int64)
        .to_numpy()
    )

    n = len(numbers)

    if n > 1:

        gaps = np.diff(
            numbers
        )

        all_gaps.extend(
            gaps.tolist()
        )

        min_gap = int(
            gaps.min()
        )

        median_gap = float(
            np.median(gaps)
        )

        mean_gap = float(
            gaps.mean()
        )

        max_gap = int(
            gaps.max()
        )

        span = int(
            numbers[-1]
            - numbers[0]
        )

    else:

        min_gap = None
        median_gap = None
        mean_gap = None
        max_gap = None
        span = 0


    first = group.iloc[0]

    tracklet_rows.append(
        {
            "tracklet_id":
                tracklet_id,

            "scene":
                first["scene"],

            "view":
                first["view"],

            "global_person_id":
                first[
                    "global_person_id"
                ],

            "images":
                n,

            "min_image_number":
                int(numbers[0]),

            "max_image_number":
                int(numbers[-1]),

            "numeric_span":
                span,

            "min_gap":
                min_gap,

            "median_gap":
                median_gap,

            "mean_gap":
                mean_gap,

            "max_gap":
                max_gap,
        }
    )


stats = pd.DataFrame(
    tracklet_rows
)

stats.to_csv(
    OUT,
    index=False,
)


# ============================================================
# Global gap statistics
# ============================================================

gaps = np.asarray(
    all_gaps,
    dtype=np.int64,
)


print("\n" + "=" * 90)
print("GLOBAL CONSECUTIVE NUMERIC GAPS")
print("=" * 90)

print(
    "Total consecutive pairs:",
    len(gaps)
)

print(
    "Min    :",
    int(
        gaps.min()
    )
)

print(
    "P01    :",
    float(
        np.quantile(
            gaps,
            0.01
        )
    )
)

print(
    "P05    :",
    float(
        np.quantile(
            gaps,
            0.05
        )
    )
)

print(
    "Median :",
    float(
        np.median(
            gaps
        )
    )
)

print(
    "Mean   :",
    round(
        float(
            gaps.mean()
        ),
        2
    )
)

print(
    "P95    :",
    float(
        np.quantile(
            gaps,
            0.95
        )
    )
)

print(
    "P99    :",
    float(
        np.quantile(
            gaps,
            0.99
        )
    )
)

print(
    "Max    :",
    int(
        gaps.max()
    )
)


# ============================================================
# Threshold distribution
# ============================================================

print("\nNUMERIC PROXIMITY")
print("-" * 90)

for threshold in [
    1,
    2,
    3,
    5,
    10,
    20,
    50,
    100,
]:

    pct = (
        gaps <= threshold
    ).mean() * 100

    print(
        f"Gap <= {threshold:<3}: "
        f"{pct:6.2f}%"
    )


# ============================================================
# Most common gaps
# ============================================================

print("\nMOST COMMON GAPS")
print("-" * 90)

gap_counts = (
    pd.Series(gaps)
    .value_counts()
    .sort_values(
        ascending=False
    )
)

print(
    gap_counts.head(
        20
    )
)


# ============================================================
# Tracklet statistics
# ============================================================

print("\n" + "=" * 90)
print("TRACKLET NUMERIC SPAN")
print("=" * 90)

print(
    stats[
        "numeric_span"
    ]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


print("\nTRACKLET MEDIAN GAP")
print("-" * 90)

print(
    stats[
        "median_gap"
    ]
    .dropna()
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


# ============================================================
# By scene
# ============================================================

print("\n" + "=" * 90)
print("BY SCENE")
print("=" * 90)

for scene in sorted(
    df["scene"].unique()
):

    scene_tracklets = set(
        df.loc[
            df["scene"] == scene,
            "tracklet_id"
        ]
    )

    x = stats[
        stats[
            "tracklet_id"
        ].isin(
            scene_tracklets
        )
    ]

    print(
        f"\n{scene.upper()}"
    )

    print(
        "Tracklets:",
        len(x)
    )

    print(
        "Median tracklet images:",
        float(
            x[
                "images"
            ].median()
        )
    )

    print(
        "Median numeric span:",
        float(
            x[
                "numeric_span"
            ].median()
        )
    )

    print(
        "Median of tracklet median gaps:",
        float(
            x[
                "median_gap"
            ]
            .dropna()
            .median()
        )
    )


# ============================================================
# Shared image_number within scene/view
# ============================================================

shared = (
    df.groupby(
        [
            "scene",
            "view",
            "image_number",
        ]
    )
    .agg(
        crops=(
            "path",
            "size"
        ),

        persons=(
            "global_person_id",
            "nunique"
        ),
    )
    .reset_index()
)


shared_multi = shared[
    shared[
        "persons"
    ] > 1
]


print("\n" + "=" * 90)
print("SHARED NUMERIC IMAGE NUMBERS")
print("=" * 90)

print(
    "(scene, view, image_number)"
    " groups with >1 person:",
    len(shared_multi)
)

print(
    "Images involved:",
    int(
        shared_multi[
            "crops"
        ].sum()
    )
)

print(
    "Maximum persons sharing one number:",
    int(
        shared_multi[
            "persons"
        ].max()
    )
)


print("\nExamples:")

print(
    shared_multi.head(
        20
    ).to_string(
        index=False
    )
)


print("\nSaved:")
print(OUT)

print("\nDONE")
