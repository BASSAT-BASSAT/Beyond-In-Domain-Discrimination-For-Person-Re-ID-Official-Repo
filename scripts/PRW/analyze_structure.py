from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRW"
    / "prw_manifest.csv"
)

OUT_DIR = (
    BASE
    / "reports/PRW"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)


print("=" * 100)
print("PRW STRUCTURE / GEOMETRY / TEMPORAL AUDIT")
print("=" * 100)


# ============================================================
# FRAME GEOMETRY
# ============================================================

frames = (
    df[
        [
            "frame_stem",
            "frame_split",
            "camera_id",
            "segment_id",
            "frame_number",
            "image_width",
            "image_height",
        ]
    ]
    .drop_duplicates(
        "frame_stem"
    )
)


print("\n" + "=" * 100)
print("FRAME GEOMETRY")
print("=" * 100)

print(
    "Unique frames:",
    len(frames)
)


resolution_counts = (
    frames.groupby(
        [
            "image_width",
            "image_height",
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
)


print(
    "\nFrame resolutions:"
)

print(
    resolution_counts
)


# ============================================================
# OUT-OF-BOUNDS BOXES
# ============================================================

df["overflow_left"] = np.maximum(
    0,
    -df["x"]
)

df["overflow_top"] = np.maximum(
    0,
    -df["y"]
)

df["overflow_right"] = np.maximum(
    0,
    df["x2"]
    - df["image_width"]
)

df["overflow_bottom"] = np.maximum(
    0,
    df["y2"]
    - df["image_height"]
)


df["max_overflow_px"] = df[
    [
        "overflow_left",
        "overflow_top",
        "overflow_right",
        "overflow_bottom",
    ]
].max(
    axis=1
)


df["original_area"] = (
    df["w"]
    * df["h"]
)

df["clipped_area"] = (
    df["clipped_w"]
    * df["clipped_h"]
)

df["retained_area_fraction"] = (
    df["clipped_area"]
    / df["original_area"]
)


oob = df[
    ~df["fully_inside_image"]
].copy()


print("\n" + "=" * 100)
print("OUT-OF-BOUNDS BOXES")
print("=" * 100)


print(
    "Count:",
    len(oob)
)


print(
    "\nBy frame split / identity role:"
)

print(
    pd.crosstab(
        oob["frame_split"],
        oob["identity_role"],
        margins=True,
    )
)


if len(oob):

    print(
        "\nMaximum overflow (pixels):"
    )

    print(
        oob[
            "max_overflow_px"
        ]
        .describe(
            percentiles=[
                .50,
                .75,
                .90,
                .95,
                .99,
            ]
        )
        .round(3)
    )


    print(
        "\nRetained crop area fraction:"
    )

    print(
        oob[
            "retained_area_fraction"
        ]
        .describe(
            percentiles=[
                .01,
                .05,
                .10,
                .25,
                .50,
                .75,
                .95,
                .99,
            ]
        )
        .round(4)
    )


    print(
        "\nRows retaining < 90% area:",
        int(
            (
                oob[
                    "retained_area_fraction"
                ]
                < 0.90
            ).sum()
        )
    )

    print(
        "Rows retaining < 75% area:",
        int(
            (
                oob[
                    "retained_area_fraction"
                ]
                < 0.75
            ).sum()
        )
    )

    print(
        "Rows retaining < 50% area:",
        int(
            (
                oob[
                    "retained_area_fraction"
                ]
                < 0.50
            ).sum()
        )
    )


    print(
        "\nLargest 20 overflow cases:"
    )

    print(
        oob[
            [
                "frame_stem",
                "frame_split",
                "person_id",
                "identity_role",
                "camera_id",
                "x",
                "y",
                "w",
                "h",
                "image_width",
                "image_height",
                "max_overflow_px",
                "retained_area_fraction",
            ]
        ]
        .sort_values(
            [
                "retained_area_fraction",
                "max_overflow_px",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(20)
        .to_string(
            index=False
        )
    )


oob.to_csv(
    OUT_DIR
    / "prw_out_of_bounds_boxes.csv",
    index=False
)


# ============================================================
# CAMERA / SEGMENT STRUCTURE
# ============================================================

print("\n" + "=" * 100)
print("CAMERA / SEGMENT STRUCTURE")
print("=" * 100)


camera_segment = (
    frames.groupby(
        [
            "camera_id",
            "segment_id",
            "frame_split",
        ]
    )
    .size()
    .rename(
        "frames"
    )
    .reset_index()
)


print(
    camera_segment.to_string(
        index=False
    )
)


segment_split_check = (
    frames.groupby(
        [
            "camera_id",
            "segment_id",
        ]
    )[
        "frame_split"
    ]
    .nunique()
)


mixed_segments = (
    segment_split_check[
        segment_split_check
        > 1
    ]
)


print(
    "\nCamera-segments appearing in BOTH "
    "train and test:",
    len(mixed_segments)
)

if len(mixed_segments):

    print(
        mixed_segments
    )


# ============================================================
# SOURCE TRAINING SUBSET
# ============================================================

source = df[
    (
        df[
            "frame_split"
        ]
        == "train"
    )
    &
    (
        df[
            "identity_role"
        ]
        == "train_id"
    )
].copy()


print("\n" + "=" * 100)
print("SUPERVISED SOURCE SUBSET")
print("=" * 100)


print(
    "Boxes:",
    len(source)
)

print(
    "Unique identities:",
    source[
        "person_id"
    ].nunique()
)

print(
    "Cameras:",
    source[
        "camera_id"
    ].nunique()
)

print(
    "Segments:",
    source[
        [
            "camera_id",
            "segment_id",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# PID DISTRIBUTION
# ============================================================

pid_boxes = (
    source.groupby(
        "person_id"
    )
    .size()
)


pid_cameras = (
    source[
        [
            "person_id",
            "camera_id",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "person_id"
    )
    .size()
)


pid_segments = (
    source[
        [
            "person_id",
            "camera_id",
            "segment_id",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "person_id"
    )
    .size()
)


pid_summary = pd.DataFrame(
    {
        "boxes":
            pid_boxes,

        "camera_count":
            pid_cameras,

        "segment_count":
            pid_segments,
    }
).reset_index()


print(
    "\nBoxes per training identity:"
)

print(
    pid_summary[
        "boxes"
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
    .round(2)
)


print(
    "\nCamera coverage per training identity:"
)

print(
    pid_summary[
        "camera_count"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nCamera coverage statistics:"
)

print(
    pid_summary[
        "camera_count"
    ]
    .describe()
    .round(2)
)


print(
    "\nTop 20 identities by box count:"
)

print(
    pid_summary.sort_values(
        "boxes",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


pid_summary.to_csv(
    OUT_DIR
    / "prw_source_pid_summary.csv",
    index=False
)


# ============================================================
# SAME PID / SAME FRAME CHECK
# ============================================================

same_pid_frame = (
    source.groupby(
        [
            "person_id",
            "frame_stem",
        ]
    )
    .size()
)


duplicates_same_frame = (
    same_pid_frame[
        same_pid_frame
        > 1
    ]
)


print("\n" + "=" * 100)
print("SAME PID / SAME FRAME")
print("=" * 100)


print(
    "Groups with >1 box:",
    len(
        duplicates_same_frame
    )
)

if len(duplicates_same_frame):

    print(
        duplicates_same_frame
        .sort_values(
            ascending=False
        )
        .head(30)
    )


# ============================================================
# TEMPORAL GAPS
# ============================================================

print("\n" + "=" * 100)
print("TEMPORAL GAP AUDIT")
print("=" * 100)


temporal_rows = []


groups = source.groupby(
    [
        "person_id",
        "camera_id",
        "segment_id",
    ]
)


for (
    pid,
    camera_id,
    segment_id
), g in groups:

    frame_numbers = np.sort(
        g[
            "frame_number"
        ]
        .unique()
    )


    if len(frame_numbers) < 2:
        continue


    gaps = np.diff(
        frame_numbers
    )


    for gap in gaps:

        temporal_rows.append(
            {
                "person_id":
                    pid,

                "camera_id":
                    camera_id,

                "segment_id":
                    segment_id,

                "gap":
                    int(
                        gap
                    ),
            }
        )


gaps = pd.DataFrame(
    temporal_rows
)


print(
    "Adjacent same-PID/camera/segment pairs:",
    len(gaps)
)


if len(gaps):

    print(
        "\nFrame-gap statistics:"
    )

    print(
        gaps[
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
        25,
        50,
        100,
        250,
        500,
        1000,
    ]:

        pct = (
            gaps[
                "gap"
            ]
            .le(
                threshold
            )
            .mean()
            * 100
        )

        print(
            f"gap <= {threshold:4d}: "
            f"{pct:6.2f}%"
        )


    print(
        "\nMost common frame gaps:"
    )

    print(
        gaps[
            "gap"
        ]
        .value_counts()
        .head(20)
    )


gaps.to_csv(
    OUT_DIR
    / "prw_temporal_gaps.csv",
    index=False
)


# ============================================================
# CAMERA DISTRIBUTION OF SOURCE DATA
# ============================================================

print("\n" + "=" * 100)
print("SOURCE DATA BY CAMERA")
print("=" * 100)


camera_source = (
    source.groupby(
        "camera_id"
    )
    .agg(
        boxes=(
            "person_id",
            "size"
        ),
        identities=(
            "person_id",
            "nunique"
        ),
        frames=(
            "frame_stem",
            "nunique"
        ),
    )
)


print(
    camera_source
)


print("\nSaved:")
print(
    OUT_DIR
    / "prw_out_of_bounds_boxes.csv"
)
print(
    OUT_DIR
    / "prw_source_pid_summary.csv"
)
print(
    OUT_DIR
    / "prw_temporal_gaps.csv"
)


print("\nDONE")
