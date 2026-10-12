from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image


MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_manifest_v2.csv"
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "video_id": str,
        "logical_camid": str,
        "posetrack_image_id": str,
        "annotation_id": str,
    },
)


records = []

corrupted = []
dimension_mismatches = []


print("=" * 90)
print("OCCLUDED-POSETRACK-REID IMAGE GEOMETRY AUDIT")
print("=" * 90)


# ============================================================
# Read every physical image
# ============================================================

for i, row in df.iterrows():

    path = Path(row["path"])

    try:

        with Image.open(path) as img:

            img.verify()

        # reopen after verify()
        with Image.open(path) as img:

            actual_width, actual_height = img.size
            image_format = img.format
            image_mode = img.mode

    except Exception as e:

        corrupted.append(
            {
                "path": str(path),
                "error": str(e),
            }
        )

        continue


    metadata_width = int(
        row["crop_width"]
    )

    metadata_height = int(
        row["crop_height"]
    )


    dimensions_match = (
        actual_width == metadata_width
        and actual_height == metadata_height
    )


    if not dimensions_match:

        dimension_mismatches.append(
            {
                "filename": row["filename"],
                "actual":
                    f"{actual_width}x{actual_height}",
                "metadata":
                    f"{metadata_width}x{metadata_height}",
            }
        )


    aspect_ratio = (
        actual_width
        / actual_height
    )


    records.append(
        {
            "split":
                row["split"],

            "person_id":
                row["person_id"],

            "video_id":
                row["video_id"],

            "filename":
                row["filename"],

            "width":
                actual_width,

            "height":
                actual_height,

            "area":
                actual_width
                * actual_height,

            "aspect_ratio":
                aspect_ratio,

            "horizontal":
                actual_width
                >= actual_height,

            "format":
                image_format,

            "mode":
                image_mode,
        }
    )


geo = pd.DataFrame(records)


# ============================================================
# Basic integrity
# ============================================================

print("\nIMAGE INTEGRITY")
print("-" * 90)

print(
    "Manifest images          :",
    len(df)
)

print(
    "Readable images          :",
    len(geo)
)

print(
    "Corrupted/unreadable     :",
    len(corrupted)
)

print(
    "Dimension mismatches     :",
    len(dimension_mismatches)
)


# ============================================================
# Overall geometry
# ============================================================

print("\nOVERALL GEOMETRY")
print("-" * 90)

for column in [
    "width",
    "height",
    "area",
    "aspect_ratio",
]:

    s = geo[column]

    print(f"\n{column}")

    print(
        "  min    :",
        round(float(s.min()), 4)
    )

    print(
        "  p01    :",
        round(float(s.quantile(0.01)), 4)
    )

    print(
        "  p05    :",
        round(float(s.quantile(0.05)), 4)
    )

    print(
        "  median :",
        round(float(s.median()), 4)
    )

    print(
        "  p95    :",
        round(float(s.quantile(0.95)), 4)
    )

    print(
        "  p99    :",
        round(float(s.quantile(0.99)), 4)
    )

    print(
        "  max    :",
        round(float(s.max()), 4)
    )


# ============================================================
# By split
# ============================================================

print("\nGEOMETRY BY SPLIT")
print("-" * 90)

for split_name in [
    "train",
    "query",
    "gallery",
]:

    x = geo[
        geo["split"]
        == split_name
    ]

    print(f"\n{split_name.upper()}")

    print(
        "Images       :",
        len(x)
    )

    print(
        "Width median :",
        float(
            x["width"].median()
        )
    )

    print(
        "Height median:",
        float(
            x["height"].median()
        )
    )

    print(
        "AR median    :",
        round(
            float(
                x["aspect_ratio"]
                .median()
            ),
            4
        )
    )

    print(
        "Horizontal   :",
        int(
            x["horizontal"]
            .sum()
        )
    )


# ============================================================
# Formats / modes
# ============================================================

print("\nIMAGE FORMAT")
print("-" * 90)

print(
    geo["format"]
    .value_counts()
    .to_string()
)

print("\nIMAGE MODE")
print("-" * 90)

print(
    geo["mode"]
    .value_counts()
    .to_string()
)


# ============================================================
# Geometry extremes
#
# We only REPORT extremes here.
# No deletion / filtering decision is made.
# ============================================================

print("\nGEOMETRY EXTREMES")
print("-" * 90)

print(
    "Horizontal images:",
    int(
        geo["horizontal"]
        .sum()
    )
)


print("\n10 smallest-area images:")

print(
    geo.nsmallest(
        10,
        "area"
    )[
        [
            "split",
            "person_id",
            "video_id",
            "width",
            "height",
            "aspect_ratio",
            "filename",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n10 narrowest images:")

print(
    geo.nsmallest(
        10,
        "aspect_ratio"
    )[
        [
            "split",
            "person_id",
            "video_id",
            "width",
            "height",
            "aspect_ratio",
            "filename",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\n10 widest images:")

print(
    geo.nlargest(
        10,
        "aspect_ratio"
    )[
        [
            "split",
            "person_id",
            "video_id",
            "width",
            "height",
            "aspect_ratio",
            "filename",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# Error examples
# ============================================================

if corrupted:

    print("\nCORRUPTED EXAMPLES")
    print("-" * 90)

    for x in corrupted[:20]:

        print(
            x
        )


if dimension_mismatches:

    print("\nDIMENSION MISMATCH EXAMPLES")
    print("-" * 90)

    for x in dimension_mismatches[:20]:

        print(
            x
        )


# ============================================================
# Save geometry manifest
# ============================================================

OUT = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_geometry.csv"
)

geo.to_csv(
    OUT,
    index=False
)


print("\nSaved geometry table:")
print(OUT)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
