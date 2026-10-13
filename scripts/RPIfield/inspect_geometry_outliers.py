from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


BASE = Path.home() / "DG_ReID_Audit"

GEO = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_geometry.csv"
)

OUT_DIR = (
    BASE
    / "samples/RPIfield"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_OUT = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_geometry_outliers.csv"
)

SHEET_OUT = (
    OUT_DIR
    / "RPIfield_geometry_outliers.jpg"
)


df = pd.read_csv(
    GEO,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
    },
    low_memory=False,
)

df = df[
    df["readable"] == True
].copy()

df["area"] = (
    df["width"]
    * df["height"]
)


print("=" * 90)
print("RPIFIELD GEOMETRY OUTLIER INSPECTION")
print("=" * 90)


print("\nLANDSCAPE / SQUARE")
print("-" * 90)

weird_orientation = df[
    df["width"] >= df["height"]
]

print(
    weird_orientation[
        [
            "camera_id",
            "person_id",
            "identity_type",
            "appearance_label",
            "frame_id",
            "width",
            "height",
            "aspect_ratio",
            "path",
        ]
    ].to_string(
        index=False
    )
)


print("\nLOWEST ASPECT RATIOS")
print("-" * 90)

print(
    df.nsmallest(
        15,
        "aspect_ratio"
    )[
        [
            "camera_id",
            "person_id",
            "appearance_label",
            "frame_id",
            "width",
            "height",
            "aspect_ratio",
            "path",
        ]
    ].to_string(
        index=False
    )
)


print("\nHIGHEST ASPECT RATIOS")
print("-" * 90)

print(
    df.nlargest(
        15,
        "aspect_ratio"
    )[
        [
            "camera_id",
            "person_id",
            "appearance_label",
            "frame_id",
            "width",
            "height",
            "aspect_ratio",
            "path",
        ]
    ].to_string(
        index=False
    )
)


print("\nSMALLEST CROPS BY AREA")
print("-" * 90)

print(
    df.nsmallest(
        15,
        "area"
    )[
        [
            "camera_id",
            "person_id",
            "appearance_label",
            "frame_id",
            "width",
            "height",
            "aspect_ratio",
            "path",
        ]
    ].to_string(
        index=False
    )
)


print("\nLARGEST CROPS BY AREA")
print("-" * 90)

print(
    df.nlargest(
        15,
        "area"
    )[
        [
            "camera_id",
            "person_id",
            "appearance_label",
            "frame_id",
            "width",
            "height",
            "aspect_ratio",
            "path",
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Candidate pool for visual inspection
# ------------------------------------------------------------

candidate_parts = [
    weird_orientation,

    df.nsmallest(
        12,
        "aspect_ratio"
    ),

    df.nlargest(
        12,
        "aspect_ratio"
    ),

    df.nsmallest(
        12,
        "area"
    ),

    df.nlargest(
        12,
        "area"
    ),
]


candidates = (
    pd.concat(
        candidate_parts,
        ignore_index=True
    )
    .drop_duplicates(
        subset=["path"]
    )
    .reset_index(
        drop=True
    )
)


candidates.to_csv(
    CSV_OUT,
    index=False
)


# ------------------------------------------------------------
# Contact sheet
# ------------------------------------------------------------

cell_w = 220
cell_h = 300
cols = 5

rows = (
    len(candidates)
    + cols - 1
) // cols


sheet = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows * cell_h,
    ),
    "white"
)

draw = ImageDraw.Draw(
    sheet
)


for i, row in candidates.iterrows():

    col = i % cols
    rr = i // cols

    x0 = col * cell_w
    y0 = rr * cell_h


    try:

        with Image.open(
            row["path"]
        ) as img:

            img = img.convert("RGB")

            img.thumbnail(
                (
                    cell_w - 20,
                    215,
                )
            )

            px = (
                x0
                + (
                    cell_w
                    - img.width
                ) // 2
            )

            py = y0 + 5

            sheet.paste(
                img,
                (
                    px,
                    py
                )
            )


        label = (
            f"Cam {int(row['camera_id'])} "
            f"PID {int(row['person_id'])}\n"
            f"{int(row['width'])}x"
            f"{int(row['height'])} "
            f"AR={row['aspect_ratio']:.3f}\n"
            f"frame={int(row['frame_id'])}"
        )

        draw.text(
            (
                x0 + 5,
                y0 + 225
            ),
            label,
            fill="black"
        )

    except Exception as e:

        draw.text(
            (
                x0 + 5,
                y0 + 20
            ),
            str(e),
            fill="black"
        )


sheet.save(
    SHEET_OUT,
    quality=95
)


print("\nSaved:")
print(CSV_OUT)
print(SHEET_OUT)

print("\nCandidates:", len(candidates))
print("\nDONE")
