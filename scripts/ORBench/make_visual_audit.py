from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/ORBench/extracted/ORBench"
)

MANIFEST = (
    BASE
    / "manifests/ORBench/orbench_rgb_quality_manifest.csv"
)

DUP_ROWS = (
    BASE
    / "reports/ORBench/orbench_rgb_pixel_duplicate_rows.csv"
)

OUT = (
    BASE
    / "samples/ORBench"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

dups = pd.read_csv(
    DUP_ROWS,
    low_memory=False
)


CELL_W = 200
CELL_H = 320
IMG_H = 245


def make_card(row, title=""):

    path = (
        ROOT
        / row["file_path"]
    )

    im = Image.open(
        path
    ).convert("RGB")

    im.thumbnail(
        (
            CELL_W - 20,
            IMG_H
        )
    )

    card = Image.new(
        "RGB",
        (
            CELL_W,
            CELL_H
        ),
        "white"
    )

    x = (
        CELL_W
        - im.width
    ) // 2

    y = (
        IMG_H
        - im.height
    ) // 2

    card.paste(
        im,
        (
            x,
            y
        )
    )

    d = ImageDraw.Draw(
        card
    )

    d.text(
        (
            5,
            IMG_H + 5
        ),
        title
    )

    d.text(
        (
            5,
            IMG_H + 23
        ),
        (
            f"ID={int(row['orbench_id'])} "
            f"{row['source_dataset']} "
            f"C={row['camera_id']}"
        )
    )

    d.text(
        (
            5,
            IMG_H + 41
        ),
        (
            f"{int(row['width'])}x"
            f"{int(row['height'])} "
            f"AR={row['aspect_ratio']:.2f}"
        )
    )

    d.text(
        (
            5,
            IMG_H + 59
        ),
        str(
            row["split"]
        )
    )

    return card


def save_grid(
    cards,
    cols,
    filename,
    title
):

    rows = (
        len(cards)
        + cols - 1
    ) // cols

    header = 40

    sheet = Image.new(
        "RGB",
        (
            cols * CELL_W,
            header + rows * CELL_H
        ),
        "white"
    )

    d = ImageDraw.Draw(
        sheet
    )

    d.text(
        (
            10,
            12
        ),
        title
    )

    for i, card in enumerate(
        cards
    ):

        x = (
            i % cols
        ) * CELL_W

        y = (
            i // cols
        ) * CELL_H + header

        sheet.paste(
            card,
            (
                x,
                y
            )
        )

    path = (
        OUT
        / filename
    )

    sheet.save(
        path,
        quality=94
    )

    print(
        "Saved:",
        path
    )


# ============================================================
# SUMMARY
# ============================================================

print("=" * 100)
print("ORBENCH VISUAL AUDIT PRECHECK")
print("=" * 100)


landscape = df[
    df["width"]
    >
    df["height"]
].copy()


square = df[
    df["width"]
    ==
    df["height"]
].copy()


print(
    "\nLandscape:",
    len(
        landscape
    )
)

print(
    landscape.groupby(
        [
            "split",
            "source_dataset",
        ]
    )
    .size()
)


print(
    "\nSquare:",
    len(
        square
    )
)

if len(square):
    print(
        square[
            [
                "file_path",
                "split",
                "source_dataset",
                "orbench_id",
                "width",
                "height",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 1. LANDSCAPE / EXTREME ASPECT RATIO
# ============================================================

extreme = (
    df.sort_values(
        "aspect_ratio",
        ascending=False
    )
    .head(60)
)


cards = []

for _, row in extreme.iterrows():

    cards.append(
        make_card(
            row,
            "HIGH ASPECT RATIO"
        )
    )


save_grid(
    cards,
    cols=6,
    filename="ORBench_extreme_aspect_ratio.jpg",
    title="ORBench RGB - Highest Aspect Ratios"
)


# ============================================================
# 2. SMALLEST CROPS
# ============================================================

df["area"] = (
    df["width"]
    *
    df["height"]
)


small = (
    df.sort_values(
        "area"
    )
    .head(60)
)


cards = []

for _, row in small.iterrows():

    cards.append(
        make_card(
            row,
            "SMALL CROP"
        )
    )


save_grid(
    cards,
    cols=6,
    filename="ORBench_smallest_crops.jpg",
    title="ORBench RGB - Smallest Crops"
)


# ============================================================
# 3. EXACT DUPLICATE PAIRS
# ============================================================

cards = []


for group_num, (
    hash_value,
    g
) in enumerate(
    dups.groupby(
        "pixel_sha256"
    ),
    start=1
):

    g = g.sort_values(
        "file_path"
    )

    for _, row in g.iterrows():

        cards.append(
            make_card(
                row,
                f"DUP GROUP {group_num}"
            )
        )


save_grid(
    cards,
    cols=6,
    filename="ORBench_exact_duplicate_pairs.jpg",
    title="ORBench RGB - Exact Pixel Duplicate Groups"
)


# ============================================================
# 4. SOURCE / CAMERA OVERVIEW
#
# Camera namespaces remain source-specific.
# ============================================================

cards = []


camera_keys = (
    df[
        [
            "source_dataset",
            "camera_id",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "source_dataset",
            "camera_id",
        ]
    )
)


for _, key in camera_keys.iterrows():

    source = key[
        "source_dataset"
    ]

    camera = key[
        "camera_id"
    ]


    g = df[
        (
            df[
                "source_dataset"
            ]
            == source
        )
        &
        (
            df[
                "camera_id"
            ]
            == camera
        )
    ]


    sample = g.sample(
        min(
            4,
            len(g)
        ),
        random_state=42
    )


    for _, row in sample.iterrows():

        cards.append(
            make_card(
                row,
                (
                    f"{source.upper()} "
                    f"CAM {int(camera)}"
                )
            )
        )


save_grid(
    cards,
    cols=4,
    filename="ORBench_source_camera_overview.jpg",
    title="ORBench RGB - Source-specific Camera Overview"
)


# ============================================================
# SAVE OUTLIER TABLE
# ============================================================

outliers = df[
    (
        df["width"]
        >=
        df["height"]
    )
    |
    (
        df["aspect_ratio"]
        > 1
    )
    |
    (
        df["aspect_ratio"]
        < 0.18
    )
].copy()


outlier_csv = (
    BASE
    / "reports/ORBench"
    / "orbench_rgb_geometry_outliers.csv"
)


outliers.to_csv(
    outlier_csv,
    index=False
)


print(
    "\nGeometry outlier rows:",
    len(
        outliers
    )
)

print(
    "Saved:",
    outlier_csv
)

print("\nDONE")
