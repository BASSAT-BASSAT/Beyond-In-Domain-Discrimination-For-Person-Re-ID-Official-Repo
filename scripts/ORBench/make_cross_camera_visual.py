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

FLAGS = (
    BASE
    / "manifests/ORBench/orbench_rgb_audit_flags.csv"
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

flags = pd.read_csv(
    FLAGS,
    low_memory=False
)


# ============================================================
# REMOVE ONLY RECOMMENDED TRAIN EXCLUSIONS
# ============================================================

exclude_paths = set(
    flags.loc[
        flags["recommended_training_exclusion"] == True,
        "file_path",
    ].astype(str)
)


train = df[
    (df["split"] == "train")
    &
    (~df["file_path"].isin(exclude_paths))
].copy()


CELL_W = 190
CELL_H = 315
IMG_H = 235


def make_card(row, title):

    image = Image.open(
        ROOT / row["file_path"]
    ).convert("RGB")

    image.thumbnail(
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
        CELL_W - image.width
    ) // 2

    y = (
        IMG_H - image.height
    ) // 2

    card.paste(
        image,
        (
            x,
            y
        )
    )

    d = ImageDraw.Draw(
        card
    )

    d.text(
        (5, IMG_H + 5),
        title
    )

    d.text(
        (5, IMG_H + 24),
        (
            f"label={int(row['json_label'])} "
            f"ORID={int(row['orbench_id'])}"
        )
    )

    d.text(
        (5, IMG_H + 43),
        (
            f"{row['source_dataset']} "
            f"cam={int(row['camera_id'])}"
        )
    )

    d.text(
        (5, IMG_H + 62),
        (
            f"{int(row['width'])}x"
            f"{int(row['height'])}"
        )
    )

    return card


def blank_card(text):

    card = Image.new(
        "RGB",
        (
            CELL_W,
            CELL_H
        ),
        "white"
    )

    d = ImageDraw.Draw(
        card
    )

    d.text(
        (
            20,
            CELL_H // 2
        ),
        text
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

    header = 42

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

    for i, card in enumerate(cards):

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

    path = OUT / filename

    sheet.save(
        path,
        quality=94
    )

    print(
        "Saved:",
        path
    )


# ============================================================
# SOURCE-SPECIFIC CROSS-CAMERA SHEETS
# ============================================================

for source in [
    "llcm",
    "sysu",
]:

    src = train[
        train[
            "source_dataset"
        ]
        == source
    ].copy()


    coverage = (
        src[
            [
                "json_label",
                "camera_id",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "json_label"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )


    print("\n" + "=" * 90)
    print(
        source.upper(),
        "CROSS-CAMERA COVERAGE"
    )
    print("=" * 90)

    print(
        coverage.describe()
    )


    candidate_ids = (
        coverage[
            coverage >= 2
        ]
        .index
        .tolist()
    )


    # Up to 10 representative multi-camera IDs.
    chosen = candidate_ids[:10]


    cameras = sorted(
        src[
            "camera_id"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    cards = []


    for label in chosen:

        g = src[
            src[
                "json_label"
            ]
            == label
        ]


        for cam in cameras:

            c = g[
                g[
                    "camera_id"
                ]
                == cam
            ]


            if len(c):

                # deterministic representative
                row = (
                    c.sort_values(
                        "file_path"
                    )
                    .iloc[
                        len(c) // 2
                    ]
                )

                cards.append(
                    make_card(
                        row,
                        f"{source.upper()} ID {label}"
                    )
                )

            else:

                cards.append(
                    blank_card(
                        f"ID {label}\nCAM {cam}\nNOT SEEN"
                    )
                )


    save_grid(
        cards,
        cols=len(cameras),
        filename=(
            f"ORBench_{source}_cross_camera.jpg"
        ),
        title=(
            f"ORBench RGB - {source.upper()} "
            f"Same Training IDs Across Cameras"
        ),
    )


print("\nDONE")
