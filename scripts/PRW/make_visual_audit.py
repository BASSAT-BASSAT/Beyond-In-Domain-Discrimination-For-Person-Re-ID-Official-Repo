from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


random.seed(42)

BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRW"
    / "prw_source_crops_manifest.csv"
)

OUT = (
    BASE
    / "samples/PRW"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(
    MANIFEST,
    low_memory=False
)


CELL_W = 190
CELL_H = 330
IMG_H = 260


def make_card(row, title):

    im = Image.open(
        row["crop_path"]
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
            6,
            IMG_H + 8
        ),
        title
    )

    d.text(
        (
            6,
            IMG_H + 28
        ),
        (
            f"PID={int(row.person_id)} "
            f"C{int(row.camera_id)} "
            f"S{int(row.segment_id)}"
        )
    )

    d.text(
        (
            6,
            IMG_H + 48
        ),
        (
            f"frame={int(row.frame_number)}"
        )
    )

    return card


def save_grid(cards, cols, filename, title):

    rows = (
        len(cards)
        + cols - 1
    ) // cols

    header = 45

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
# 1. CAMERA OVERVIEW
# 8 random crops / camera
# ============================================================

cards = []

for cam in sorted(
    df.camera_id.unique()
):

    g = df[
        df.camera_id == cam
    ]

    sample = g.sample(
        min(
            8,
            len(g)
        ),
        random_state=42 + int(cam)
    )

    for _, row in sample.iterrows():

        cards.append(
            make_card(
                row,
                f"CAMERA {int(cam)}"
            )
        )


save_grid(
    cards,
    cols=8,
    filename="PRW_camera_overview.jpg",
    title="PRW - CAMERA OVERVIEW"
)


# ============================================================
# 2. CROSS-CAMERA IDENTITIES
# IDs visible in >= 5 cameras
# ============================================================

coverage = (
    df[
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


candidate_ids = (
    coverage[
        coverage >= 5
    ]
    .index
    .tolist()
)


chosen = candidate_ids[:8]

cards = []


for pid in chosen:

    g = df[
        df.person_id == pid
    ]

    for cam in range(
        1,
        7
    ):

        c = g[
            g.camera_id == cam
        ]

        if len(c):

            row = c.iloc[
                len(c) // 2
            ]

            cards.append(
                make_card(
                    row,
                    f"PID {int(pid)}"
                )
            )

        else:

            blank = Image.new(
                "RGB",
                (
                    CELL_W,
                    CELL_H
                ),
                "white"
            )

            d = ImageDraw.Draw(
                blank
            )

            d.text(
                (
                    20,
                    CELL_H // 2
                ),
                (
                    f"PID {pid}\n"
                    f"CAM {cam}\n"
                    f"NOT SEEN"
                )
            )

            cards.append(
                blank
            )


save_grid(
    cards,
    cols=6,
    filename="PRW_cross_camera_identities.jpg",
    title="PRW - SAME IDENTITIES ACROSS CAMERAS"
)


# ============================================================
# 3. LOW-SAMPLE IDENTITIES
# ============================================================

counts = (
    df.groupby(
        "person_id"
    )
    .size()
)


rare_ids = (
    counts[
        counts <= 3
    ]
    .sort_values()
    .index
    .tolist()
)


cards = []


for pid in rare_ids:

    g = df[
        df.person_id == pid
    ]

    for _, row in g.iterrows():

        cards.append(
            make_card(
                row,
                (
                    f"RARE PID {int(pid)} "
                    f"({len(g)} total)"
                )
            )
        )


save_grid(
    cards,
    cols=6,
    filename="PRW_rare_identities.jpg",
    title="PRW - IDENTITIES WITH <= 3 TRAINING CROPS"
)


print("\nDONE")
