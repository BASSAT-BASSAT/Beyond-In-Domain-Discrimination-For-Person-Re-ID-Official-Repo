from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRW"
    / "prw_manifest.csv"
)

SAMPLES = (
    BASE
    / "samples/PRW"
)

REPORTS = (
    BASE
    / "reports/PRW"
)

SAMPLES.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)


source = df[
    (df["frame_split"] == "train")
    &
    (df["identity_role"] == "train_id")
].copy()


# ============================================================
# IoU
# ============================================================

def iou(a, b):

    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)

    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = iw * ih

    area_a = (
        (a.x2 - a.x)
        *
        (a.y2 - a.y)
    )

    area_b = (
        (b.x2 - b.x)
        *
        (b.y2 - b.y)
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# Find duplicate PID/frame groups
# ============================================================

counts = (
    source.groupby(
        [
            "person_id",
            "frame_stem",
        ]
    )
    .size()
)


problems = (
    counts[
        counts > 1
    ]
    .reset_index(
        name="count"
    )
)


summary = []


for _, item in problems.iterrows():

    pid = int(
        item.person_id
    )

    frame = item.frame_stem


    g = source[
        (source["person_id"] == pid)
        &
        (source["frame_stem"] == frame)
    ]


    if len(g) != 2:
        category = "other"
        pair_iou = np.nan

    else:

        pair_iou = iou(
            g.iloc[0],
            g.iloc[1]
        )


        if pair_iou >= 0.999:

            category = (
                "exact_duplicate"
            )

        elif pair_iou >= 0.50:

            category = (
                "overlapping_duplicate"
            )

        else:

            category = (
                "distinct_conflict"
            )


    summary.append(
        {
            "person_id":
                pid,

            "frame_stem":
                frame,

            "iou":
                pair_iou,

            "category":
                category,
        }
    )


summary = pd.DataFrame(
    summary
)


print("=" * 100)
print("PRW SAME-PID SAME-FRAME CLASSIFICATION")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nCategory counts:"
)

print(
    summary[
        "category"
    ].value_counts()
)


summary.to_csv(
    REPORTS
    / "prw_same_pid_same_frame_classification.csv",
    index=False
)


# ============================================================
# Only investigate distinct conflicts
# ============================================================

conflicts = summary[
    summary[
        "category"
    ]
    == "distinct_conflict"
].copy()


# Frames that are themselves ambiguous for each PID
ambiguous_frames = {}

for pid, g in summary.groupby(
    "person_id"
):

    ambiguous_frames[
        int(pid)
    ] = set(
        g[
            "frame_stem"
        ]
    )


# ============================================================
# Crop helper
# ============================================================

CELL_W = 180
CELL_H = 300

IMAGE_H = 230


def make_crop(
    row,
    title,
):

    path = Path(
        row.frame_path
    )

    im = Image.open(
        path
    ).convert(
        "RGB"
    )


    pad_x = max(
        8,
        int(
            row.w * 0.15
        )
    )

    pad_y = max(
        8,
        int(
            row.h * 0.08
        )
    )


    x1 = max(
        0,
        int(
            np.floor(
                row.x
            )
            - pad_x
        )
    )

    y1 = max(
        0,
        int(
            np.floor(
                row.y
            )
            - pad_y
        )
    )

    x2 = min(
        im.width,
        int(
            np.ceil(
                row.x2
            )
            + pad_x
        )
    )

    y2 = min(
        im.height,
        int(
            np.ceil(
                row.y2
            )
            + pad_y
        )
    )


    crop = im.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


    crop.thumbnail(
        (
            CELL_W - 20,
            IMAGE_H
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


    px = (
        CELL_W
        - crop.width
    ) // 2


    py = (
        IMAGE_H
        - crop.height
    ) // 2


    card.paste(
        crop,
        (
            px,
            py
        )
    )


    draw = ImageDraw.Draw(
        card
    )


    draw.rectangle(
        [
            0,
            0,
            CELL_W - 1,
            CELL_H - 1,
        ],
        outline="black",
        width=1,
    )


    draw.text(
        (
            6,
            IMAGE_H + 8
        ),
        title
    )


    draw.text(
        (
            6,
            IMAGE_H + 28
        ),
        f"f={int(row.frame_number)}"
    )


    return card


def blank_card(
    text
):

    card = Image.new(
        "RGB",
        (
            CELL_W,
            CELL_H
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        card
    )

    draw.rectangle(
        [
            0,
            0,
            CELL_W - 1,
            CELL_H - 1,
        ],
        outline="black",
        width=1,
    )

    draw.text(
        (
            15,
            CELL_H // 2
        ),
        text
    )

    return card


# ============================================================
# Build temporal rows
# ============================================================

all_rows = []


for _, conflict in conflicts.iterrows():

    pid = int(
        conflict.person_id
    )

    frame_stem = conflict.frame_stem


    current = source[
        (source["person_id"] == pid)
        &
        (
            source[
                "frame_stem"
            ]
            == frame_stem
        )
    ].sort_values(
        "box_index"
    )


    base = current.iloc[0]

    camera = int(
        base.camera_id
    )

    segment = int(
        base.segment_id
    )

    current_frame = int(
        base.frame_number
    )


    # same PID, same camera + segment
    history = source[
        (source["person_id"] == pid)
        &
        (source["camera_id"] == camera)
        &
        (source["segment_id"] == segment)
    ].copy()


    # Exclude all ambiguous same-PID/frame examples
    history = history[
        ~history[
            "frame_stem"
        ].isin(
            ambiguous_frames[
                pid
            ]
        )
    ]


    before = (
        history[
            history[
                "frame_number"
            ]
            < current_frame
        ]
        .sort_values(
            "frame_number",
            ascending=False
        )
        .drop_duplicates(
            "frame_number"
        )
        .head(2)
        .sort_values(
            "frame_number"
        )
    )


    after = (
        history[
            history[
                "frame_number"
            ]
            > current_frame
        ]
        .sort_values(
            "frame_number"
        )
        .drop_duplicates(
            "frame_number"
        )
        .head(2)
    )


    cards = []


    # two previous
    before_rows = list(
        before.itertuples(
            index=False
        )
    )


    while len(
        before_rows
    ) < 2:

        before_rows.insert(
            0,
            None
        )


    for j, row in enumerate(
        before_rows
    ):

        if row is None:

            cards.append(
                blank_card(
                    "NO PREVIOUS"
                )
            )

        else:

            gap = (
                current_frame
                - int(
                    row.frame_number
                )
            )

            cards.append(
                make_crop(
                    row,
                    f"PREV Δ={gap}",
                )
            )


    # candidate A / B
    for j, row in enumerate(
        current.itertuples(
            index=False
        ),
        start=1,
    ):

        cards.append(
            make_crop(
                row,
                f"CANDIDATE {j}",
            )
        )


    # two next
    after_rows = list(
        after.itertuples(
            index=False
        )
    )


    while len(
        after_rows
    ) < 2:

        after_rows.append(
            None
        )


    for row in after_rows:

        if row is None:

            cards.append(
                blank_card(
                    "NO NEXT"
                )
            )

        else:

            gap = (
                int(
                    row.frame_number
                )
                - current_frame
            )

            cards.append(
                make_crop(
                    row,
                    f"NEXT Δ={gap}",
                )
            )


    row_width = (
        len(cards)
        * CELL_W
    )

    header_h = 55

    row_canvas = Image.new(
        "RGB",
        (
            row_width,
            CELL_H + header_h
        ),
        "white"
    )


    draw = ImageDraw.Draw(
        row_canvas
    )


    draw.text(
        (
            10,
            8
        ),
        (
            f"PID {pid} | "
            f"{frame_stem} | "
            f"Cam {camera} | "
            f"Segment {segment}"
        )
    )


    draw.text(
        (
            10,
            28
        ),
        (
            "PREV PREV | "
            "CANDIDATE 1 | "
            "CANDIDATE 2 | "
            "NEXT NEXT"
        )
    )


    x = 0

    for card in cards:

        row_canvas.paste(
            card,
            (
                x,
                header_h
            )
        )

        x += CELL_W


    all_rows.append(
        row_canvas
    )


# ============================================================
# Final sheet
# ============================================================

if all_rows:

    width = max(
        row.width
        for row in all_rows
    )

    height = sum(
        row.height
        for row in all_rows
    )


    sheet = Image.new(
        "RGB",
        (
            width,
            height
        ),
        "white"
    )


    y = 0

    for row in all_rows:

        sheet.paste(
            row,
            (
                0,
                y
            )
        )

        y += row.height


    out = (
        SAMPLES
        / "PRW_same_pid_conflict_temporal_context.jpg"
    )


    sheet.save(
        out,
        quality=95
    )


    print(
        "\nSaved:",
        out
    )


print("\nDONE")
