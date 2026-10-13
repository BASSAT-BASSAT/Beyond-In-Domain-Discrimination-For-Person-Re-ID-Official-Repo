from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRW"
    / "prw_manifest.csv"
)

REPORTS = (
    BASE
    / "reports/PRW"
)

SAMPLES = (
    BASE
    / "samples/PRW"
)

SAMPLES.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)


# ============================================================
# IoU
# ============================================================

def iou(a, b):

    ax1 = a["x"]
    ay1 = a["y"]
    ax2 = a["x2"]
    ay2 = a["y2"]

    bx1 = b["x"]
    by1 = b["y"]
    bx2 = b["x2"]
    by2 = b["y2"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        *
        max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
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
# SAME PID / SAME FRAME
# ============================================================

counts = (
    df[
        (
            df["frame_split"] == "train"
        )
        &
        (
            df["identity_role"] == "train_id"
        )
    ]
    .groupby(
        [
            "person_id",
            "frame_stem",
        ]
    )
    .size()
)


problem_groups = (
    counts[
        counts > 1
    ]
    .reset_index(
        name="count"
    )
)


print("=" * 100)
print("SAME PID / SAME FRAME INVESTIGATION")
print("=" * 100)

print(
    "Groups:",
    len(problem_groups)
)


pair_rows = []


for _, item in problem_groups.iterrows():

    pid = int(
        item["person_id"]
    )

    frame = item[
        "frame_stem"
    ]


    g = df[
        (
            df[
                "person_id"
            ]
            == pid
        )
        &
        (
            df[
                "frame_stem"
            ]
            == frame
        )
    ].copy()


    print(
        "\n",
        "-" * 90,
        sep=""
    )

    print(
        f"PID {pid} | "
        f"Frame {frame} | "
        f"Boxes {len(g)}"
    )


    print(
        g[
            [
                "box_index",
                "camera_id",
                "x",
                "y",
                "w",
                "h",
            ]
        ].to_string(
            index=False
        )
    )


    if len(g) == 2:

        a = g.iloc[0]
        b = g.iloc[1]

        pair_iou = iou(
            a,
            b
        )

        center_a = np.array(
            [
                a.x + a.w / 2,
                a.y + a.h / 2,
            ]
        )

        center_b = np.array(
            [
                b.x + b.w / 2,
                b.y + b.h / 2,
            ]
        )

        center_distance = float(
            np.linalg.norm(
                center_a
                - center_b
            )
        )


        print(
            f"IoU            : "
            f"{pair_iou:.6f}"
        )

        print(
            f"Center distance: "
            f"{center_distance:.3f}px"
        )


        pair_rows.append(
            {
                "person_id":
                    pid,

                "frame_stem":
                    frame,

                "iou":
                    pair_iou,

                "center_distance":
                    center_distance,
            }
        )


pair_df = pd.DataFrame(
    pair_rows
)


pair_df.to_csv(
    REPORTS
    / "prw_same_pid_same_frame_pairs.csv",
    index=False
)


# ============================================================
# DRAW SAME-PID FRAMES
# ============================================================

cards = []


for _, item in problem_groups.iterrows():

    pid = int(
        item["person_id"]
    )

    frame = item[
        "frame_stem"
    ]


    g = df[
        (
            df[
                "person_id"
            ]
            == pid
        )
        &
        (
            df[
                "frame_stem"
            ]
            == frame
        )
    ]


    image_path = Path(
        g.iloc[0][
            "frame_path"
        ]
    )


    im = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    # scale for contact sheet
    max_w = 700

    if im.width > max_w:

        scale = (
            max_w
            / im.width
        )

        im = im.resize(
            (
                int(
                    im.width
                    * scale
                ),
                int(
                    im.height
                    * scale
                ),
            )
        )

    else:

        scale = 1.0


    draw = ImageDraw.Draw(
        im
    )


    for j, (_, row) in enumerate(
        g.iterrows(),
        start=1
    ):

        x1 = (
            row.x
            * scale
        )

        y1 = (
            row.y
            * scale
        )

        x2 = (
            row.x2
            * scale
        )

        y2 = (
            row.y2
            * scale
        )


        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            width=4
        )


        draw.text(
            (
                x1 + 3,
                y1 + 3
            ),
            f"PID {pid} / box {j}"
        )


    canvas = Image.new(
        "RGB",
        (
            im.width,
            im.height + 45
        ),
        "white"
    )

    canvas.paste(
        im,
        (
            0,
            45
        )
    )


    title = ImageDraw.Draw(
        canvas
    )

    title.text(
        (
            10,
            10
        ),
        f"{frame} | PID {pid}"
    )


    cards.append(
        canvas
    )


if cards:

    width = max(
        x.width
        for x in cards
    )

    height = sum(
        x.height
        for x in cards
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

    for card in cards:

        sheet.paste(
            card,
            (
                0,
                y
            )
        )

        y += card.height


    out = (
        SAMPLES
        / "PRW_same_pid_same_frame.jpg"
    )

    sheet.save(
        out,
        quality=92
    )

    print(
        "\nSaved:",
        out
    )


# ============================================================
# WORST OOB BOXES
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


df["original_area"] = (
    df["w"]
    * df["h"]
)


df["clipped_x1"] = np.maximum(
    0,
    df["x"]
)

df["clipped_y1"] = np.maximum(
    0,
    df["y"]
)

df["clipped_x2"] = np.minimum(
    df["image_width"],
    df["x2"]
)

df["clipped_y2"] = np.minimum(
    df["image_height"],
    df["y2"]
)


df["clipped_area"] = (
    np.maximum(
        0,
        df["clipped_x2"]
        - df["clipped_x1"]
    )
    *
    np.maximum(
        0,
        df["clipped_y2"]
        - df["clipped_y1"]
    )
)


df["retained_area_fraction"] = (
    df["clipped_area"]
    /
    df["original_area"]
)


worst = (
    df[
        ~df[
            "fully_inside_image"
        ]
    ]
    .sort_values(
        "retained_area_fraction"
    )
    .head(12)
)


cards = []


for _, row in worst.iterrows():

    im = Image.open(
        row[
            "frame_path"
        ]
    ).convert(
        "RGB"
    )


    # crop a padded region around bbox
    pad = 100

    cx1 = max(
        0,
        int(
            row.x
            - pad
        )
    )

    cy1 = max(
        0,
        int(
            row.y
            - pad
        )
    )

    cx2 = min(
        im.width,
        int(
            row.x2
            + pad
        )
    )

    cy2 = min(
        im.height,
        int(
            row.y2
            + pad
        )
    )


    region = im.crop(
        (
            cx1,
            cy1,
            cx2,
            cy2
        )
    )


    draw = ImageDraw.Draw(
        region
    )


    bx1 = row.x - cx1
    by1 = row.y - cy1
    bx2 = row.x2 - cx1
    by2 = row.y2 - cy1


    draw.rectangle(
        [
            bx1,
            by1,
            bx2,
            by2,
        ],
        width=4
    )


    card = Image.new(
        "RGB",
        (
            max(
                400,
                region.width
            ),
            region.height + 65
        ),
        "white"
    )


    card.paste(
        region,
        (
            0,
            65
        )
    )


    d = ImageDraw.Draw(
        card
    )


    d.text(
        (
            8,
            8
        ),
        (
            f"{row.frame_stem} | "
            f"PID {row.person_id} | "
            f"{row.identity_role}"
        )
    )


    d.text(
        (
            8,
            30
        ),
        (
            "retained="
            f"{row.retained_area_fraction:.3f}"
        )
    )


    cards.append(
        card
    )


if cards:

    width = max(
        x.width
        for x in cards
    )

    height = sum(
        x.height
        for x in cards
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

    for card in cards:

        sheet.paste(
            card,
            (
                0,
                y
            )
        )

        y += card.height


    out = (
        SAMPLES
        / "PRW_worst_out_of_bounds.jpg"
    )

    sheet.save(
        out,
        quality=92
    )

    print(
        "Saved:",
        out
    )


print("\nDONE")
