from pathlib import Path
from itertools import combinations
import random

import pandas as pd
from PIL import Image, ImageDraw


random.seed(42)

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/DeepChange/extracted"
)

OUT = (
    Path.home()
    / "DG_ReID_Audit/samples/DeepChange"
)

OUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(
    MANIFEST,
    parse_dates=["timestamp"]
)

train = (
    df[df["split"] == "train"]
    .copy()
)

train["path"] = train["filename"].apply(
    lambda x: ROOT / "train-set" / x
)


# ============================================================
# Helpers
# ============================================================

DISPLAY_W = 128
DISPLAY_H = 256
LABEL_H = 45


def load_image(path):
    img = Image.open(path).convert("RGB")

    return img.resize(
        (DISPLAY_W, DISPLAY_H)
    )


def representative_per_tracklet(data):
    """
    Pick the middle frame of each tracklet.
    """

    rows = []

    for tracklet_id, g in data.groupby("tracklet_id"):

        g = g.sort_values("frame_id")

        row = g.iloc[len(g) // 2]

        rows.append(row)

    return pd.DataFrame(rows)


def draw_single_cell(canvas, x, y, row, label):

    img = load_image(row["path"])

    canvas.paste(
        img,
        (x, y)
    )

    draw = ImageDraw.Draw(canvas)

    draw.multiline_text(
        (
            x + 2,
            y + DISPLAY_H + 2
        ),
        label,
        fill="white",
        spacing=2,
    )


# ============================================================
# 1. RANDOM TRAIN
# ============================================================

sample = train.sample(
    n=min(25, len(train)),
    random_state=42
)

cols = 5
rows_n = 5

cell_w = DISPLAY_W
cell_h = DISPLAY_H + LABEL_H

sheet = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows_n * cell_h
    ),
    "black"
)

for i, (_, row) in enumerate(sample.iterrows()):

    col = i % cols
    r = i // cols

    x = col * cell_w
    y = r * cell_h

    label = (
        f"PID {row['pid']} C{row['camera_id']}\n"
        f"{row['timestamp'].date()}"
    )

    draw_single_cell(
        sheet,
        x,
        y,
        row,
        label,
    )

sheet.save(
    OUT / "random_train.jpg",
    quality=95
)


# ============================================================
# Tracklet representatives
# ============================================================

reps = representative_per_tracklet(train)


# ============================================================
# Build tracklet pairs
# ============================================================

same_cam_long = []
diff_cam_long = []


for pid, g in reps.groupby("pid"):

    records = list(
        g.to_dict("records")
    )

    for a, b in combinations(records, 2):

        gap_days = abs(
            (
                pd.Timestamp(b["timestamp"])
                - pd.Timestamp(a["timestamp"])
            ).total_seconds()
        ) / 86400

        if gap_days < 90:
            continue

        pair = {
            "a": a,
            "b": b,
            "gap_days": gap_days,
        }

        if a["camera_id"] == b["camera_id"]:
            same_cam_long.append(pair)
        else:
            diff_cam_long.append(pair)


# ============================================================
# Pair-sheet helper
# 10 pairs = 5 rows × 4 columns
# ============================================================

def create_pair_sheet(pairs, filename):

    random.shuffle(pairs)

    pairs = pairs[:10]

    pair_cols = 2
    image_cols = pair_cols * 2
    rows_n = 5

    cell_w = DISPLAY_W
    cell_h = DISPLAY_H + LABEL_H

    sheet = Image.new(
        "RGB",
        (
            image_cols * cell_w,
            rows_n * cell_h
        ),
        "black"
    )

    for i, pair in enumerate(pairs):

        pair_col = i % pair_cols
        r = i // pair_cols

        base_x = pair_col * 2 * cell_w
        y = r * cell_h

        for j, key in enumerate(["a", "b"]):

            row = pair[key]

            x = base_x + j * cell_w

            label = (
                f"PID {row['pid']} "
                f"C{row['camera_id']}\n"
                f"{row['timestamp'].date()}"
            )

            draw_single_cell(
                sheet,
                x,
                y,
                row,
                label
            )

        draw = ImageDraw.Draw(sheet)

        draw.text(
            (
                base_x + 2,
                y + DISPLAY_H + LABEL_H - 13
            ),
            f"gap={pair['gap_days']:.0f} days",
            fill="white",
        )

    sheet.save(
        OUT / filename,
        quality=95
    )


create_pair_sheet(
    same_cam_long,
    "same_camera_longterm.jpg"
)

create_pair_sheet(
    diff_cam_long,
    "diff_camera_longterm.jpg"
)


# ============================================================
# 4. PID TIMELINES
#
# Pick identities with:
# - >=4 unique dates
# - large temporal span
#
# One representative image per date.
# ============================================================

pid_stats = (
    reps
    .groupby("pid")
    .agg(
        dates=("date", "nunique"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
    )
)

pid_stats["span_days"] = (
    pid_stats["last"]
    - pid_stats["first"]
).dt.total_seconds() / 86400


candidates = (
    pid_stats[
        pid_stats["dates"] >= 4
    ]
    .sort_values(
        "span_days",
        ascending=False
    )
    .head(5)
)


timeline_rows = []

for pid in candidates.index:

    g = reps[
        reps["pid"] == pid
    ].copy()

    g = g.sort_values("timestamp")

    # One image per date
    selected = (
        g
        .groupby("date", sort=True)
        .first()
        .reset_index()
    )

    # Keep maximum 5 dates spread over timeline
    if len(selected) > 5:

        positions = [
            round(i * (len(selected) - 1) / 4)
            for i in range(5)
        ]

        selected = selected.iloc[
            positions
        ]

    timeline_rows.append(
        (pid, selected)
    )


cols = 5
rows_n = len(timeline_rows)

cell_w = DISPLAY_W
cell_h = DISPLAY_H + LABEL_H

sheet = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows_n * cell_h
    ),
    "black"
)


for r, (pid, selected) in enumerate(timeline_rows):

    for col, (_, row) in enumerate(selected.iterrows()):

        x = col * cell_w
        y = r * cell_h

        label = (
            f"PID {pid} C{row['camera_id']}\n"
            f"{pd.Timestamp(row['timestamp']).date()}"
        )

        draw_single_cell(
            sheet,
            x,
            y,
            row,
            label
        )


sheet.save(
    OUT / "pid_timelines.jpg",
    quality=95
)


print("=" * 70)
print("DEEPCHANGE VISUAL AUDIT")
print("=" * 70)

print("\nCreated:")

for name in [
    "random_train.jpg",
    "same_camera_longterm.jpg",
    "diff_camera_longterm.jpg",
    "pid_timelines.jpg",
]:
    print(OUT / name)

print("\nPair pools:")
print(
    "Same-camera >=90d pairs:",
    len(same_cam_long)
)
print(
    "Diff-camera >=90d pairs:",
    len(diff_cam_long)
)
