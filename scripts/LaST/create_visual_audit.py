from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import math

QUALITY = (
    Path.home()
    / "DG_ReID_Audit/reports/LaST/image_quality_audit.csv"
)

OUT = (
    Path.home()
    / "DG_ReID_Audit/samples/LaST"
)

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(QUALITY, low_memory=False)

train = df[
    (df["split"] == "train")
    & (df["ok"] == True)
].copy()


def make_sheet(rows, output_path, title, cols=5):

    thumb_w = 180
    thumb_h = 320
    label_h = 65

    rows_n = math.ceil(len(rows) / cols)

    canvas = Image.new(
        "RGB",
        (
            cols * thumb_w,
            rows_n * (thumb_h + label_h) + 40
        ),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 10),
        title,
        fill="black"
    )

    for i, row in enumerate(rows):

        r = i // cols
        c = i % cols

        x = c * thumb_w
        y = 40 + r * (thumb_h + label_h)

        try:
            img = Image.open(row["path"]).convert("RGB")

            img.thumbnail(
                (thumb_w - 10, thumb_h - 10)
            )

            px = x + (thumb_w - img.width) // 2
            py = y + (thumb_h - img.height) // 2

            canvas.paste(
                img,
                (px, py)
            )

        except Exception:
            pass

        text = (
            f"PID {int(row['pid']):06d}\n"
            f"{int(row['width'])}x{int(row['height'])} "
            f"r={row['aspect_ratio']:.2f}\n"
            f"C{int(row['clothes_id'])} "
            f"V{int(row['video_id'])}"
        )

        draw.text(
            (x + 5, y + thumb_h),
            text,
            fill="black"
        )

    canvas.save(output_path)


# --------------------------------------------------
# Random train samples
# --------------------------------------------------

random_sample = (
    train.sample(
        n=min(25, len(train)),
        random_state=42
    )
)

make_sheet(
    random_sample.to_dict("records"),
    OUT / "random_train.jpg",
    "LaST — Random Train Samples"
)


# --------------------------------------------------
# Tiny samples
# --------------------------------------------------

tiny = (
    train[train["tiny"] == True]
    .sort_values("area")
    .head(25)
)

make_sheet(
    tiny.to_dict("records"),
    OUT / "tiny_train.jpg",
    "LaST — Tiny Train Crops"
)


# --------------------------------------------------
# Narrow samples
# --------------------------------------------------

narrow = (
    train.sort_values("aspect_ratio")
    .head(25)
)

make_sheet(
    narrow.to_dict("records"),
    OUT / "narrow_train.jpg",
    "LaST — Narrowest Train Crops"
)


# --------------------------------------------------
# Wide samples
# --------------------------------------------------

wide = (
    train.sort_values(
        "aspect_ratio",
        ascending=False
    )
    .head(25)
)

make_sheet(
    wide.to_dict("records"),
    OUT / "wide_train.jpg",
    "LaST — Widest Train Crops"
)


# --------------------------------------------------
# Multi-clothes examples
# --------------------------------------------------

clothes_counts = (
    train.groupby("pid")["clothes_id"]
    .nunique()
)

multi_pids = clothes_counts[
    clothes_counts >= 5
].index.tolist()

selected = []

random.seed(42)
random.shuffle(multi_pids)

for pid in multi_pids[:5]:

    p = train[
        train["pid"] == pid
    ]

    for clothes_id, g in p.groupby("clothes_id"):

        selected.append(
            g.sample(
                1,
                random_state=int(pid + clothes_id)
            ).iloc[0]
        )

        if len(selected) >= 25:
            break

    if len(selected) >= 25:
        break

make_sheet(
    selected,
    OUT / "multi_clothes_train.jpg",
    "LaST — Multi-Clothes Identity Samples"
)

print("Saved visual audit sheets to:")
print(OUT)

for p in sorted(OUT.glob("*.jpg")):
    print(" ", p.name)
