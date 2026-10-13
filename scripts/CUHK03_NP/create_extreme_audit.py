from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

REPORT_DIR = Path.home() / "DG_ReID_Audit/reports/CUHK03_NP"
OUT = REPORT_DIR / "visual_extreme_cases.jpg"

rows = []

for variant in ["detected", "labeled"]:

    stats = pd.read_csv(
        REPORT_DIR / f"{variant}_image_stats.csv"
    )

    manifest = pd.read_csv(
        Path.home()
        / f"DG_ReID_Audit/manifests/CUHK03_NP/{variant}_manifest.csv"
    )

    df = manifest.merge(
        stats[
            [
                "filename",
                "width",
                "height",
                "aspect_ratio",
                "area",
            ]
        ],
        on="filename",
        how="left",
    )

    train = df[df["split"] == "train"]

    cases = [
        ("min_ratio", train.nsmallest(2, "aspect_ratio")),
        ("max_ratio", train.nlargest(2, "aspect_ratio")),
        ("smallest", train.nsmallest(2, "area")),
        ("largest", train.nlargest(2, "area")),
    ]

    for label, subset in cases:
        for _, r in subset.iterrows():
            rows.append(
                {
                    "variant": variant,
                    "case": label,
                    "filename": r["filename"],
                    "path": r["path"],
                    "pid": r["pid"],
                    "view": r["view"],
                    "width": r["width"],
                    "height": r["height"],
                    "ratio": r["aspect_ratio"],
                }
            )


CELL_W = 230
CELL_H = 390
COLS = 4
ROWS = (len(rows) + COLS - 1) // COLS

canvas = Image.new(
    "RGB",
    (CELL_W * COLS, CELL_H * ROWS),
    "white",
)

draw = ImageDraw.Draw(canvas)

for i, item in enumerate(rows):

    col = i % COLS
    row = i // COLS

    x0 = col * CELL_W
    y0 = row * CELL_H

    img = Image.open(item["path"]).convert("RGB")
    img.thumbnail(
        (170, 300),
        Image.Resampling.LANCZOS,
    )

    x = x0 + (CELL_W - img.width) // 2
    y = y0 + 50 + (300 - img.height) // 2

    canvas.paste(img, (x, y))

    draw.text(
        (x0 + 5, y0 + 5),
        f'{item["variant"]} | {item["case"]}',
        fill="black",
    )

    draw.text(
        (x0 + 5, y0 + 22),
        f'PID {item["pid"]} | V{item["view"]}',
        fill="black",
    )

    draw.text(
        (x0 + 5, y0 + 350),
        f'{item["width"]}x{item["height"]} ratio={item["ratio"]:.3f}',
        fill="black",
    )

    draw.text(
        (x0 + 5, y0 + 368),
        item["filename"],
        fill="black",
    )

canvas.save(OUT, quality=95)

print("Saved:", OUT)
