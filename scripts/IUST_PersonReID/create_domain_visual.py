from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/IUST_PersonReID/iust_manifest_raw.csv"
)

STATS = (
    Path.home()
    / "DG_ReID_Audit/reports/IUST_PersonReID/image_stats.csv"
)

OUT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/IUST_PersonReID"
)

df = pd.read_csv(MANIFEST)
stats = pd.read_csv(STATS)

train = df[df["split"] == "train"].copy()

# =========================================================
# 1. CAMERA VISUAL SHEET
# =========================================================

cameras = sorted(train["camera"].unique())

SAMPLES_PER_CAMERA = 5
CELL_W = 150
CELL_H = 300

canvas = Image.new(
    "RGB",
    (
        CELL_W * SAMPLES_PER_CAMERA,
        CELL_H * len(cameras)
    ),
    "white"
)

draw = ImageDraw.Draw(canvas)

for row_idx, camera in enumerate(cameras):

    subset = train[
        train["camera"] == camera
    ]

    # deterministic random sample
    samples = subset.sample(
        min(SAMPLES_PER_CAMERA, len(subset)),
        random_state=42 + camera
    )

    for col_idx, (_, item) in enumerate(samples.iterrows()):

        img = Image.open(item["path"]).convert("RGB")

        img.thumbnail(
            (120, 220),
            Image.Resampling.LANCZOS
        )

        x0 = col_idx * CELL_W
        y0 = row_idx * CELL_H

        x = x0 + (CELL_W - img.width) // 2
        y = y0 + 30

        canvas.paste(img, (x, y))

        draw.text(
            (x0 + 5, y0 + 5),
            f"C{camera} | PID {item['pid']}",
            fill="black"
        )

        draw.text(
            (x0 + 5, y0 + 260),
            f"F{item['frame']}",
            fill="black"
        )

camera_output = OUT_DIR / "visual_by_camera.jpg"
canvas.save(camera_output, quality=95)

# =========================================================
# 2. EXTREME IMAGE SHEET
# =========================================================

train_stats = stats[
    stats["split"] == "train"
].copy()

cases = []

cases.extend(
    [("min_ratio", r) for _, r in
     train_stats.nsmallest(5, "aspect_ratio").iterrows()]
)

cases.extend(
    [("max_ratio", r) for _, r in
     train_stats.nlargest(5, "aspect_ratio").iterrows()]
)

cases.extend(
    [("smallest_area", r) for _, r in
     train_stats.nsmallest(5, "area").iterrows()]
)

COLS = 5
ROWS = 3

canvas2 = Image.new(
    "RGB",
    (230 * COLS, 380 * ROWS),
    "white"
)

draw2 = ImageDraw.Draw(canvas2)

for i, (label, item) in enumerate(cases):

    col = i % COLS
    row = i // COLS

    x0 = col * 230
    y0 = row * 380

    img = Image.open(item["path"]).convert("RGB")

    img.thumbnail(
        (180, 280),
        Image.Resampling.LANCZOS
    )

    x = x0 + (230 - img.width) // 2
    y = y0 + 45

    canvas2.paste(img, (x, y))

    draw2.text(
        (x0 + 5, y0 + 5),
        label,
        fill="black"
    )

    draw2.text(
        (x0 + 5, y0 + 22),
        f"PID {item['pid']} C{item['camera']}",
        fill="black"
    )

    draw2.text(
        (x0 + 5, y0 + 330),
        f"{item['width']}x{item['height']}",
        fill="black"
    )

    draw2.text(
        (x0 + 5, y0 + 348),
        f"ratio={item['aspect_ratio']:.3f}",
        fill="black"
    )

extreme_output = OUT_DIR / "visual_extreme_images.jpg"
canvas2.save(extreme_output, quality=95)

print("Saved:")
print(camera_output)
print(extreme_output)
