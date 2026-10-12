from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/IUST_PersonReID/iust_manifest_raw.csv"
)

OUT = (
    Path.home()
    / "DG_ReID_Audit/reports/IUST_PersonReID/temporal_sequences.jpg"
)

df = pd.read_csv(MANIFEST)
train = df[df["split"] == "train"].copy()

# Pick groups with many images
groups = (
    train.groupby(["pid", "camera"])
    .size()
    .sort_values(ascending=False)
)

selected_groups = groups.head(6).index.tolist()

CELL_W = 130
CELL_H = 270
FRAMES_PER_GROUP = 10

canvas = Image.new(
    "RGB",
    (
        CELL_W * FRAMES_PER_GROUP,
        CELL_H * len(selected_groups)
    ),
    "white"
)

draw = ImageDraw.Draw(canvas)

for row_idx, (pid, camera) in enumerate(selected_groups):

    g = train[
        (train["pid"] == pid) &
        (train["camera"] == camera)
    ].sort_values("frame")

    # Take 10 evenly spaced samples across the sequence
    if len(g) > FRAMES_PER_GROUP:
        indices = [
            round(i * (len(g) - 1) / (FRAMES_PER_GROUP - 1))
            for i in range(FRAMES_PER_GROUP)
        ]
        g = g.iloc[indices]

    for col_idx, (_, item) in enumerate(g.iterrows()):

        img = Image.open(item["path"]).convert("RGB")

        img.thumbnail(
            (110, 210),
            Image.Resampling.LANCZOS
        )

        x0 = col_idx * CELL_W
        y0 = row_idx * CELL_H

        x = x0 + (CELL_W - img.width) // 2
        y = y0 + 30

        canvas.paste(img, (x, y))

        draw.text(
            (x0 + 3, y0 + 3),
            f"PID {pid} C{camera}",
            fill="black"
        )

        draw.text(
            (x0 + 3, y0 + 245),
            f"F{item['frame']}",
            fill="black"
        )

canvas.save(OUT, quality=95)

print("Selected groups:")
for group in selected_groups:
    print(group)

print("\nSaved:", OUT)
