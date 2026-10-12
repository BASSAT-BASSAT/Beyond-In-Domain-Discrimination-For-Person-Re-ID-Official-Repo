from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import random

ROOT = Path.home() / "DG_ReID_Audit/datasets/CUHK03_NP/extracted/cuhk03-np"
MANIFEST = Path.home() / "DG_ReID_Audit/manifests/CUHK03_NP"
OUT = Path.home() / "DG_ReID_Audit/reports/CUHK03_NP"

OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

det = pd.read_csv(MANIFEST / "detected_manifest.csv")
lab = pd.read_csv(MANIFEST / "labeled_manifest.csv")

det = det[det["split"] == "train"]
lab = lab[lab["split"] == "train"]

common_pids = sorted(
    set(det["pid"]) &
    set(lab["pid"])
)

selected_pids = random.sample(common_pids, 12)

CELL_W = 180
CELL_H = 360
IMAGE_W = 150
IMAGE_H = 300

canvas = Image.new(
    "RGB",
    (CELL_W * 4, CELL_H * len(selected_pids)),
    "white"
)

draw = ImageDraw.Draw(canvas)

def load_fit(path):
    img = Image.open(path).convert("RGB")

    img.thumbnail(
        (IMAGE_W, IMAGE_H),
        Image.Resampling.LANCZOS
    )

    bg = Image.new(
        "RGB",
        (IMAGE_W, IMAGE_H),
        "white"
    )

    x = (IMAGE_W - img.width) // 2
    y = (IMAGE_H - img.height) // 2

    bg.paste(img, (x, y))

    return bg


for row_idx, pid in enumerate(selected_pids):

    samples = []

    for variant_name, df in [
        ("Detected", det),
        ("Labeled", lab),
    ]:

        for view in [1, 2]:

            candidates = df[
                (df["pid"] == pid) &
                (df["view"] == view)
            ]

            if len(candidates) == 0:
                samples.append(
                    (variant_name, view, None)
                )
            else:
                sample = candidates.sample(
                    1,
                    random_state=pid + view
                ).iloc[0]

                samples.append(
                    (
                        variant_name,
                        view,
                        Path(sample["path"])
                    )
                )

    for col_idx, (variant, view, path) in enumerate(samples):

        x0 = col_idx * CELL_W
        y0 = row_idx * CELL_H

        label = f"PID {pid} | {variant} | V{view}"

        draw.text(
            (x0 + 5, y0 + 5),
            label,
            fill="black"
        )

        if path is not None:

            img = load_fit(path)

            canvas.paste(
                img,
                (
                    x0 + 15,
                    y0 + 35
                )
            )

            draw.text(
                (
                    x0 + 5,
                    y0 + 338
                ),
                path.name,
                fill="black"
            )

        else:

            draw.text(
                (
                    x0 + 40,
                    y0 + 150
                ),
                "MISSING",
                fill="red"
            )


output = OUT / "visual_detected_vs_labeled.jpg"

canvas.save(
    output,
    quality=95
)

print("Selected PIDs:", selected_pids)
print("Saved:", output)
