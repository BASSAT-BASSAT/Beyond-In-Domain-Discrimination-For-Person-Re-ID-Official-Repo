from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_manifest_v2.csv"
)

GEOMETRY = (
    BASE
    / "manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_geometry.csv"
)

NEGATIVE = (
    BASE
    / "manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_negative_kps_v2.csv"
)

OUT = (
    BASE
    / "samples/Occluded-PoseTrack-ReID"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


manifest = pd.read_csv(
    MANIFEST,
    dtype={"video_id": str}
)

geo = pd.read_csv(
    GEOMETRY,
    dtype={"video_id": str}
)

neg = pd.read_csv(
    NEGATIVE,
    dtype={"video_id": str}
)


# ============================================================
# Helper: make contact sheet
# ============================================================

def make_sheet(df, filename, title, max_images=20):

    df = df.head(max_images).copy()

    if len(df) == 0:
        print("Skipping empty:", title)
        return

    cols = 5
    cell_w = 240
    cell_h = 430
    label_h = 70

    rows = math.ceil(
        len(df) / cols
    )

    sheet = Image.new(
        "RGB",
        (
            cols * cell_w,
            rows * cell_h + 50
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (10, 10),
        title,
        fill="black"
    )

    for i, (_, row) in enumerate(df.iterrows()):

        manifest_row = manifest[
            manifest["filename"]
            == row["filename"]
        ]

        if len(manifest_row) != 1:
            continue

        m = manifest_row.iloc[0]

        img_path = Path(
            m["path"]
        )

        with Image.open(img_path) as img:

            img = img.convert("RGB")

            max_w = cell_w - 20
            max_h = cell_h - label_h - 20

            img.thumbnail(
                (
                    max_w,
                    max_h
                )
            )

            col = i % cols
            row_idx = i // cols

            x0 = (
                col * cell_w
                + (cell_w - img.width) // 2
            )

            y0 = (
                50
                + row_idx * cell_h
                + 5
            )

            sheet.paste(
                img,
                (x0, y0)
            )

            label_y = (
                50
                + row_idx * cell_h
                + cell_h
                - label_h
            )

            label = (
                f"{m['split']} | "
                f"PID={m['person_id']}\n"
                f"V={m['video_id']}\n"
                f"{m['filename']}"
            )

            draw.text(
                (
                    col * cell_w + 5,
                    label_y
                ),
                label,
                fill="black"
            )

    output = OUT / filename

    sheet.save(
        output,
        quality=95
    )

    print("Saved:", output)


# ============================================================
# 1. Smallest crops
# ============================================================

smallest = geo.nsmallest(
    20,
    "area"
)

make_sheet(
    smallest,
    "01_smallest_crops.jpg",
    "20 Smallest Crops"
)


# ============================================================
# 2. Narrowest
# ============================================================

narrowest = geo.nsmallest(
    20,
    "aspect_ratio"
)

make_sheet(
    narrowest,
    "02_narrowest_crops.jpg",
    "20 Narrowest Crops"
)


# ============================================================
# 3. Widest / horizontal
# ============================================================

widest = geo.nlargest(
    20,
    "aspect_ratio"
)

make_sheet(
    widest,
    "03_widest_crops.jpg",
    "20 Widest / Horizontal Crops"
)


# ============================================================
# 4. Most crowded
# ============================================================

crowded = neg.sort_values(
    [
        "negative_people",
        "negative_visible_keypoints"
    ],
    ascending=False
)

make_sheet(
    crowded,
    "04_most_crowded.jpg",
    "20 Most Crowded Crops"
)


# ============================================================
# 5. No negative people
# ============================================================

no_negative = (
    neg[
        neg["negative_people"] == 0
    ]
    .sample(
        n=min(
            20,
            (
                neg["negative_people"]
                == 0
            ).sum()
        ),
        random_state=42
    )
)

make_sheet(
    no_negative,
    "05_no_negative_people.jpg",
    "Random Crops With No Negative People"
)


# ============================================================
# 6. Random samples by split
# ============================================================

for split_name in [
    "train",
    "query",
    "gallery",
]:

    sample = (
        manifest[
            manifest["split"]
            == split_name
        ]
        .sample(
            n=20,
            random_state=42
        )
    )

    make_sheet(
        sample,
        f"06_random_{split_name}.jpg",
        f"Random {split_name.upper()} Samples"
    )


# ============================================================
# 7. Exact duplicate pair
# ============================================================

duplicate_names = [
    "4624_016560_10165600065.jpg",
    "4624_016560_10165600066.jpg",
]

duplicate_rows = manifest[
    manifest["filename"]
    .isin(
        duplicate_names
    )
]

make_sheet(
    duplicate_rows,
    "07_exact_duplicate_pair.jpg",
    "Exact Byte-Duplicate Pair"
)


# ============================================================
# 8. Raw person_id overlap = 1601
# ============================================================

overlap = manifest[
    manifest["person_id"]
    == 1601
]

make_sheet(
    overlap,
    "08_person_id_1601_overlap.jpg",
    "Raw person_id=1601 — Train / Eval Overlap",
    max_images=30
)


print("\nDONE")
print("Visual audit sheets:")
print(OUT)
