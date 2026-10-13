from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/DeepChange/extracted"
)

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/DeepChange"
)

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIRS = {
    "train": "train-set",
    "val_query": "val-set-query",
    "val_gallery": "val-set-gallery",
    "test_query": "test-set-query",
    "test_gallery": "test-set-gallery",
}


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST)

rows = []
missing = []
corrupt = []


print("=" * 80)
print("DEEPCHANGE IMAGE QUALITY AUDIT")
print("=" * 80)

print("\nManifest images:", len(df))


# ============================================================
# Scan every manifest image
# ============================================================

for i, row in df.iterrows():

    split = row["split"]
    filename = row["filename"]

    image_path = (
        ROOT
        / SPLIT_DIRS[split]
        / filename
    )

    if not image_path.exists():

        missing.append({
            "split": split,
            "filename": filename,
            "path": str(image_path),
        })

        continue


    try:

        # First pass: file integrity
        with Image.open(image_path) as img:
            img.verify()

        # Re-open because verify() invalidates decoder state
        with Image.open(image_path) as img:

            img.load()

            width, height = img.size
            mode = img.mode

        ratio = (
            width / height
            if height > 0
            else float("nan")
        )

        tiny = (
            width < 50
            or height < 100
        )

        very_narrow = (
            ratio < 0.20
        )

        horizontal = (
            ratio > 1.0
        )

        very_wide = (
            ratio > 2.0
        )

        extreme_any = (
            tiny
            or very_narrow
            or horizontal
        )


        rows.append({
            "filename": filename,
            "split": split,
            "pid": row["pid"],
            "camera_id": row["camera_id"],
            "tracklet_id": row["tracklet_id"],

            "width": width,
            "height": height,
            "aspect_ratio": ratio,
            "mode": mode,

            "tiny": tiny,
            "very_narrow": very_narrow,
            "horizontal": horizontal,
            "very_wide": very_wide,
            "extreme_any": extreme_any,

            "path": str(image_path),
        })


    except Exception as e:

        corrupt.append({
            "split": split,
            "filename": filename,
            "error": repr(e),
            "path": str(image_path),
        })


quality = pd.DataFrame(rows)


# ============================================================
# Basic integrity
# ============================================================

print("\n" + "=" * 80)
print("INTEGRITY")
print("=" * 80)

print("Successfully read :", len(quality))
print("Missing files     :", len(missing))
print("Corrupted files   :", len(corrupt))

if len(quality):

    print("\nImage modes:")
    print(
        quality["mode"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# Geometry by split
# ============================================================

def summarize_split(split):

    x = quality[
        quality["split"] == split
    ]

    print("\n" + "=" * 80)
    print(split.upper())
    print("=" * 80)

    print("Images:", len(x))

    if not len(x):
        return


    for column, label in [
        ("width", "WIDTH"),
        ("height", "HEIGHT"),
        ("aspect_ratio", "ASPECT RATIO (W/H)"),
    ]:

        s = x[column]

        print(f"\n{label}")

        print("Min    :", round(float(s.min()), 4))
        print("Mean   :", round(float(s.mean()), 4))
        print("Std    :", round(float(s.std()), 4))
        print("25%    :", round(float(s.quantile(.25)), 4))
        print("Median :", round(float(s.median()), 4))
        print("75%    :", round(float(s.quantile(.75)), 4))
        print("95%    :", round(float(s.quantile(.95)), 4))
        print("Max    :", round(float(s.max()), 4))


    print("\nDiagnostic flags:")

    for flag in [
        "tiny",
        "very_narrow",
        "horizontal",
        "very_wide",
        "extreme_any",
    ]:

        count = int(x[flag].sum())
        pct = count / len(x) * 100

        print(
            f"{flag:<15}: "
            f"{count:6d} "
            f"({pct:6.2f}%)"
        )


for split in [
    "train",
    "val_query",
    "val_gallery",
    "test_query",
    "test_gallery",
]:
    summarize_split(split)


# ============================================================
# Extreme examples from TRAIN
# ============================================================

train = quality[
    quality["split"] == "train"
]

if len(train):

    print("\n" + "=" * 80)
    print("TRAIN EXTREME EXAMPLES")
    print("=" * 80)

    smallest_width = train.loc[
        train["width"].idxmin()
    ]

    smallest_height = train.loc[
        train["height"].idxmin()
    ]

    narrowest = train.loc[
        train["aspect_ratio"].idxmin()
    ]

    widest = train.loc[
        train["aspect_ratio"].idxmax()
    ]

    print(
        "\nSmallest width:",
        smallest_width["filename"],
        f'{smallest_width["width"]}x{smallest_width["height"]}'
    )

    print(
        "Smallest height:",
        smallest_height["filename"],
        f'{smallest_height["width"]}x{smallest_height["height"]}'
    )

    print(
        "Narrowest:",
        narrowest["filename"],
        f'{narrowest["width"]}x{narrowest["height"]}',
        "ratio=",
        round(float(narrowest["aspect_ratio"]), 4)
    )

    print(
        "Widest:",
        widest["filename"],
        f'{widest["width"]}x{widest["height"]}',
        "ratio=",
        round(float(widest["aspect_ratio"]), 4)
    )


# ============================================================
# Save
# ============================================================

quality_path = (
    REPORT_DIR
    / "image_quality_audit.csv"
)

quality.to_csv(
    quality_path,
    index=False
)

pd.DataFrame(missing).to_csv(
    REPORT_DIR / "missing_images.csv",
    index=False
)

pd.DataFrame(corrupt).to_csv(
    REPORT_DIR / "corrupt_images.csv",
    index=False
)

print("\nSaved:")
print(quality_path)
print(REPORT_DIR / "missing_images.csv")
print(REPORT_DIR / "corrupt_images.csv")
