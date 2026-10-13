from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pandas as pd
import numpy as np

MANIFEST = Path.home() / "DG_ReID_Audit/manifests/LaST/last_manifest.csv"
OUT_DIR = Path.home() / "DG_ReID_Audit/reports/LaST"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST, low_memory=False)

def inspect_row(row):
    path = row["path"]

    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            img.load()

        return {
            "ok": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "width": np.nan,
            "height": np.nan,
            "mode": None,
            "error": str(e),
        }

records = df.to_dict("records")

print("Scanning images:", len(records))

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(inspect_row, records))

quality = pd.DataFrame(results)

full = pd.concat(
    [
        df.reset_index(drop=True),
        quality
    ],
    axis=1
)

full["aspect_ratio"] = (
    full["width"] / full["height"]
)

full["area"] = (
    full["width"] * full["height"]
)

# Diagnostic flags only — no deletion
full["tiny"] = (
    (full["width"] < 50)
    | (full["height"] < 100)
)

full["very_narrow"] = (
    full["aspect_ratio"] < 0.20
)

full["horizontal"] = (
    full["aspect_ratio"] > 1.0
)

full["very_wide"] = (
    full["aspect_ratio"] > 2.0
)

full["extreme_any"] = (
    full["tiny"]
    | full["very_narrow"]
    | full["horizontal"]
)

print("\n" + "=" * 80)
print("LaST IMAGE QUALITY AUDIT")
print("=" * 80)

print("\nTotal images :", len(full))
print("Readable     :", int(full["ok"].sum()))
print("Corrupted    :", int((~full["ok"]).sum()))

print("\nImage modes:")
print(full["mode"].value_counts(dropna=False))

for split in [
    "train",
    "val_query",
    "val_gallery",
    "test_query",
    "test_gallery",
]:
    sdf = full[
        (full["split"] == split)
        & (full["ok"])
    ]

    print("\n" + "=" * 80)
    print(split.upper())
    print("=" * 80)

    print("\nImages:", len(sdf))

    print("\nWidth:")
    print(
        sdf["width"]
        .describe()
        .round(2)
    )

    print("\nHeight:")
    print(
        sdf["height"]
        .describe()
        .round(2)
    )

    print("\nAspect ratio W/H:")
    print(
        sdf["aspect_ratio"]
        .describe()
        .round(3)
    )

    print("\nQuality flags:")

    for col in [
        "tiny",
        "very_narrow",
        "horizontal",
        "very_wide",
        "extreme_any",
    ]:
        n = int(sdf[col].sum())
        pct = n / len(sdf) * 100

        print(
            f"{col:<15}: "
            f"{n:7d} "
            f"({pct:6.2f}%)"
        )

print("\n" + "=" * 80)
print("TRAIN EXTREME EXAMPLES")
print("=" * 80)

train = full[
    (full["split"] == "train")
    & full["ok"]
]

print("\nSmallest width:")
print(
    train.nsmallest(
        10,
        "width"
    )[
        [
            "filename",
            "width",
            "height",
            "aspect_ratio",
        ]
    ].to_string(index=False)
)

print("\nSmallest height:")
print(
    train.nsmallest(
        10,
        "height"
    )[
        [
            "filename",
            "width",
            "height",
            "aspect_ratio",
        ]
    ].to_string(index=False)
)

print("\nNarrowest:")
print(
    train.nsmallest(
        10,
        "aspect_ratio"
    )[
        [
            "filename",
            "width",
            "height",
            "aspect_ratio",
        ]
    ].to_string(index=False)
)

print("\nWidest:")
print(
    train.nlargest(
        10,
        "aspect_ratio"
    )[
        [
            "filename",
            "width",
            "height",
            "aspect_ratio",
        ]
    ].to_string(index=False)
)

full.to_csv(
    OUT_DIR / "image_quality_audit.csv",
    index=False
)

print(
    "\nSaved:",
    OUT_DIR / "image_quality_audit.csv"
)
