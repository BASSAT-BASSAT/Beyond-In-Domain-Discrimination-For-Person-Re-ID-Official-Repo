from pathlib import Path
from PIL import Image
import pandas as pd

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/IUST_PersonReID/iust_manifest_raw.csv"
)

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/IUST_PersonReID"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

rows = []
corrupt = []

print("=" * 70)
print("IUST IMAGE QUALITY AUDIT")
print("=" * 70)

for _, row in df.iterrows():

    path = Path(row["path"])

    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode

            # Force actual decoding
            img.load()

        rows.append({
            "filename": row["filename"],
            "split": row["split"],
            "pid": row["pid"],
            "camera": row["camera"],
            "frame": row["frame"],
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "area": width * height,
            "mode": mode,
            "path": str(path),
        })

    except Exception as e:

        corrupt.append({
            "filename": row["filename"],
            "path": str(path),
            "error": str(e),
        })


stats = pd.DataFrame(rows)
corrupt_df = pd.DataFrame(corrupt)

print("\nImages checked :", len(df))
print("Readable       :", len(stats))
print("Corrupt        :", len(corrupt))

print("\nImage modes:")
print(
    stats["mode"]
    .value_counts()
)


for split in ["train", "query", "test"]:

    sdf = stats[
        stats["split"] == split
    ]

    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    print("\nWidth:")
    print(
        sdf["width"]
        .describe()[
            ["min", "mean", "50%", "max"]
        ]
        .round(2)
    )

    print("\nHeight:")
    print(
        sdf["height"]
        .describe()[
            ["min", "mean", "50%", "max"]
        ]
        .round(2)
    )

    print("\nAspect ratio (W/H):")
    print(
        sdf["aspect_ratio"]
        .describe()[
            ["min", "mean", "50%", "max"]
        ]
        .round(3)
    )

    print("\nMost common resolutions:")
    print(
        sdf.groupby(
            ["width", "height"]
        )
        .size()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    print("\nVery small images:")

    small = sdf[
        (sdf["width"] < 50)
        | (sdf["height"] < 100)
    ]

    print("Count:", len(small))

    if len(small) > 0:
        print(
            small[
                [
                    "filename",
                    "width",
                    "height",
                    "aspect_ratio",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


stats.to_csv(
    REPORT_DIR / "image_stats.csv",
    index=False
)

corrupt_df.to_csv(
    REPORT_DIR / "corrupt_images.csv",
    index=False
)

print("\nSaved:")
print(REPORT_DIR / "image_stats.csv")
print(REPORT_DIR / "corrupt_images.csv")
