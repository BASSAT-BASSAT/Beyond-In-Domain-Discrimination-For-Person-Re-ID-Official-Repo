from pathlib import Path
from PIL import Image
import pandas as pd

MANIFEST_DIR = Path.home() / "DG_ReID_Audit/manifests/CUHK03_NP"
REPORT_DIR = Path.home() / "DG_ReID_Audit/reports/CUHK03_NP"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

for variant in ["detected", "labeled"]:

    df = pd.read_csv(
        MANIFEST_DIR / f"{variant}_manifest.csv"
    )

    rows = []
    corrupt = []

    print("\n" + "=" * 70)
    print(variant.upper())
    print("=" * 70)

    for i, row in df.iterrows():

        path = Path(row["path"])

        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode

                # Force actual decoding, not just header reading
                img.load()

            rows.append({
                "filename": row["filename"],
                "split": row["split"],
                "pid": row["pid"],
                "view": row["view"],
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "area": width * height,
                "mode": mode,
            })

        except Exception as e:
            corrupt.append({
                "filename": row["filename"],
                "path": str(path),
                "error": str(e),
            })

    stats = pd.DataFrame(rows)
    corrupt_df = pd.DataFrame(corrupt)

    print("Images checked :", len(df))
    print("Readable       :", len(stats))
    print("Corrupt        :", len(corrupt))

    print("\nImage modes:")
    print(stats["mode"].value_counts())

    for split in ["train", "query", "gallery"]:

        sdf = stats[stats["split"] == split]

        print(f"\n---------- {split.upper()} ----------")

        print("Width:")
        print(
            sdf["width"].describe()[
                ["min", "mean", "50%", "max"]
            ].round(2)
        )

        print("\nHeight:")
        print(
            sdf["height"].describe()[
                ["min", "mean", "50%", "max"]
            ].round(2)
        )

        print("\nAspect ratio (W/H):")
        print(
            sdf["aspect_ratio"].describe()[
                ["min", "mean", "50%", "max"]
            ].round(3)
        )

        print("\nMost common resolutions:")
        print(
            sdf.groupby(["width", "height"])
               .size()
               .sort_values(ascending=False)
               .head(10)
        )

    stats.to_csv(
        REPORT_DIR / f"{variant}_image_stats.csv",
        index=False
    )

    corrupt_df.to_csv(
        REPORT_DIR / f"{variant}_corrupt_images.csv",
        index=False
    )

    print(
        "\nSaved:",
        REPORT_DIR / f"{variant}_image_stats.csv"
    )
