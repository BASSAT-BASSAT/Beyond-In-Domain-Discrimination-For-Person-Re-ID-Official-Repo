from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
from PIL import Image


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRID2011"
    / "prid2011_manifest.csv"
)

OUT_DIR = (
    BASE
    / "manifests/PRID2011"
)

GEOMETRY_OUT = (
    OUT_DIR
    / "prid2011_geometry.csv"
)

CORRUPT_OUT = (
    OUT_DIR
    / "prid2011_corrupt_images.csv"
)


def inspect_image(item):

    idx, path = item

    try:

        with Image.open(path) as img:

            width, height = img.size
            mode = img.mode
            fmt = img.format

            img.load()

        return {
            "idx": idx,
            "ok": True,
            "width": width,
            "height": height,
            "area": width * height,
            "aspect_ratio": width / height,
            "mode": mode,
            "format": fmt,
            "error": None,
        }

    except Exception as e:

        return {
            "idx": idx,
            "ok": False,
            "width": None,
            "height": None,
            "area": None,
            "aspect_ratio": None,
            "mode": None,
            "format": None,
            "error": str(e),
        }


df = pd.read_csv(
    MANIFEST,
    dtype={
        "person_local_id": str,
        "global_person_id": str,
        "trajectory_id": str,
    },
)


items = list(
    enumerate(
        df["path"].tolist()
    )
)


print("=" * 90)
print("PRID2011 IMAGE INTEGRITY / GEOMETRY AUDIT")
print("=" * 90)

print(
    "\nImages:",
    len(items)
)


workers = min(
    8,
    max(
        1,
        cpu_count() - 1
    )
)

print(
    "Workers:",
    workers
)

print(
    "\nDecoding images..."
)


results = []


with Pool(
    processes=workers
) as pool:

    for n, result in enumerate(
        pool.imap_unordered(
            inspect_image,
            items,
            chunksize=200,
        ),
        start=1,
    ):

        results.append(
            result
        )

        if (
            n % 10000 == 0
            or n == len(items)
        ):

            print(
                f"processed "
                f"{n:,} / {len(items):,}"
            )


results = (
    pd.DataFrame(results)
    .sort_values("idx")
)


good = results[
    results["ok"]
].copy()

bad = results[
    ~results["ok"]
].copy()


manifest_indexed = (
    df.reset_index()
    .rename(
        columns={
            "index": "idx"
        }
    )
)


good = good.merge(
    manifest_indexed,
    on="idx",
    how="left"
)

bad = bad.merge(
    manifest_indexed,
    on="idx",
    how="left"
)


good.to_csv(
    GEOMETRY_OUT,
    index=False
)

bad.to_csv(
    CORRUPT_OUT,
    index=False
)


print("\n" + "=" * 90)
print("IMAGE INTEGRITY")
print("=" * 90)

print(
    "Manifest images  :",
    len(df)
)

print(
    "Readable images  :",
    len(good)
)

print(
    "Corrupted images :",
    len(bad)
)


print("\nFORMATS")
print("-" * 90)

print(
    good["format"]
    .value_counts()
)


print("\nMODES")
print("-" * 90)

print(
    good["mode"]
    .value_counts()
)


def print_stats(title, x):

    print(f"\n{title}")

    print(
        x.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .round(2)
    )


print("\n" + "=" * 90)
print("OVERALL GEOMETRY")
print("=" * 90)

print_stats(
    "WIDTH",
    good["width"]
)

print_stats(
    "HEIGHT",
    good["height"]
)

print_stats(
    "ASPECT RATIO",
    good["aspect_ratio"]
)


print("\n" + "=" * 90)
print("MOST COMMON RESOLUTIONS")
print("=" * 90)

resolution_counts = (
    good.groupby(
        [
            "width",
            "height",
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    resolution_counts.head(20)
)

print(
    "\nUnique resolutions:",
    len(resolution_counts)
)


print("\n" + "=" * 90)
print("BY VERSION / CAMERA")
print("=" * 90)

for version in [
    "single_shot",
    "multi_shot",
]:

    for camera in [
        "cam_a",
        "cam_b",
    ]:

        x = good[
            (good["version"] == version)
            &
            (good["camera"] == camera)
        ]

        print(
            f"\n{version} / {camera}"
        )

        print(
            "Images:",
            len(x)
        )

        print(
            "Unique resolutions:",
            x[
                [
                    "width",
                    "height",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(
            "Most common:"
        )

        print(
            x.groupby(
                [
                    "width",
                    "height",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
            .head(10)
        )


print("\nSaved:")
print(GEOMETRY_OUT)
print(CORRUPT_OUT)

print("\nDONE")
