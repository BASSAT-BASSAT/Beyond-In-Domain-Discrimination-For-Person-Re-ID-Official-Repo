from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

import pandas as pd
from PIL import Image


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/LPW"
    / "lpw_manifest.csv"
)

OUT_DIR = (
    BASE
    / "manifests/LPW"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GEOMETRY_OUT = (
    OUT_DIR
    / "lpw_geometry.csv"
)

CORRUPT_OUT = (
    OUT_DIR
    / "lpw_corrupt_images.csv"
)


# ============================================================
# Worker
# ============================================================

def inspect_image(item):

    idx, path = item

    try:

        with Image.open(path) as img:

            width, height = img.size
            mode = img.mode
            fmt = img.format

            # Force actual decode
            img.load()

        return {
            "idx": idx,
            "ok": True,
            "width": width,
            "height": height,
            "area": width * height,
            "aspect_ratio": (
                width / height
                if height > 0
                else None
            ),
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


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(
    MANIFEST,
    dtype={
        "person_local_id": str,
        "global_person_id": str,
        "tracklet_id": str,
    },
)


items = list(
    enumerate(
        df["path"].tolist()
    )
)


print("=" * 90)
print("LPW IMAGE INTEGRITY / GEOMETRY AUDIT")
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

print("\nDecoding images...")


results = []


with Pool(
    processes=workers
) as pool:

    for n, result in enumerate(
        pool.imap_unordered(
            inspect_image,
            items,
            chunksize=500,
        ),
        start=1,
    ):

        results.append(
            result
        )

        if (
            n % 50000 == 0
            or n == len(items)
        ):

            print(
                f"  processed "
                f"{n:,} / {len(items):,}"
            )


results_df = pd.DataFrame(
    results
).sort_values(
    "idx"
)


# ============================================================
# Separate readable / corrupt
# ============================================================

good = results_df[
    results_df["ok"]
].copy()

bad = results_df[
    ~results_df["ok"]
].copy()


# Merge metadata back
good = good.merge(
    df.reset_index().rename(
        columns={
            "index": "idx"
        }
    ),
    on="idx",
    how="left",
)


bad = bad.merge(
    df.reset_index().rename(
        columns={
            "index": "idx"
        }
    ),
    on="idx",
    how="left",
)


# ============================================================
# Save
# ============================================================

geometry_cols = [
    "dataset",
    "scene",
    "view",
    "person_local_id",
    "global_person_id",
    "tracklet_id",
    "official_split",
    "official_role",
    "image_number",
    "filename",
    "path",
    "width",
    "height",
    "area",
    "aspect_ratio",
    "mode",
    "format",
]


good[
    geometry_cols
].to_csv(
    GEOMETRY_OUT,
    index=False,
)


bad.to_csv(
    CORRUPT_OUT,
    index=False,
)


# ============================================================
# Integrity
# ============================================================

print("\n" + "=" * 90)
print("IMAGE INTEGRITY")
print("=" * 90)

print(
    "Manifest images      :",
    len(df)
)

print(
    "Readable images      :",
    len(good)
)

print(
    "Corrupted images     :",
    len(bad)
)


print("\nImage formats:")

print(
    good["format"]
    .value_counts()
)


print("\nImage modes:")

print(
    good["mode"]
    .value_counts()
)


# ============================================================
# Geometry helper
# ============================================================

def print_stats(name, x):

    print(
        f"\n{name}"
    )

    print(
        "Min    :",
        round(
            float(x.min()),
            4
        )
    )

    print(
        "P01    :",
        round(
            float(
                x.quantile(0.01)
            ),
            4
        )
    )

    print(
        "P05    :",
        round(
            float(
                x.quantile(0.05)
            ),
            4
        )
    )

    print(
        "Median :",
        round(
            float(
                x.median()
            ),
            4
        )
    )

    print(
        "Mean   :",
        round(
            float(
                x.mean()
            ),
            4
        )
    )

    print(
        "P95    :",
        round(
            float(
                x.quantile(0.95)
            ),
            4
        )
    )

    print(
        "P99    :",
        round(
            float(
                x.quantile(0.99)
            ),
            4
        )
    )

    print(
        "Max    :",
        round(
            float(x.max()),
            4
        )
    )


# ============================================================
# Overall geometry
# ============================================================

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
    "AREA",
    good["area"]
)

print_stats(
    "ASPECT RATIO (W/H)",
    good["aspect_ratio"]
)


# ============================================================
# Exact resolutions
# ============================================================

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
    resolution_counts
    .head(20)
)


print(
    "\nUnique resolutions:",
    len(
        resolution_counts
    )
)


# ============================================================
# Expected 128x256 check
# ============================================================

expected = (
    (good["width"] == 128)
    &
    (good["height"] == 256)
)


print("\n" + "=" * 90)
print("128 x 256 CHECK")
print("=" * 90)

print(
    "128x256 images:",
    int(
        expected.sum()
    ),
    f"({expected.mean() * 100:.2f}%)"
)

print(
    "Other geometry:",
    int(
        (~expected).sum()
    ),
    f"({(~expected).mean() * 100:.2f}%)"
)


# ============================================================
# Geometry by scene
# ============================================================

print("\n" + "=" * 90)
print("GEOMETRY BY SCENE")
print("=" * 90)

for scene in sorted(
    good["scene"].unique()
):

    x = good[
        good["scene"] == scene
    ]

    print(
        f"\n{scene.upper()}"
    )

    print(
        "Images:",
        len(x)
    )

    print(
        "128x256:",
        int(
            (
                (x["width"] == 128)
                &
                (x["height"] == 256)
            ).sum()
        ),
        f"({(
            (
                (x['width'] == 128)
                &
                (x['height'] == 256)
            ).mean()
            * 100
        ):.2f}%)"
    )


# ============================================================
# Extreme geometry
# ============================================================

print("\n" + "=" * 90)
print("GEOMETRY EXTREMES")
print("=" * 90)


smallest = good.nsmallest(
    10,
    "area"
)

print("\nSmallest images:")

print(
    smallest[
        [
            "scene",
            "view",
            "global_person_id",
            "width",
            "height",
            "area",
            "filename",
        ]
    ].to_string(
        index=False
    )
)


narrowest = good.nsmallest(
    10,
    "aspect_ratio"
)

print("\nNarrowest images:")

print(
    narrowest[
        [
            "scene",
            "view",
            "global_person_id",
            "width",
            "height",
            "aspect_ratio",
            "filename",
        ]
    ].to_string(
        index=False
    )
)


widest = good.nlargest(
    10,
    "aspect_ratio"
)

print("\nWidest images:")

print(
    widest[
        [
            "scene",
            "view",
            "global_person_id",
            "width",
            "height",
            "aspect_ratio",
            "filename",
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(GEOMETRY_OUT)
print(CORRUPT_OUT)

print("\nDONE")
