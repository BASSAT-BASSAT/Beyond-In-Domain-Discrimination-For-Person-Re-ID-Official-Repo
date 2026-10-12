from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
from PIL import Image


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
)

OUT_DIR = (
    BASE
    / "manifests/RPIfield"
)

GEOMETRY_OUT = (
    OUT_DIR
    / "rpifield_geometry.csv"
)

CORRUPT_OUT = (
    OUT_DIR
    / "rpifield_corrupt_images.csv"
)


def inspect_image(item):

    idx, path = item

    try:

        with Image.open(path) as img:

            fmt = img.format
            mode = img.mode
            width, height = img.size

            # Force full decoding
            img.load()

        return {
            "index": idx,
            "readable": True,
            "format": fmt,
            "mode": mode,
            "width": width,
            "height": height,
            "aspect_ratio": (
                width / height
                if height > 0
                else None
            ),
            "error": None,
        }

    except Exception as e:

        return {
            "index": idx,
            "readable": False,
            "format": None,
            "mode": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "error": repr(e),
        }


df = pd.read_csv(
    MANIFEST,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
        "identity_type": str,
    },
    low_memory=False,
).reset_index(
    drop=True
)


print("=" * 90)
print("RPIFIELD IMAGE INTEGRITY / GEOMETRY AUDIT")
print("=" * 90)

print(
    "\nImages:",
    len(df)
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


items = [
    (
        idx,
        path,
    )
    for idx, path in enumerate(
        df["path"]
    )
]


results = []


print("\nDecoding images...")


with Pool(
    processes=workers
) as pool:

    for n, result in enumerate(
        pool.imap_unordered(
            inspect_image,
            items,
            chunksize=250,
        ),
        start=1,
    ):

        results.append(
            result
        )

        if (
            n % 25000 == 0
            or n == len(items)
        ):

            print(
                f"processed "
                f"{n:,} / {len(items):,}"
            )


res = pd.DataFrame(
    results
).sort_values(
    "index"
).reset_index(
    drop=True
)


geo = pd.concat(
    [
        df[
            [
                "camera_id",
                "person_id",
                "identity_type",
                "appearance_label",
                "sequence_id",
                "frame_id",
                "path",
            ]
        ].reset_index(
            drop=True
        ),
        res.drop(
            columns=[
                "index"
            ]
        ),
    ],
    axis=1,
)


geo.to_csv(
    GEOMETRY_OUT,
    index=False
)


corrupt = geo[
    ~geo["readable"]
].copy()

corrupt.to_csv(
    CORRUPT_OUT,
    index=False
)


print("\n" + "=" * 90)
print("IMAGE INTEGRITY")
print("=" * 90)

print(
    "Manifest images  :",
    len(geo)
)

print(
    "Readable images  :",
    int(
        geo[
            "readable"
        ].sum()
    )
)

print(
    "Corrupted images :",
    len(corrupt)
)


valid = geo[
    geo["readable"]
].copy()


print("\nFORMATS")
print("-" * 90)

print(
    valid[
        "format"
    ].value_counts()
)


print("\nMODES")
print("-" * 90)

print(
    valid[
        "mode"
    ].value_counts()
)


print("\n" + "=" * 90)
print("OVERALL GEOMETRY")
print("=" * 90)


for column, title in [
    (
        "width",
        "WIDTH",
    ),
    (
        "height",
        "HEIGHT",
    ),
    (
        "aspect_ratio",
        "ASPECT RATIO",
    ),
]:

    print(
        f"\n{title}"
    )

    print(
        valid[
            column
        ]
        .describe(
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
        .round(3)
    )


print("\n" + "=" * 90)
print("MOST COMMON RESOLUTIONS")
print("=" * 90)


resolution_counts = (
    valid.groupby(
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
    resolution_counts.head(
        30
    )
)


print(
    "\nUnique resolutions:",
    len(
        resolution_counts
    )
)


print("\n" + "=" * 90)
print("ORIENTATION")
print("=" * 90)


portrait = (
    valid["height"]
    > valid["width"]
)

square = (
    valid["height"]
    == valid["width"]
)

landscape = (
    valid["width"]
    > valid["height"]
)


print(
    "Portrait :",
    int(
        portrait.sum()
    ),
    f"({portrait.mean() * 100:.2f}%)"
)

print(
    "Square   :",
    int(
        square.sum()
    ),
    f"({square.mean() * 100:.2f}%)"
)

print(
    "Landscape:",
    int(
        landscape.sum()
    ),
    f"({landscape.mean() * 100:.2f}%)"
)


print("\n" + "=" * 90)
print("BY CAMERA")
print("=" * 90)


for camera_id in sorted(
    valid[
        "camera_id"
    ].unique()
):

    x = valid[
        valid[
            "camera_id"
        ]
        == camera_id
    ]

    resolutions = (
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
    )


    print(
        f"\nCam_{camera_id}"
    )

    print(
        "Images:",
        len(x)
    )

    print(
        "Unique resolutions:",
        len(
            resolutions
        )
    )

    print(
        "Width median:",
        float(
            x[
                "width"
            ].median()
        )
    )

    print(
        "Height median:",
        float(
            x[
                "height"
            ].median()
        )
    )

    print(
        "Aspect median:",
        round(
            float(
                x[
                    "aspect_ratio"
                ].median()
            ),
            3
        )
    )

    print(
        "Most common resolutions:"
    )

    print(
        resolutions.head(
            5
        )
    )


print("\n" + "=" * 90)
print("KNOWN ACTORS VS DISTRACTORS")
print("=" * 90)


for identity_type in [
    "known_actor",
    "distractor",
]:

    x = valid[
        valid[
            "identity_type"
        ]
        == identity_type
    ]


    print(
        f"\n{identity_type}"
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
        "Median width:",
        float(
            x[
                "width"
            ].median()
        )
    )

    print(
        "Median height:",
        float(
            x[
                "height"
            ].median()
        )
    )


print("\nSaved:")
print(GEOMETRY_OUT)
print(CORRUPT_OUT)

print("\nDONE")
