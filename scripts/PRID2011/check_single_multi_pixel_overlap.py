from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/PRID2011/extracted/prid_2011"
)

PREV = (
    BASE
    / "manifests/PRID2011"
    / "single_multi_overlap.csv"
)

OUT = (
    BASE
    / "manifests/PRID2011"
    / "single_multi_pixel_overlap.csv"
)


def pixel_hash(path):

    with Image.open(path) as img:

        img = img.convert("RGB")

        payload = (
            str(img.size).encode()
            + img.tobytes()
        )

        return hashlib.sha256(
            payload
        ).hexdigest()


prev = pd.read_csv(PREV)

unmatched = prev[
    prev["matches_in_multi"] == 0
].copy()


print("=" * 90)
print("PRID2011 SINGLE vs MULTI PIXEL-LEVEL OVERLAP")
print("=" * 90)

print(
    "\nByte-unmatched single-shot images:",
    len(unmatched)
)


rows = []


for n, (_, row) in enumerate(
    unmatched.iterrows(),
    start=1,
):

    camera = row["camera"]
    person = row["person"]

    single_path = Path(
        row["single_path"]
    )

    trajectory_dir = (
        ROOT
        / "multi_shot"
        / camera
        / person
    )


    single_pixel_hash = pixel_hash(
        single_path
    )


    matches = []


    for multi_path in sorted(
        trajectory_dir.glob("*.png")
    ):

        if pixel_hash(
            multi_path
        ) == single_pixel_hash:

            matches.append(
                multi_path.name
            )


    rows.append(
        {
            "camera":
                camera,

            "person":
                person,

            "single_path":
                str(single_path),

            "pixel_matches":
                len(matches),

            "matched_files":
                ";".join(matches),
        }
    )


    if (
        n % 100 == 0
        or n == len(unmatched)
    ):

        print(
            f"processed {n} / {len(unmatched)}"
        )


result = pd.DataFrame(
    rows
)


print("\n" + "=" * 90)
print("RESULT")
print("=" * 90)

print(
    "Pixel-identical match:",
    int(
        (
            result["pixel_matches"] > 0
        ).sum()
    )
)

print(
    "Still unmatched:",
    int(
        (
            result["pixel_matches"] == 0
        ).sum()
    )
)

print(
    "More than one pixel match:",
    int(
        (
            result["pixel_matches"] > 1
        ).sum()
    )
)


print("\nBY CAMERA")
print("-" * 90)

for camera in [
    "cam_a",
    "cam_b",
]:

    x = result[
        result["camera"] == camera
    ]

    print(
        f"\n{camera}"
    )

    print(
        "Checked:",
        len(x)
    )

    print(
        "Pixel matched:",
        int(
            (
                x["pixel_matches"] > 0
            ).sum()
        )
    )

    print(
        "Still unmatched:",
        int(
            (
                x["pixel_matches"] == 0
            ).sum()
        )
    )


still_unmatched = result[
    result[
        "pixel_matches"
    ] == 0
]


if len(still_unmatched) > 0:

    print("\nSTILL UNMATCHED EXAMPLES")
    print("-" * 90)

    print(
        still_unmatched.head(20)
        .to_string(
            index=False
        )
    )


result.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)

print("\nDONE")
