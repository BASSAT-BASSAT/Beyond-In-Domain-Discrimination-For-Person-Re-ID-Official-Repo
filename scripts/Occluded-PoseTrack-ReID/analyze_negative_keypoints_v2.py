from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/Occluded-PoseTrack-ReID"
    / "extracted/occluded_posetrack_reid"
)

IMAGES = ROOT / "images"

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_manifest_v2.csv"
)

OUT = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_negative_kps_v2.csv"
)


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(
    MANIFEST,
    dtype={
        "annotation_id": str,
        "video_id": str,
        "logical_camid": str,
        "posetrack_image_id": str,
    },
)


# ============================================================
# Load negative_kps from train + val annotation JSONs
# ============================================================

negative_map = {}

for source_partition in ["train", "val"]:

    ann_file = (
        IMAGES
        / "anns"
        / f"{source_partition}_reid_crops_anns.json"
    )

    with open(
        ann_file,
        "r",
        encoding="utf-8",
    ) as f:

        anns = json.load(f)

    for annotation_id, negative_kps in (
        anns["negative_kps"].items()
    ):

        negative_map[
            str(annotation_id)
        ] = negative_kps


# ============================================================
# Analyze selected ReID crops
# ============================================================

rows = []

missing_negative_kps = []
invalid_shapes = []


for _, row in df.iterrows():

    annotation_id = str(
        row["annotation_id"]
    )

    negative_kps = negative_map.get(
        annotation_id
    )


    # --------------------------------------------------------
    # Missing annotation
    # --------------------------------------------------------

    if negative_kps is None:

        missing_negative_kps.append(
            {
                "annotation_id":
                    annotation_id,

                "filename":
                    row["filename"],
            }
        )

        continue


    # --------------------------------------------------------
    # Empty list is VALID.
    #
    # [] means there are no negative people associated
    # with this selected ReID crop.
    # --------------------------------------------------------

    if (
        isinstance(
            negative_kps,
            list
        )
        and len(negative_kps) == 0
    ):

        rows.append(
            {
                "annotation_id":
                    annotation_id,

                "split":
                    row["split"],

                "person_id":
                    row["person_id"],

                "video_id":
                    row["video_id"],

                "posetrack_image_id":
                    row[
                        "posetrack_image_id"
                    ],

                "filename":
                    row["filename"],

                "negative_people":
                    0,

                "negative_visible_keypoints":
                    0,

                "max_visible_kps_single_negative":
                    0,

                "has_negative_person":
                    False,
            }
        )

        continue


    # --------------------------------------------------------
    # Convert non-empty negative_kps to numpy
    #
    # Expected:
    # (N_other_people, 17, 3)
    # --------------------------------------------------------

    try:

        arr = np.asarray(
            negative_kps,
            dtype=float,
        )

    except Exception as e:

        invalid_shapes.append(
            {
                "annotation_id":
                    annotation_id,

                "shape":
                    "conversion_error",

                "error":
                    str(e),

                "filename":
                    row["filename"],
            }
        )

        continue


    if (
        arr.ndim != 3
        or arr.shape[1:] != (17, 3)
    ):

        invalid_shapes.append(
            {
                "annotation_id":
                    annotation_id,

                "shape":
                    arr.shape,

                "filename":
                    row["filename"],
            }
        )

        continue


    # ========================================================
    # Visibility channel
    #
    # Each keypoint:
    # [x, y, visibility/confidence]
    #
    # We count visibility > 0 as visible.
    # ========================================================

    visible_mask = (
        arr[:, :, 2] > 0
    )


    # Visible keypoints for each negative person
    visible_per_negative_person = (
        visible_mask.sum(
            axis=1
        )
    )


    # A negative person counts only if they have
    # at least one visible keypoint.
    negative_people = int(
        (
            visible_per_negative_person
            > 0
        ).sum()
    )


    total_visible_negative_kps = int(
        visible_mask.sum()
    )


    if (
        len(
            visible_per_negative_person
        )
        > 0
    ):

        max_visible_single_negative = int(
            visible_per_negative_person.max()
        )

    else:

        max_visible_single_negative = 0


    rows.append(
        {
            "annotation_id":
                annotation_id,

            "split":
                row["split"],

            "person_id":
                row["person_id"],

            "video_id":
                row["video_id"],

            "posetrack_image_id":
                row[
                    "posetrack_image_id"
                ],

            "filename":
                row["filename"],

            "negative_people":
                negative_people,

            "negative_visible_keypoints":
                total_visible_negative_kps,

            "max_visible_kps_single_negative":
                max_visible_single_negative,

            "has_negative_person":
                negative_people > 0,
        }
    )


# ============================================================
# Build dataframe
# ============================================================

res = pd.DataFrame(
    rows
)


# ============================================================
# Save
# ============================================================

res.to_csv(
    OUT,
    index=False,
)


# ============================================================
# Report
# ============================================================

print("=" * 90)
print("OCCLUDED-POSETRACK-REID NEGATIVE KEYPOINT AUDIT V2")
print("=" * 90)


# ============================================================
# 1. Integrity
# ============================================================

print("\nRECORD INTEGRITY")
print("-" * 90)

print(
    "Manifest images              :",
    len(df)
)

print(
    "Analyzed images              :",
    len(res)
)

print(
    "Missing negative_kps         :",
    len(
        missing_negative_kps
    )
)

print(
    "Invalid non-empty kps shapes :",
    len(
        invalid_shapes
    )
)


# ============================================================
# 2. Overall negative-person structure
# ============================================================

print("\nOVERALL NEGATIVE-PERSON STRUCTURE")
print("-" * 90)

no_negative = int(
    (
        res[
            "negative_people"
        ]
        == 0
    ).sum()
)

with_negative = int(
    (
        res[
            "negative_people"
        ]
        >= 1
    ).sum()
)


print(
    "Images with no negative person:",
    no_negative,
    f"({no_negative / len(res) * 100:.2f}%)"
)

print(
    "Images with >=1 negative person:",
    with_negative,
    f"({with_negative / len(res) * 100:.2f}%)"
)


print("\nNegative people per crop:")

negative_people_dist = Counter(
    res[
        "negative_people"
    ].tolist()
)

for (
    n_people,
    n_images
) in sorted(
    negative_people_dist.items()
):

    print(
        f"  {n_people:>2} negative person(s) "
        f"→ {n_images:>6} image(s)"
    )


# ============================================================
# 3. Visible negative keypoints
# ============================================================

print("\nVISIBLE NEGATIVE KEYPOINTS")
print("-" * 90)

visible_kps = res[
    "negative_visible_keypoints"
]


print(
    "Min    :",
    int(
        visible_kps.min()
    )
)

print(
    "Median :",
    float(
        visible_kps.median()
    )
)

print(
    "Mean   :",
    round(
        float(
            visible_kps.mean()
        ),
        2
    )
)

print(
    "P95    :",
    float(
        visible_kps.quantile(
            0.95
        )
    )
)

print(
    "P99    :",
    float(
        visible_kps.quantile(
            0.99
        )
    )
)

print(
    "Max    :",
    int(
        visible_kps.max()
    )
)


# ============================================================
# 4. By split
# ============================================================

print("\nBY SPLIT")
print("-" * 90)

for split_name in [
    "train",
    "query",
    "gallery",
]:

    x = res[
        res["split"]
        == split_name
    ]

    has_negative = (
        x[
            "negative_people"
        ]
        >= 1
    )

    no_negative_split = (
        x[
            "negative_people"
        ]
        == 0
    )


    print(
        f"\n{split_name.upper()}"
    )

    print(
        "Images:",
        len(x)
    )

    print(
        "No negative person:",
        int(
            no_negative_split.sum()
        ),
        f"({no_negative_split.mean() * 100:.2f}%)"
    )

    print(
        "With >=1 negative person:",
        int(
            has_negative.sum()
        ),
        f"({has_negative.mean() * 100:.2f}%)"
    )

    print(
        "Median negative people:",
        float(
            x[
                "negative_people"
            ].median()
        )
    )

    print(
        "Mean negative people:",
        round(
            float(
                x[
                    "negative_people"
                ].mean()
            ),
            2
        )
    )

    print(
        "Median visible negative kps:",
        float(
            x[
                "negative_visible_keypoints"
            ].median()
        )
    )

    print(
        "Mean visible negative kps:",
        round(
            float(
                x[
                    "negative_visible_keypoints"
                ].mean()
            ),
            2
        )
    )

    print(
        "P95 visible negative kps:",
        float(
            x[
                "negative_visible_keypoints"
            ].quantile(
                0.95
            )
        )
    )


# ============================================================
# 5. Analyze only crowded crops
# ============================================================

crowded = res[
    res[
        "negative_people"
    ]
    >= 1
].copy()


print("\nCROWDED CROPS ONLY")
print("-" * 90)

print(
    "Crowded images:",
    len(crowded)
)

if len(crowded) > 0:

    print(
        "Median negative people:",
        float(
            crowded[
                "negative_people"
            ].median()
        )
    )

    print(
        "Mean negative people:",
        round(
            float(
                crowded[
                    "negative_people"
                ].mean()
            ),
            2
        )
    )

    print(
        "Median visible negative kps:",
        float(
            crowded[
                "negative_visible_keypoints"
            ].median()
        )
    )

    print(
        "Mean visible negative kps:",
        round(
            float(
                crowded[
                    "negative_visible_keypoints"
                ].mean()
            ),
            2
        )
    )


# ============================================================
# 6. Most crowded examples
# ============================================================

print("\nMOST NEGATIVE PEOPLE")
print("-" * 90)

print(
    res.nlargest(
        20,
        [
            "negative_people",
            "negative_visible_keypoints",
        ],
    )[
        [
            "split",
            "person_id",
            "video_id",
            "negative_people",
            "negative_visible_keypoints",
            "filename",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 7. Highest number of visible negative keypoints
# ============================================================

print("\nMOST VISIBLE NEGATIVE KEYPOINTS")
print("-" * 90)

print(
    res.nlargest(
        20,
        "negative_visible_keypoints",
    )[
        [
            "split",
            "person_id",
            "video_id",
            "negative_people",
            "negative_visible_keypoints",
            "filename",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 8. Examples with zero negative people
# ============================================================

zero_negative = res[
    res[
        "negative_people"
    ]
    == 0
]


print("\nNO-NEGATIVE-PERSON EXAMPLES")
print("-" * 90)

print(
    zero_negative.head(
        20
    )[
        [
            "split",
            "person_id",
            "video_id",
            "filename",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 9. Error examples
# ============================================================

if missing_negative_kps:

    print("\nMISSING NEGATIVE_KPS EXAMPLES")
    print("-" * 90)

    for item in (
        missing_negative_kps[:20]
    ):

        print(
            item
        )


if invalid_shapes:

    print("\nINVALID NON-EMPTY SHAPE EXAMPLES")
    print("-" * 90)

    for item in (
        invalid_shapes[:20]
    ):

        print(
            item
        )


# ============================================================
# Final
# ============================================================

print("\nSaved:")
print(OUT)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)