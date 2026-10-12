from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


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
    / "occluded_posetrack_reid_negative_kps.csv"
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "annotation_id": str,
        "video_id": str,
        "posetrack_image_id": str,
    },
)


# ============================================================
# Load negative_kps from train + val annotation files
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

    for annotation_id, value in (
        anns["negative_kps"].items()
    ):

        negative_map[str(annotation_id)] = value


# ============================================================
# Analyze each selected crop
# ============================================================

rows = []

invalid_shapes = []


for _, row in df.iterrows():

    annotation_id = str(
        row["annotation_id"]
    )

    negative_kps = negative_map.get(
        annotation_id
    )

    if negative_kps is None:

        rows.append(
            {
                "annotation_id": annotation_id,
                "split": row["split"],
                "person_id": row["person_id"],
                "video_id": row["video_id"],
                "filename": row["filename"],
                "negative_people": 0,
                "negative_visible_keypoints": 0,
                "max_visible_kps_single_negative": 0,
            }
        )

        continue


    arr = np.asarray(
        negative_kps,
        dtype=float
    )


    # Expected shape:
    # (N_other_people, 17, 3)
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


    # visibility/confidence channel
    vis = (
        arr[:, :, 2] > 0
    )


    visible_per_person = (
        vis.sum(axis=1)
    )


    # Official generation code keeps skeletons
    # that have at least one visible keypoint.
    negative_people = int(
        (visible_per_person > 0)
        .sum()
    )


    total_visible_negative_kps = int(
        vis.sum()
    )


    max_visible_single = (
        int(
            visible_per_person.max()
        )
        if len(visible_per_person)
        else 0
    )


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

            "filename":
                row["filename"],

            "negative_people":
                negative_people,

            "negative_visible_keypoints":
                total_visible_negative_kps,

            "max_visible_kps_single_negative":
                max_visible_single,
        }
    )


res = pd.DataFrame(rows)

res.to_csv(
    OUT,
    index=False
)


print("=" * 90)
print("OCCLUDED-POSETRACK-REID NEGATIVE KEYPOINT AUDIT")
print("=" * 90)


print("\nRECORD INTEGRITY")
print("-" * 90)

print(
    "Manifest images:",
    len(df)
)

print(
    "Analyzed images:",
    len(res)
)

print(
    "Invalid negative_kps shapes:",
    len(invalid_shapes)
)


# ============================================================
# Overall statistics
# ============================================================

print("\nOVERALL NEGATIVE-PERSON STRUCTURE")
print("-" * 90)

print(
    "Images with no negative person:",
    int(
        (res["negative_people"] == 0)
        .sum()
    )
)

print(
    "Images with >=1 negative person:",
    int(
        (res["negative_people"] >= 1)
        .sum()
    )
)


print("\nNegative people per crop:")

dist = Counter(
    res["negative_people"]
    .tolist()
)

for n_people, n_images in sorted(
    dist.items()
):

    print(
        f"  {n_people:>2} negative person(s) "
        f"→ {n_images:>6} image(s)"
    )


# ============================================================
# Visible negative keypoints
# ============================================================

print("\nVISIBLE NEGATIVE KEYPOINTS")
print("-" * 90)

s = res[
    "negative_visible_keypoints"
]

print(
    "Min    :",
    int(s.min())
)

print(
    "Median :",
    float(s.median())
)

print(
    "Mean   :",
    round(
        float(s.mean()),
        2
    )
)

print(
    "P95    :",
    float(
        s.quantile(0.95)
    )
)

print(
    "P99    :",
    float(
        s.quantile(0.99)
    )
)

print(
    "Max    :",
    int(s.max())
)


# ============================================================
# By split
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

    with_negative = (
        x["negative_people"] >= 1
    )

    print(
        f"\n{split_name.upper()}"
    )

    print(
        "Images:",
        len(x)
    )

    print(
        "With >=1 negative person:",
        int(
            with_negative.sum()
        ),
        f"({with_negative.mean() * 100:.2f}%)"
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


# ============================================================
# Most crowded examples
# ============================================================

print("\nMOST NEGATIVE PEOPLE")
print("-" * 90)

print(
    res.nlargest(
        20,
        [
            "negative_people",
            "negative_visible_keypoints",
        ]
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


print("\nMOST VISIBLE NEGATIVE KEYPOINTS")
print("-" * 90)

print(
    res.nlargest(
        20,
        "negative_visible_keypoints"
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


if invalid_shapes:

    print("\nINVALID SHAPE EXAMPLES")
    print("-" * 90)

    for x in invalid_shapes[:20]:
        print(x)


print("\nSaved:")
print(OUT)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
