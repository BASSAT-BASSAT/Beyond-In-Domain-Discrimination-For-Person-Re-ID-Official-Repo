from pathlib import Path
from collections import Counter
import json
import re

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

OUT_DIR = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_FILE = (
    OUT_DIR
    / "occluded_posetrack_reid_manifest.csv"
)


# ============================================================
# Confirmed filename schema
#
# PID_VideoID_PoseTrackImageID.jpg
#
# Example:
# 1659_000003_10000030011.jpg
# ============================================================

PATTERN = re.compile(
    r"^(?P<pid>\d+)_"
    r"(?P<video_id>\d+)_"
    r"(?P<posetrack_image_id>\d+)"
    r"\.jpg$",
    re.IGNORECASE,
)


rows = []
invalid_filenames = []
missing_physical = []
folder_video_mismatches = []
unexpected_roles = []


# ============================================================
# Process TRAIN and VAL annotation packages
# ============================================================

for source_partition in [
    "train",
    "val",
]:

    ann_file = (
        IMAGES
        / "anns"
        / f"{source_partition}_reid_crops_anns.json"
    )

    sampling_file = (
        ROOT
        / f"{source_partition}_dataset_sampling.json"
    )


    with open(
        ann_file,
        "r",
        encoding="utf-8",
    ) as f:

        anns = json.load(f)


    with open(
        sampling_file,
        "r",
        encoding="utf-8",
    ) as f:

        sampling = json.load(f)


    # --------------------------------------------------------
    # Iterate over annotation records
    #
    # Only records with a real crop path correspond to
    # physical ReID images.
    # --------------------------------------------------------

    for annotation_id, crop_path in (
        anns["reid_crop_path"].items()
    ):

        if not isinstance(crop_path, str):
            continue

        if not crop_path.strip():
            continue


        filename = Path(crop_path).name
        relative_folder = Path(crop_path).parent.name


        # ----------------------------------------------------
        # Parse confirmed filename schema
        # ----------------------------------------------------

        match = PATTERN.match(filename)

        if not match:

            invalid_filenames.append(
                {
                    "source_partition":
                        source_partition,

                    "annotation_id":
                        annotation_id,

                    "crop_path":
                        crop_path,
                }
            )

            continue


        pid = match.group("pid")
        video_id = match.group("video_id")
        posetrack_image_id = (
            match.group("posetrack_image_id")
        )


        # Keep leading zeros for video ID
        # because it is an identifier, not a quantity.
        logical_camid = video_id


        # ----------------------------------------------------
        # Sampling role
        # ----------------------------------------------------

        role = (
            sampling["split"]
            .get(annotation_id)
        )


        allowed_roles = {
            "train",
            "query",
            "gallery",
        }


        if role not in allowed_roles:

            unexpected_roles.append(
                {
                    "annotation_id":
                        annotation_id,

                    "role":
                        role,

                    "crop_path":
                        crop_path,
                }
            )


        # ----------------------------------------------------
        # Build expected physical image path
        # ----------------------------------------------------

        physical_path = (
            IMAGES
            / source_partition
            / crop_path
        )


        exists = physical_path.is_file()


        if not exists:

            missing_physical.append(
                {
                    "annotation_id":
                        annotation_id,

                    "expected_path":
                        str(physical_path),
                }
            )


        # ----------------------------------------------------
        # Folder must match Video ID
        # ----------------------------------------------------

        folder_matches_video = (
            relative_folder == video_id
        )


        if not folder_matches_video:

            folder_video_mismatches.append(
                {
                    "filename":
                        filename,

                    "folder":
                        relative_folder,

                    "video_id":
                        video_id,
                }
            )


        # ----------------------------------------------------
        # Pose metadata availability
        #
        # We DO NOT interpret the semantics of negative_kps
        # here. We only record whether the annotation exists.
        # Detailed pose analysis comes later.
        # ----------------------------------------------------

        negative_kps = (
            anns["negative_kps"]
            .get(annotation_id)
        )


        negative_kps_available = (
            negative_kps is not None
        )


        # ----------------------------------------------------
        # Add manifest row
        # ----------------------------------------------------

        rows.append(
            {
                "dataset":
                    "Occluded-PoseTrack-ReID",

                "source_partition":
                    source_partition,

                "split":
                    role,

                "filename":
                    filename,

                "path":
                    str(physical_path),

                "pid":
                    int(pid),

                # Keep zero-padding
                "video_id":
                    video_id,

                # Logical ReID camid used from video_id.
                # NOT a physical surveillance camera.
                "logical_camid":
                    video_id,

                "posetrack_image_id":
                    posetrack_image_id,

                "annotation_id":
                    str(annotation_id),

                "crop_width":
                    anns["reid_crop_width"]
                    .get(annotation_id),

                "crop_height":
                    anns["reid_crop_height"]
                    .get(annotation_id),

                "negative_kps_available":
                    negative_kps_available,

                "folder_matches_video":
                    folder_matches_video,
            }
        )


# ============================================================
# Build dataframe
# ============================================================

df = pd.DataFrame(rows)


# Preserve identifier strings exactly
df["video_id"] = (
    df["video_id"]
    .astype("string")
)

df["logical_camid"] = (
    df["logical_camid"]
    .astype("string")
)

df["posetrack_image_id"] = (
    df["posetrack_image_id"]
    .astype("string")
)

df["annotation_id"] = (
    df["annotation_id"]
    .astype("string")
)


# ============================================================
# Save manifest
# ============================================================

df.to_csv(
    OUT_FILE,
    index=False
)


# ============================================================
# Report
# ============================================================

print("=" * 90)
print("OCCLUDED-POSETRACK-REID MANIFEST")
print("=" * 90)

print("\nManifest rows:", len(df))

print("\nSPLIT COUNTS")
print("-" * 90)

print(
    df["split"]
    .value_counts()
    .sort_index()
)


print("\nSOURCE PARTITIONS")
print("-" * 90)

print(
    df["source_partition"]
    .value_counts()
    .sort_index()
)


print("\nIDENTITIES")
print("-" * 90)

for split in [
    "train",
    "query",
    "gallery",
]:

    x = df[
        df["split"] == split
    ]

    print(
        f"{split:<10}"
        f"images={len(x):6d} | "
        f"PIDs={x['pid'].nunique():5d} | "
        f"videos={x['video_id'].nunique():4d}"
    )


print("\nFILENAME / FILE INTEGRITY")
print("-" * 90)

print(
    "Invalid filenames          :",
    len(invalid_filenames)
)

print(
    "Missing physical images    :",
    len(missing_physical)
)

print(
    "Folder/video mismatches    :",
    len(folder_video_mismatches)
)

print(
    "Unexpected sampling roles  :",
    len(unexpected_roles)
)


print("\nANNOTATION INTEGRITY")
print("-" * 90)

print(
    "Unique annotation IDs      :",
    df["annotation_id"].nunique()
)

print(
    "Unique filenames           :",
    df["filename"].nunique()
)

print(
    "Duplicate annotation IDs   :",
    int(
        df["annotation_id"]
        .duplicated()
        .sum()
    )
)

print(
    "Duplicate filenames        :",
    int(
        df["filename"]
        .duplicated()
        .sum()
    )
)

print(
    "Null crop widths           :",
    int(
        df["crop_width"]
        .isna()
        .sum()
    )
)

print(
    "Null crop heights          :",
    int(
        df["crop_height"]
        .isna()
        .sum()
    )
)

print(
    "negative_kps unavailable   :",
    int(
        (~df["negative_kps_available"])
        .sum()
    )
)


# ============================================================
# Train / Val PID overlap
# ============================================================

train_ids = set(
    df.loc[
        df["split"] == "train",
        "pid"
    ]
)

eval_ids = set(
    df.loc[
        df["split"].isin(
            ["query", "gallery"]
        ),
        "pid"
    ]
)


print("\nTRAIN / EVAL PID SEPARATION")
print("-" * 90)

print(
    "Train PIDs:",
    len(train_ids)
)

print(
    "Eval PIDs :",
    len(eval_ids)
)

print(
    "Train ∩ Eval:",
    len(
        train_ids
        & eval_ids
    )
)


# ============================================================
# Query / Gallery coverage
# ============================================================

query_ids = set(
    df.loc[
        df["split"] == "query",
        "pid"
    ]
)

gallery_ids = set(
    df.loc[
        df["split"] == "gallery",
        "pid"
    ]
)


print("\nQUERY / GALLERY PID COVERAGE")
print("-" * 90)

print(
    "Query PIDs:",
    len(query_ids)
)

print(
    "Gallery PIDs:",
    len(gallery_ids)
)

print(
    "Query IDs missing from gallery:",
    len(
        query_ids
        - gallery_ids
    )
)

print(
    "Gallery-only IDs:",
    len(
        gallery_ids
        - query_ids
    )
)


print("\nSaved:")
print(OUT_FILE)
