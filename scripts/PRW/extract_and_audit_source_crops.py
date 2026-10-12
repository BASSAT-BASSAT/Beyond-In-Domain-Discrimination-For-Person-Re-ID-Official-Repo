from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd
from PIL import Image


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRW"
    / "prw_manifest.csv"
)

FLAGS = (
    BASE
    / "manifests/PRW"
    / "prw_audit_flags.csv"
)

OUT_DIR = (
    BASE
    / "datasets/PRW/derived/source_train_crops"
)

REPORT_DIR = (
    BASE
    / "reports/PRW"
)

MANIFEST_DIR = (
    BASE
    / "manifests/PRW"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

flags = pd.read_csv(
    FLAGS,
    low_memory=False
)


source = df[
    (df["frame_split"] == "train")
    &
    (df["identity_role"] == "train_id")
].copy()


# ============================================================
# HARD EXCLUSION KEYS
# ============================================================

excluded = flags[
    flags["recommended_training_exclusion"] == True
]


exclude_keys = set(
    zip(
        excluded["frame_stem"].astype(str),
        excluded["box_index"].astype(int),
        excluded["person_id"].astype(int),
    )
)


source["hard_exclude"] = source.apply(
    lambda r: (
        str(r["frame_stem"]),
        int(r["box_index"]),
        int(r["person_id"]),
    ) in exclude_keys,
    axis=1,
)


clean = source[
    ~source["hard_exclude"]
].copy()


print("=" * 100)
print("PRW CLEAN SOURCE CROP EXTRACTION")
print("=" * 100)

print(
    "Official source rows:",
    len(source)
)

print(
    "Hard exclusions:",
    int(source["hard_exclude"].sum())
)

print(
    "Clean rows:",
    len(clean)
)


if len(clean) != 14889:
    raise RuntimeError(
        f"Expected 14889 clean rows, got {len(clean)}"
    )


# ============================================================
# HASH FUNCTION
# Exact decoded RGB pixel hash
# ============================================================

def pixel_hash(image):

    arr = np.asarray(
        image.convert("RGB")
    )

    h = hashlib.sha256()

    h.update(
        f"{arr.shape[1]}x{arr.shape[0]}x3|".encode()
    )

    h.update(
        arr.tobytes()
    )

    return h.hexdigest()


# ============================================================
# EXTRACT
# ============================================================

records = []

current_frame = None
current_image = None


for n, (_, row) in enumerate(
    clean.sort_values(
        [
            "frame_stem",
            "box_index",
        ]
    ).iterrows(),
    start=1,
):

    frame_stem = str(
        row["frame_stem"]
    )

    frame_path = Path(
        row["frame_path"]
    )


    # ----------------------------------
    # Avoid reopening same frame repeatedly
    # ----------------------------------

    if frame_stem != current_frame:

        if current_image is not None:
            current_image.close()

        current_image = Image.open(
            frame_path
        ).convert(
            "RGB"
        )

        current_frame = frame_stem


    image_width, image_height = (
        current_image.size
    )


    # ----------------------------------
    # Deterministic integer crop:
    # floor top-left
    # ceil bottom-right
    # then clip to image
    # ----------------------------------

    raw_x1 = float(
        row["x"]
    )

    raw_y1 = float(
        row["y"]
    )

    raw_x2 = float(
        row["x2"]
    )

    raw_y2 = float(
        row["y2"]
    )


    x1 = max(
        0,
        int(
            np.floor(
                raw_x1
            )
        )
    )

    y1 = max(
        0,
        int(
            np.floor(
                raw_y1
            )
        )
    )

    x2 = min(
        image_width,
        int(
            np.ceil(
                raw_x2
            )
        )
    )

    y2 = min(
        image_height,
        int(
            np.ceil(
                raw_y2
            )
        )
    )


    if x2 <= x1 or y2 <= y1:

        raise RuntimeError(
            f"Invalid clipped crop: "
            f"{frame_stem}, "
            f"PID={row.person_id}, "
            f"box={row.box_index}"
        )


    crop = current_image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )


    phash = pixel_hash(
        crop
    )


    pid = int(
        row["person_id"]
    )

    camera = int(
        row["camera_id"]
    )

    segment = int(
        row["segment_id"]
    )

    frame_number = int(
        row["frame_number"]
    )

    box_index = int(
        row["box_index"]
    )


    # PID subfolder keeps future dataset loading simple
    pid_dir = (
        OUT_DIR
        / f"{pid:04d}"
    )

    pid_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    crop_filename = (
        f"{pid:04d}_"
        f"c{camera}s{segment}_"
        f"{frame_number:06d}_"
        f"b{box_index:02d}.jpg"
    )


    crop_path = (
        pid_dir
        / crop_filename
    )


    # Deterministic JPEG export.
    crop.save(
        crop_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )


    file_sha256 = hashlib.sha256(
        crop_path.read_bytes()
    ).hexdigest()


    records.append(
        {
            "dataset":
                "PRW",

            "person_id":
                pid,

            "camera_id":
                camera,

            "segment_id":
                segment,

            "frame_number":
                frame_number,

            "frame_stem":
                frame_stem,

            "box_index":
                box_index,

            "source_frame_path":
                str(
                    frame_path
                ),

            "crop_filename":
                crop_filename,

            "crop_path":
                str(
                    crop_path
                ),

            "crop_width":
                crop.width,

            "crop_height":
                crop.height,

            "raw_x":
                raw_x1,

            "raw_y":
                raw_y1,

            "raw_x2":
                raw_x2,

            "raw_y2":
                raw_y2,

            "clipped_x1":
                x1,

            "clipped_y1":
                y1,

            "clipped_x2":
                x2,

            "clipped_y2":
                y2,

            "was_clipped":
                not bool(
                    row[
                        "fully_inside_image"
                    ]
                ),

            "pixel_sha256":
                phash,

            "file_sha256":
                file_sha256,
        }
    )


    if (
        n % 2000 == 0
        or n == len(clean)
    ):

        print(
            f"processed "
            f"{n:,} / "
            f"{len(clean):,}"
        )


if current_image is not None:
    current_image.close()


crop_df = pd.DataFrame(
    records
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 100)
print("DERIVED SOURCE MANIFEST")
print("=" * 100)


print(
    "Crops:",
    len(
        crop_df
    )
)

print(
    "Unique identities:",
    crop_df[
        "person_id"
    ].nunique()
)

print(
    "Cameras:",
    crop_df[
        "camera_id"
    ].nunique()
)

print(
    "Clipped crops:",
    int(
        crop_df[
            "was_clipped"
        ].sum()
    )
)


print(
    "\nCrop width:"
)

print(
    crop_df[
        "crop_width"
    ]
    .describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .95,
            .99,
        ]
    )
    .round(2)
)


print(
    "\nCrop height:"
)

print(
    crop_df[
        "crop_height"
    ]
    .describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .95,
            .99,
        ]
    )
    .round(2)
)


# ============================================================
# PID COUNTS AFTER CLEANING
# ============================================================

pid_counts = (
    crop_df.groupby(
        "person_id"
    )
    .size()
)


print(
    "\nBoxes per PID after cleaning:"
)

print(
    pid_counts.describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .95,
            .99,
        ]
    )
    .round(2)
)


print(
    "\nIDs with 1 crop:",
    int(
        (
            pid_counts == 1
        ).sum()
    )
)

print(
    "IDs with < 4 crops:",
    int(
        (
            pid_counts < 4
        ).sum()
    )
)


# ============================================================
# PIXEL DUPLICATE GROUPS
# ============================================================

group_sizes = (
    crop_df.groupby(
        "pixel_sha256"
    )
    .size()
)


dup_hashes = set(
    group_sizes[
        group_sizes > 1
    ].index
)


dup = crop_df[
    crop_df[
        "pixel_sha256"
    ].isin(
        dup_hashes
    )
].copy()


print("\n" + "=" * 100)
print("EXACT PIXEL DUPLICATE AUDIT")
print("=" * 100)


print(
    "Duplicate groups:",
    len(
        dup_hashes
    )
)

print(
    "Images in duplicate groups:",
    len(
        dup
    )
)

print(
    "Extra pixel copies:",
    int(
        sum(
            group_sizes[
                group_sizes > 1
            ]
            - 1
        )
    )
)


cross_pid = 0
cross_camera = 0
cross_frame = 0
same_pid = 0


group_records = []


for hash_value, g in dup.groupby(
    "pixel_sha256"
):

    pids = sorted(
        g[
            "person_id"
        ]
        .unique()
        .tolist()
    )

    cameras = sorted(
        g[
            "camera_id"
        ]
        .unique()
        .tolist()
    )

    frames = sorted(
        g[
            "frame_stem"
        ]
        .unique()
        .tolist()
    )


    is_cross_pid = (
        len(
            pids
        )
        > 1
    )

    is_cross_camera = (
        len(
            cameras
        )
        > 1
    )

    is_cross_frame = (
        len(
            frames
        )
        > 1
    )


    if is_cross_pid:
        cross_pid += 1
    else:
        same_pid += 1

    if is_cross_camera:
        cross_camera += 1

    if is_cross_frame:
        cross_frame += 1


    group_records.append(
        {
            "pixel_sha256":
                hash_value,

            "group_size":
                len(g),

            "cross_pid":
                is_cross_pid,

            "cross_camera":
                is_cross_camera,

            "cross_frame":
                is_cross_frame,

            "person_ids":
                ",".join(
                    map(
                        str,
                        pids
                    )
                ),

            "camera_ids":
                ",".join(
                    map(
                        str,
                        cameras
                    )
                ),

            "frames":
                "|".join(
                    frames
                ),
        }
    )


groups_df = pd.DataFrame(
    group_records
)


print(
    "Cross-PID groups:",
    cross_pid
)

print(
    "Same-PID groups:",
    same_pid
)

print(
    "Cross-camera groups:",
    cross_camera
)

print(
    "Cross-frame groups:",
    cross_frame
)


if len(groups_df):

    print(
        "\nGroup size distribution:"
    )

    print(
        groups_df[
            "group_size"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\nCross-PID examples:"
    )

    examples = groups_df[
        groups_df[
            "cross_pid"
        ]
    ].head(20)

    if len(examples):
        print(
            examples.to_string(
                index=False
            )
        )
    else:
        print(
            "None"
        )


# ============================================================
# BYTE DUPLICATES OF DERIVED JPEG FILES
# ============================================================

file_group_sizes = (
    crop_df.groupby(
        "file_sha256"
    )
    .size()
)


file_dup_groups = (
    file_group_sizes
    > 1
)


print("\n" + "=" * 100)
print("DERIVED JPEG BYTE DUPLICATES")
print("=" * 100)

print(
    "Duplicate groups:",
    int(
        file_dup_groups.sum()
    )
)

print(
    "Extra byte-identical copies:",
    int(
        (
            file_group_sizes[
                file_group_sizes > 1
            ]
            - 1
        ).sum()
    )
)


# ============================================================
# SAVE
# ============================================================

derived_manifest = (
    MANIFEST_DIR
    / "prw_source_crops_manifest.csv"
)

dup_rows_out = (
    REPORT_DIR
    / "prw_source_pixel_duplicate_rows.csv"
)

dup_groups_out = (
    REPORT_DIR
    / "prw_source_pixel_duplicate_groups.csv"
)


crop_df.to_csv(
    derived_manifest,
    index=False
)

dup.to_csv(
    dup_rows_out,
    index=False
)

groups_df.to_csv(
    dup_groups_out,
    index=False
)


print("\nSaved:")
print(
    derived_manifest
)
print(
    dup_rows_out
)
print(
    dup_groups_out
)


print("\nDONE")
