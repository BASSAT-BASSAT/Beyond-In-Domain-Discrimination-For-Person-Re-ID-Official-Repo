from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/LaST/last_manifest.csv"
)

df = pd.read_csv(MANIFEST, low_memory=False)

train = df[df["split"] == "train"].copy()

print("=" * 75)
print("LaST TRAIN — TEMPORAL REDUNDANCY AUDIT")
print("=" * 75)

# --------------------------------------------------
# Images per PID × clothes state
# --------------------------------------------------

groups = (
    train.groupby(["pid", "clothes_id"])
    .size()
)

print("\nImages per (PID, clothes ID):")
print(groups.describe().round(2))

print("\nPercentiles:")
print(
    groups.quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).round(2)
)

# --------------------------------------------------
# Duplicate frame IDs inside same PID/video
# --------------------------------------------------

dupes = train.duplicated(
    subset=["pid", "video_id", "frame_id"],
    keep=False
)

print(
    "\nImages sharing same PID/video/frame:",
    int(dupes.sum())
)

duplicate_groups = (
    train.groupby(
        ["pid", "video_id", "frame_id"]
    )
    .size()
)

print(
    "Frame groups with >1 bbox:",
    int((duplicate_groups > 1).sum())
)

# --------------------------------------------------
# Consecutive frame gaps
# within PID × clothes
# --------------------------------------------------

all_gaps = []

for (_, _), g in train.groupby(
    ["pid", "clothes_id"]
):

    frames = np.sort(
        g["frame_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(frames) >= 2:
        gaps = np.diff(frames)
        all_gaps.extend(gaps.tolist())

gaps = pd.Series(all_gaps, dtype="int64")

print("\nFrame-gap statistics:")

if len(gaps):

    print(gaps.describe().round(2))

    print("\nFrame-gap percentiles:")
    print(
        gaps.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        ).round(2)
    )

    for threshold in [
        1, 2, 3, 5, 10,
        20, 30, 60, 100
    ]:

        pct = (
            (gaps <= threshold).mean()
            * 100
        )

        print(
            f"Gaps <= {threshold:3d}: "
            f"{pct:6.2f}%"
        )

    print("\nMost common frame gaps:")
    print(
        gaps.value_counts()
        .head(20)
        .sort_index()
    )

# --------------------------------------------------
# Image ID uniqueness
# --------------------------------------------------

image_id_dupes = train.duplicated(
    subset=["pid", "image_id"],
    keep=False
)

print(
    "\nDuplicate image_id within PID:",
    int(image_id_dupes.sum())
)

print("\nDone.")
