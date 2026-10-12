from pathlib import Path
import pandas as pd

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/LaST/last_manifest.csv"
)

df = pd.read_csv(MANIFEST, low_memory=False)
train = df[df["split"] == "train"].copy()

print("=" * 80)
print("SAME PID / VIDEO / FRAME GROUPS")
print("=" * 80)

same_frame = (
    train.groupby(
        ["pid", "video_id", "frame_id"]
    )
    .filter(lambda g: len(g) > 1)
    .sort_values(
        ["pid", "video_id", "frame_id", "bbox_id"]
    )
)

print(
    same_frame[
        [
            "pid",
            "image_id",
            "video_id",
            "frame_id",
            "bbox_id",
            "clothes_id",
            "filename",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("DUPLICATE IMAGE_ID WITHIN PID")
print("=" * 80)

dup_img = train[
    train.duplicated(
        subset=["pid", "image_id"],
        keep=False
    )
].sort_values(["pid", "image_id"])

print(
    dup_img[
        [
            "pid",
            "image_id",
            "video_id",
            "frame_id",
            "bbox_id",
            "clothes_id",
            "filename",
        ]
    ].to_string(index=False)
)
