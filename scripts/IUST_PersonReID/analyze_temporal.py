from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/IUST_PersonReID/iust_manifest_raw.csv"
)

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/IUST_PersonReID"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

# Training أهم جزء بالنسبة للـ fine-tuning
train = df[df["split"] == "train"].copy()

print("=" * 70)
print("IUST TEMPORAL / FRAME REDUNDANCY AUDIT")
print("=" * 70)

# --------------------------------------------------
# 1. PID-camera groups
# --------------------------------------------------

group_sizes = (
    train.groupby(["pid", "camera"])
    .size()
)

print("\nPID-camera groups:")
print("Number of groups:", len(group_sizes))

print("\nImages per PID-camera group:")
print(group_sizes.describe().round(2))

print("\nPercentiles:")
print(
    group_sizes.quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).round(2)
)

# --------------------------------------------------
# 2. Frame gaps inside each PID-camera
# --------------------------------------------------

gap_rows = []
group_rows = []

for (pid, camera), g in train.groupby(["pid", "camera"]):

    g = g.sort_values("frame")

    frames = g["frame"].to_numpy()

    # duplicate frame IDs داخل نفس PID-camera
    duplicate_frames = len(frames) - len(np.unique(frames))

    if len(frames) >= 2:
        gaps = np.diff(frames)

        for gap in gaps:
            gap_rows.append({
                "pid": pid,
                "camera": camera,
                "gap": int(gap),
            })

        group_rows.append({
            "pid": pid,
            "camera": camera,
            "images": len(frames),
            "min_frame": int(frames.min()),
            "max_frame": int(frames.max()),
            "frame_span": int(frames.max() - frames.min()),
            "median_gap": float(np.median(gaps)),
            "min_gap": int(gaps.min()),
            "max_gap": int(gaps.max()),
            "duplicate_frames": duplicate_frames,
        })

    else:
        group_rows.append({
            "pid": pid,
            "camera": camera,
            "images": len(frames),
            "min_frame": int(frames[0]),
            "max_frame": int(frames[0]),
            "frame_span": 0,
            "median_gap": np.nan,
            "min_gap": np.nan,
            "max_gap": np.nan,
            "duplicate_frames": duplicate_frames,
        })

gaps_df = pd.DataFrame(gap_rows)
groups_df = pd.DataFrame(group_rows)

print("\n" + "=" * 70)
print("FRAME GAPS")
print("=" * 70)

print("\nTotal consecutive frame gaps:", len(gaps_df))

print("\nGap statistics:")
print(
    gaps_df["gap"]
    .describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
    .round(2)
)

print("\nMost common frame gaps:")
print(
    gaps_df["gap"]
    .value_counts()
    .head(20)
)

# --------------------------------------------------
# 3. How many gaps are very close?
# --------------------------------------------------

thresholds = [1, 3, 6, 12, 30, 60, 100, 300]

print("\nPercentage of consecutive gaps <= threshold:")

for t in thresholds:

    pct = (
        (gaps_df["gap"] <= t).mean()
        * 100
    )

    print(
        f"gap <= {t:>3}: {pct:6.2f}%"
    )

# --------------------------------------------------
# 4. Duplicate frame IDs
# --------------------------------------------------

print("\nDuplicate frame IDs within same PID-camera:")
print(
    int(groups_df["duplicate_frames"].sum())
)

print(
    "Groups containing duplicate frame IDs:",
    int((groups_df["duplicate_frames"] > 0).sum())
)

# --------------------------------------------------
# 5. Groups with smallest median gaps
# --------------------------------------------------

print("\nGroups with smallest median frame gap:")

print(
    groups_df
    .dropna()
    .sort_values(
        ["median_gap", "images"],
        ascending=[True, False]
    )
    .head(15)
    .to_string(index=False)
)

# --------------------------------------------------
# Save
# --------------------------------------------------

gaps_df.to_csv(
    REPORT_DIR / "train_frame_gaps.csv",
    index=False
)

groups_df.to_csv(
    REPORT_DIR / "train_pid_camera_temporal.csv",
    index=False
)

print("\nSaved:")
print(REPORT_DIR / "train_frame_gaps.csv")
print(REPORT_DIR / "train_pid_camera_temporal.csv")
