from pathlib import Path
import pandas as pd
import numpy as np

manifest = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

df = pd.read_csv(manifest, parse_dates=["timestamp"])

train = df[df["split"] == "train"].copy()

print("=" * 80)
print("DEEPCHANGE TRAIN DIVERSITY")
print("=" * 80)

print("\nTrain images :", len(train))
print("Train IDs    :", train["pid"].nunique())
print("Tracklets    :", train["tracklet_id"].nunique())


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def describe(series, name):
    print(f"\n{name}")
    print("-" * len(name))

    print("Min    :", series.min())
    print("Mean   :", round(series.mean(), 2))
    print("Median :", round(series.median(), 2))
    print("Std    :", round(series.std(), 2))
    print("Max    :", series.max())

    print("25%    :", series.quantile(0.25))
    print("75%    :", series.quantile(0.75))
    print("95%    :", series.quantile(0.95))


# ---------------------------------------------------------
# Per-PID statistics
# ---------------------------------------------------------

images_per_pid = train.groupby("pid").size()

tracklets_per_pid = (
    train.groupby("pid")["tracklet_id"]
    .nunique()
)

cameras_per_pid = (
    train.groupby("pid")["camera_id"]
    .nunique()
)

dates_per_pid = (
    train.groupby("pid")["date"]
    .nunique()
)


describe(
    images_per_pid,
    "Images per PID"
)

describe(
    tracklets_per_pid,
    "Tracklets per PID"
)

describe(
    cameras_per_pid,
    "Cameras per PID"
)

describe(
    dates_per_pid,
    "Unique dates per PID"
)


# ---------------------------------------------------------
# Temporal span per PID
# ---------------------------------------------------------

pid_time = train.groupby("pid")["timestamp"].agg(
    first="min",
    last="max"
)

pid_time["span_days"] = (
    pid_time["last"] - pid_time["first"]
).dt.total_seconds() / 86400


describe(
    pid_time["span_days"],
    "Temporal span per PID (days)"
)


print("\nPID temporal-span thresholds:")

for days in [0, 1, 7, 30, 90, 180, 270]:

    count = int(
        (pid_time["span_days"] >= days).sum()
    )

    pct = count / len(pid_time) * 100

    print(
        f">= {days:3d} days : "
        f"{count:3d} / {len(pid_time)} "
        f"({pct:.2f}%)"
    )


# ---------------------------------------------------------
# Multi-camera / multi-date identities
# ---------------------------------------------------------

multi_camera = (cameras_per_pid >= 2)
multi_date = (dates_per_pid >= 2)
multi_tracklet = (tracklets_per_pid >= 2)

print("\nIdentity diversity:")

print(
    ">=2 tracklets :",
    int(multi_tracklet.sum()),
    f"({multi_tracklet.mean()*100:.2f}%)"
)

print(
    ">=2 cameras   :",
    int(multi_camera.sum()),
    f"({multi_camera.mean()*100:.2f}%)"
)

print(
    ">=2 dates     :",
    int(multi_date.sum()),
    f"({multi_date.mean()*100:.2f}%)"
)


# ---------------------------------------------------------
# Tracklet structure
# ---------------------------------------------------------

images_per_tracklet = (
    train.groupby("tracklet_id").size()
)

describe(
    images_per_tracklet,
    "Images per Tracklet"
)


# ---------------------------------------------------------
# Tracklet consistency
# A tracklet ideally belongs to one:
# PID, camera, date/time recording
# ---------------------------------------------------------

track_info = train.groupby("tracklet_id").agg(
    pids=("pid", "nunique"),
    cameras=("camera_id", "nunique"),
    dates=("date", "nunique"),
    timestamps=("timestamp", "nunique"),
)

print("\nTracklet consistency:")
print(
    "Tracklets with >1 PID       :",
    int((track_info["pids"] > 1).sum())
)

print(
    "Tracklets with >1 camera    :",
    int((track_info["cameras"] > 1).sum())
)

print(
    "Tracklets with >1 date      :",
    int((track_info["dates"] > 1).sum())
)

print(
    "Tracklets with >1 timestamp :",
    int((track_info["timestamps"] > 1).sum())
)


# ---------------------------------------------------------
# Camera 2 check
# ---------------------------------------------------------

print("\nCamera 2 in train:")
print(
    len(train[train["camera_id"] == 2])
)


# ---------------------------------------------------------
# Save per-PID table
# ---------------------------------------------------------

summary = pd.DataFrame({
    "images": images_per_pid,
    "tracklets": tracklets_per_pid,
    "cameras": cameras_per_pid,
    "dates": dates_per_pid,
}).join(
    pid_time["span_days"]
)

out = (
    Path.home()
    / "DG_ReID_Audit/reports/DeepChange"
    / "train_pid_diversity.csv"
)

summary.to_csv(out)

print("\nSaved:", out)
