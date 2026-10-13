from pathlib import Path
from itertools import combinations

import pandas as pd


MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/DeepChange"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    parse_dates=["timestamp"]
)

train = (
    df[df["split"] == "train"]
    .copy()
)


# ============================================================
# 1. Collapse image rows into one row per Tracklet
# ============================================================

tracklets = (
    train
    .groupby("tracklet_id")
    .agg(
        pid=("pid", "first"),
        camera_id=("camera_id", "first"),
        date=("date", "first"),
        timestamp=("timestamp", "first"),
        images=("filename", "size"),
        min_frame=("frame_id", "min"),
        max_frame=("frame_id", "max"),
        unique_frames=("frame_id", "nunique"),
    )
    .reset_index()
)


print("=" * 80)
print("TRACKLET-LEVEL STRUCTURE")
print("=" * 80)

print("\nTracklets:", len(tracklets))


# ============================================================
# 2. Check whether multiple tracklets share
#    the same PID + timestamp
# ============================================================

same_time = (
    tracklets
    .groupby(["pid", "timestamp"])
    .size()
)

multi_tracklet_same_time = (
    same_time[same_time > 1]
)


print("\nSame PID + same timestamp:")
print(
    "Groups with >1 tracklet:",
    len(multi_tracklet_same_time)
)

print(
    "Tracklets involved:",
    int(multi_tracklet_same_time.sum())
    if len(multi_tracklet_same_time)
    else 0
)

if len(multi_tracklet_same_time):
    print("\nLargest same-time groups:")
    print(
        multi_tracklet_same_time
        .sort_values(ascending=False)
        .head(10)
    )


# ============================================================
# 3. Build session-level table
#
# Session here means:
# same PID + same timestamp
# ============================================================

sessions = (
    tracklets
    .groupby(["pid", "timestamp"])
    .agg(
        tracklets=("tracklet_id", "size"),
        cameras=("camera_id", "nunique"),
        images=("images", "sum"),
        date=("date", "first"),
    )
    .reset_index()
)


sessions_per_pid = (
    sessions
    .groupby("pid")
    .size()
)


print("\n" + "=" * 80)
print("SESSIONS PER PID")
print("=" * 80)

print("Min    :", sessions_per_pid.min())
print("Mean   :", round(sessions_per_pid.mean(), 2))
print("Median :", sessions_per_pid.median())
print("Max    :", sessions_per_pid.max())

print(
    "\nPIDs with >=2 sessions:",
    int((sessions_per_pid >= 2).sum()),
    "/",
    len(sessions_per_pid),
    f"({(sessions_per_pid >= 2).mean()*100:.2f}%)"
)


# ============================================================
# 4. Consecutive temporal gaps
#
# We use UNIQUE timestamps here,
# not individual image frames.
# ============================================================

gap_rows = []

for pid, group in sessions.groupby("pid"):

    group = (
        group
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    times = group["timestamp"].tolist()

    for i in range(1, len(times)):

        previous = times[i - 1]
        current = times[i]

        gap_hours = (
            current - previous
        ).total_seconds() / 3600

        gap_rows.append({
            "pid": pid,
            "previous_timestamp": previous,
            "current_timestamp": current,
            "gap_hours": gap_hours,
            "gap_days": gap_hours / 24,
        })


gaps = pd.DataFrame(gap_rows)


print("\n" + "=" * 80)
print("CONSECUTIVE SESSION GAPS")
print("=" * 80)

if len(gaps):

    s = gaps["gap_days"]

    print("Gap count :", len(s))
    print("Min days  :", round(s.min(), 4))
    print("Mean days :", round(s.mean(), 2))
    print("Median    :", round(s.median(), 2))
    print("25%       :", round(s.quantile(0.25), 2))
    print("75%       :", round(s.quantile(0.75), 2))
    print("95%       :", round(s.quantile(0.95), 2))
    print("Max       :", round(s.max(), 2))


    buckets = {
        "< 1 hour":
            gaps["gap_hours"] < 1,

        "1 hour - <1 day":
            (gaps["gap_hours"] >= 1)
            & (gaps["gap_hours"] < 24),

        "1 - <7 days":
            (gaps["gap_days"] >= 1)
            & (gaps["gap_days"] < 7),

        "7 - <30 days":
            (gaps["gap_days"] >= 7)
            & (gaps["gap_days"] < 30),

        "30 - <90 days":
            (gaps["gap_days"] >= 30)
            & (gaps["gap_days"] < 90),

        "90 - <180 days":
            (gaps["gap_days"] >= 90)
            & (gaps["gap_days"] < 180),

        ">= 180 days":
            gaps["gap_days"] >= 180,
    }


    print("\nGap buckets:")

    for name, mask in buckets.items():

        count = int(mask.sum())
        pct = count / len(gaps) * 100

        print(
            f"{name:<18} : "
            f"{count:4d} "
            f"({pct:6.2f}%)"
        )


# ============================================================
# 5. Pairwise tracklet diversity
#
# For each PID compare all Tracklet pairs.
#
# Four categories:
#
# Same date + same camera
# Same date + different camera
# Different date + same camera
# Different date + different camera
# ============================================================

category_names = [
    "same_date_same_camera",
    "same_date_diff_camera",
    "diff_date_same_camera",
    "diff_date_diff_camera",
]

pair_counts = {
    name: 0
    for name in category_names
}

pid_has_category = {
    name: set()
    for name in category_names
}

pair_rows = []


for pid, group in tracklets.groupby("pid"):

    records = (
        group[
            [
                "tracklet_id",
                "camera_id",
                "date",
                "timestamp",
            ]
        ]
        .to_dict("records")
    )

    for a, b in combinations(records, 2):

        same_date = (
            a["date"] == b["date"]
        )

        same_camera = (
            a["camera_id"] == b["camera_id"]
        )

        if same_date and same_camera:
            category = "same_date_same_camera"

        elif same_date and not same_camera:
            category = "same_date_diff_camera"

        elif not same_date and same_camera:
            category = "diff_date_same_camera"

        else:
            category = "diff_date_diff_camera"


        gap_days = abs(
            (
                pd.Timestamp(b["timestamp"])
                - pd.Timestamp(a["timestamp"])
            ).total_seconds()
        ) / 86400


        pair_counts[category] += 1
        pid_has_category[category].add(pid)

        pair_rows.append({
            "pid": pid,
            "tracklet_a": a["tracklet_id"],
            "tracklet_b": b["tracklet_id"],
            "category": category,
            "gap_days": gap_days,
        })


pairs = pd.DataFrame(pair_rows)


print("\n" + "=" * 80)
print("TRACKLET PAIR TYPES")
print("=" * 80)

total_pairs = len(pairs)

for category in category_names:

    count = pair_counts[category]

    pct = (
        count / total_pairs * 100
        if total_pairs
        else 0
    )

    pid_count = len(
        pid_has_category[category]
    )

    pid_pct = (
        pid_count
        / train["pid"].nunique()
        * 100
    )

    print(
        f"\n{category}"
    )

    print(
        "Pairs:",
        count,
        f"({pct:.2f}%)"
    )

    print(
        "PIDs with >=1 such pair:",
        pid_count,
        f"({pid_pct:.2f}%)"
    )


# ============================================================
# 6. Long-term SAME-CAMERA pairs
#
# Particularly interesting because camera is fixed.
# ============================================================

same_cam_long = pairs[
    pairs["category"]
    == "diff_date_same_camera"
].copy()


print("\n" + "=" * 80)
print("SAME-CAMERA / DIFFERENT-DATE PAIRS")
print("=" * 80)

print(
    "Total pairs:",
    len(same_cam_long)
)

if len(same_cam_long):

    print(
        "Median gap days:",
        round(
            same_cam_long["gap_days"].median(),
            2
        )
    )

    print(
        "Mean gap days:",
        round(
            same_cam_long["gap_days"].mean(),
            2
        )
    )

    for days in [7, 30, 90, 180]:

        subset = (
            same_cam_long[
                same_cam_long["gap_days"] >= days
            ]
        )

        pid_count = (
            subset["pid"].nunique()
        )

        print(
            f">= {days:3d} days:"
            f" {len(subset):5d} pairs"
            f" | {pid_count:3d} PIDs"
        )


# ============================================================
# Save outputs
# ============================================================

tracklets.to_csv(
    REPORT_DIR
    / "train_tracklet_level.csv",
    index=False
)

sessions.to_csv(
    REPORT_DIR
    / "train_session_level.csv",
    index=False
)

gaps.to_csv(
    REPORT_DIR
    / "train_session_gaps.csv",
    index=False
)

pairs.to_csv(
    REPORT_DIR
    / "train_tracklet_pairs.csv",
    index=False
)


print("\nSaved:")
print(
    REPORT_DIR
    / "train_tracklet_level.csv"
)

print(
    REPORT_DIR
    / "train_session_level.csv"
)

print(
    REPORT_DIR
    / "train_session_gaps.csv"
)

print(
    REPORT_DIR
    / "train_tracklet_pairs.csv"
)
