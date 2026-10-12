from pathlib import Path
import pandas as pd

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/LaST/last_manifest.csv"
)

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/LaST"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

# Train only
train = df[df["split"] == "train"].copy()

print("=" * 75)
print("LaST TRAIN — IDENTITY / VIDEO / CLOTHES AUDIT")
print("=" * 75)

# --------------------------------------------------
# 1. Images per identity
# --------------------------------------------------

images_per_pid = train.groupby("pid").size()

print("\nImages per PID:")
print(f"IDs    : {len(images_per_pid)}")
print(f"Min    : {images_per_pid.min()}")
print(f"Max    : {images_per_pid.max()}")
print(f"Mean   : {images_per_pid.mean():.2f}")
print(f"Median : {images_per_pid.median():.2f}")
print(f"Std    : {images_per_pid.std():.2f}")

print("\nImages-per-PID percentiles:")
print(
    images_per_pid.quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).round(2)
)

# --------------------------------------------------
# 2. Videos per identity
# --------------------------------------------------

videos_per_pid = (
    train.groupby("pid")["video_id"]
    .nunique()
)

print("\nVideos per PID:")
print(videos_per_pid.value_counts().sort_index())

print("\nVideo coverage statistics:")
print(
    videos_per_pid.describe().round(2)
)

print(
    "\nIDs appearing in only ONE video:",
    int((videos_per_pid == 1).sum())
)

# --------------------------------------------------
# 3. Clothes annotation coverage
# clothes_id == 0 means unknown
# --------------------------------------------------

known = train[train["clothes_id"] != 0].copy()
unknown = train[train["clothes_id"] == 0].copy()

print("\n" + "=" * 75)
print("CLOTHES ANNOTATION COVERAGE")
print("=" * 75)

print("\nTrain images:", len(train))

print(
    "Known clothes images:",
    len(known),
    f"({len(known) / len(train) * 100:.2f}%)"
)

print(
    "Unknown clothes images:",
    len(unknown),
    f"({len(unknown) / len(train) * 100:.2f}%)"
)

# --------------------------------------------------
# 4. Known clothes IDs per identity
# IMPORTANT:
# clothes labels are analyzed within each PID
# --------------------------------------------------

known_clothes_per_pid = (
    known.groupby("pid")["clothes_id"]
    .nunique()
)

# Include PIDs with zero known clothes labels
all_pids = sorted(train["pid"].unique())

known_clothes_per_pid = (
    known_clothes_per_pid
    .reindex(all_pids, fill_value=0)
)

print("\nKnown clothes IDs per PID:")
print(
    known_clothes_per_pid
    .value_counts()
    .sort_index()
)

print("\nClothes coverage statistics:")
print(
    known_clothes_per_pid
    .describe()
    .round(2)
)

ids_no_known_clothes = int(
    (known_clothes_per_pid == 0).sum()
)

ids_one_known_clothes = int(
    (known_clothes_per_pid == 1).sum()
)

ids_multi_clothes = int(
    (known_clothes_per_pid >= 2).sum()
)

print("\nIdentity-level clothes summary:")
print("IDs with 0 known clothes labels :", ids_no_known_clothes)
print("IDs with 1 known clothes label  :", ids_one_known_clothes)
print("IDs with >=2 clothes labels      :", ids_multi_clothes)

print(
    "Multi-clothes ID percentage     :",
    f"{ids_multi_clothes / len(all_pids) * 100:.2f}%"
)

# --------------------------------------------------
# 5. Videos associated with each known clothes state
# --------------------------------------------------

pid_clothes_video = (
    known.groupby(["pid", "clothes_id"])["video_id"]
    .nunique()
    .reset_index(name="videos_for_clothes")
)

print("\nVideos per (PID, known-clothes-ID):")
print(
    pid_clothes_video["videos_for_clothes"]
    .describe()
    .round(2)
)

# --------------------------------------------------
# 6. For multi-clothes identities:
# do their different clothes appear across videos?
# --------------------------------------------------

multi_clothes_pids = set(
    known_clothes_per_pid[
        known_clothes_per_pid >= 2
    ].index
)

multi = known[
    known["pid"].isin(multi_clothes_pids)
].copy()

multi_videos_per_pid = (
    multi.groupby("pid")["video_id"]
    .nunique()
)

if len(multi_videos_per_pid) > 0:

    print("\nVideos per multi-clothes PID:")
    print(
        multi_videos_per_pid
        .describe()
        .round(2)
    )

    print(
        "\nMulti-clothes IDs appearing in >=2 videos:",
        int((multi_videos_per_pid >= 2).sum()),
        "/",
        len(multi_videos_per_pid)
    )

# --------------------------------------------------
# 7. Save detailed per-PID report
# --------------------------------------------------

pid_report = pd.DataFrame({
    "images": images_per_pid,
    "videos": videos_per_pid,
    "known_clothes_ids": known_clothes_per_pid,
})

known_images_per_pid = (
    known.groupby("pid").size()
    .reindex(all_pids, fill_value=0)
)

unknown_images_per_pid = (
    unknown.groupby("pid").size()
    .reindex(all_pids, fill_value=0)
)

pid_report["known_clothes_images"] = known_images_per_pid
pid_report["unknown_clothes_images"] = unknown_images_per_pid

pid_report.to_csv(
    REPORT_DIR / "train_pid_video_clothes.csv"
)

pid_clothes_video.to_csv(
    REPORT_DIR / "train_pid_clothes_video.csv",
    index=False
)

print("\nSaved:")
print(REPORT_DIR / "train_pid_video_clothes.csv")
print(REPORT_DIR / "train_pid_clothes_video.csv")
