from pathlib import Path
import pandas as pd

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

print("=" * 70)
print("IUST DISTRIBUTION AUDIT")
print("=" * 70)

for split in ["train", "test"]:

    sdf = df[df["split"] == split].copy()

    print(f"\n{'=' * 25} {split.upper()} {'=' * 25}")

    # -----------------------------------------------
    # Images per identity
    # -----------------------------------------------

    images_per_pid = sdf.groupby("pid").size()

    print("\nImages per ID:")
    print(f"IDs    : {len(images_per_pid)}")
    print(f"Min    : {images_per_pid.min()}")
    print(f"Max    : {images_per_pid.max()}")
    print(f"Mean   : {images_per_pid.mean():.2f}")
    print(f"Median : {images_per_pid.median():.2f}")
    print(f"Std    : {images_per_pid.std():.2f}")

    print("\nImages-per-ID percentiles:")
    print(
        images_per_pid.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        ).round(2)
    )

    # -----------------------------------------------
    # Cameras per identity
    # -----------------------------------------------

    cameras_per_pid = (
        sdf.groupby("pid")["camera"]
        .nunique()
    )

    print("\nCameras per ID:")
    print(cameras_per_pid.value_counts().sort_index())

    print("\nCamera coverage statistics:")
    print(
        cameras_per_pid.describe().round(2)
    )

    single_camera_ids = (
        cameras_per_pid[cameras_per_pid == 1]
        .index.tolist()
    )

    print(
        "\nIDs appearing in ONE camera only:",
        len(single_camera_ids)
    )

    # -----------------------------------------------
    # Camera distribution
    # -----------------------------------------------

    camera_stats = (
        sdf.groupby("camera")
        .agg(
            images=("filename", "count"),
            unique_pids=("pid", "nunique"),
        )
        .sort_index()
    )

    print("\nPer-camera distribution:")
    print(camera_stats)

    # -----------------------------------------------
    # Most / least represented IDs
    # -----------------------------------------------

    print("\n10 IDs with most images:")
    print(
        images_per_pid
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n10 IDs with fewest images:")
    print(
        images_per_pid
        .sort_values()
        .head(10)
    )

    # -----------------------------------------------
    # Save detailed per-PID report
    # -----------------------------------------------

    pid_report = pd.DataFrame({
        "images": images_per_pid,
        "cameras": cameras_per_pid,
    })

    pid_report.to_csv(
        REPORT_DIR /
        f"{split}_pid_distribution.csv"
    )

    camera_stats.to_csv(
        REPORT_DIR /
        f"{split}_camera_distribution.csv"
    )


print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

query = df[df["split"] == "query"]

query_per_pid = query.groupby("pid").size()

print("\nQuery images per ID:")
print(query_per_pid.value_counts().sort_index())

print("\nQuery cameras:")
print(
    query["camera"]
    .value_counts()
    .sort_index()
)

print("\nSaved reports to:")
print(REPORT_DIR)
