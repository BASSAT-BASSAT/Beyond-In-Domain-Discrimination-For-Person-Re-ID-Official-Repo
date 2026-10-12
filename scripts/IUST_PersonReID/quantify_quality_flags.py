from pathlib import Path
import pandas as pd

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/IUST_PersonReID"
)

stats = pd.read_csv(
    REPORT_DIR / "image_stats.csv"
)

# --------------------------------------------------
# Diagnostic flags
# NOT deletion rules.
# --------------------------------------------------

stats["tiny"] = (
    (stats["width"] < 50)
    | (stats["height"] < 100)
)

stats["very_narrow"] = (
    stats["aspect_ratio"] < 0.20
)

stats["horizontal"] = (
    stats["aspect_ratio"] > 1.0
)

stats["very_wide"] = (
    stats["aspect_ratio"] > 2.0
)

stats["extreme_any"] = (
    stats["tiny"]
    | stats["very_narrow"]
    | stats["horizontal"]
)

print("=" * 70)
print("IUST QUALITY FLAG AUDIT")
print("=" * 70)

for split in ["train", "query", "test"]:

    sdf = stats[
        stats["split"] == split
    ]

    print("\n" + "=" * 25, split.upper(), "=" * 25)

    total = len(sdf)

    print("Total images:", total)

    for flag in [
        "tiny",
        "very_narrow",
        "horizontal",
        "very_wide",
        "extreme_any",
    ]:

        count = int(sdf[flag].sum())
        pct = count / total * 100

        print(
            f"{flag:<15}: "
            f"{count:>6} "
            f"({pct:6.2f}%)"
        )


# --------------------------------------------------
# TRAIN: distribution by camera
# --------------------------------------------------

train = stats[
    stats["split"] == "train"
].copy()

print("\n" + "=" * 70)
print("TRAIN FLAGS BY CAMERA")
print("=" * 70)

camera_stats = (
    train.groupby("camera")
    .agg(
        images=("filename", "count"),

        tiny=("tiny", "sum"),
        very_narrow=("very_narrow", "sum"),
        horizontal=("horizontal", "sum"),
        very_wide=("very_wide", "sum"),
        extreme_any=("extreme_any", "sum"),
    )
)

camera_stats["extreme_pct"] = (
    camera_stats["extreme_any"]
    / camera_stats["images"]
    * 100
)

print(
    camera_stats
    .sort_values(
        "extreme_pct",
        ascending=False
    )
    .round(2)
)


# --------------------------------------------------
# How many identities are affected?
# --------------------------------------------------

print("\n" + "=" * 70)
print("IDENTITIES AFFECTED IN TRAIN")
print("=" * 70)

total_pids = train["pid"].nunique()

for flag in [
    "tiny",
    "very_narrow",
    "horizontal",
    "very_wide",
]:

    affected_pids = (
        train.loc[
            train[flag],
            "pid"
        ]
        .nunique()
    )

    print(
        f"{flag:<15}: "
        f"{affected_pids}/{total_pids} IDs "
        f"({affected_pids / total_pids * 100:.2f}%)"
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

stats.to_csv(
    REPORT_DIR / "image_quality_flags.csv",
    index=False
)

camera_stats.to_csv(
    REPORT_DIR / "train_quality_flags_by_camera.csv"
)

print("\nSaved:")
print(REPORT_DIR / "image_quality_flags.csv")
print(REPORT_DIR / "train_quality_flags_by_camera.csv")
