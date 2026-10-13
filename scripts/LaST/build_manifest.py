from pathlib import Path
import re
import pandas as pd

ROOT = Path.home() / "DG_ReID_Audit/datasets/LaST/extracted/last"
OUT = Path.home() / "DG_ReID_Audit/manifests/LaST"

OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Normal filename
# PID_IMAGEID_VIDEOID_FRAMEID_BBOXID_CLOTHESID.jpg
# ---------------------------------------------------------

normal_pattern = re.compile(
    r"^(?P<pid>\d{6})_"
    r"(?P<image_id>\d{3})_"
    r"(?P<video_id>\d{5})_"
    r"(?P<frame_id>\d+)_"
    r"(?P<bbox_id>\d+)_"
    r"(?P<clothes_id>\d{3})\.jpg$"
)

# ---------------------------------------------------------
# Known anomaly:
# 005859__003_00142_1165_03_000.jpg
# ---------------------------------------------------------

double_underscore_pattern = re.compile(
    r"^(?P<pid>\d{6})__"
    r"(?P<image_id>\d{3})_"
    r"(?P<video_id>\d{5})_"
    r"(?P<frame_id>\d+)_"
    r"(?P<bbox_id>\d+)_"
    r"(?P<clothes_id>\d{3})\.jpg$"
)

sources = [
    ("train", ROOT / "train"),
    ("val_query", ROOT / "val/query"),
    ("val_gallery", ROOT / "val/gallery"),
    ("test_query", ROOT / "test/query"),
    ("test_gallery", ROOT / "test/gallery"),
]

rows = []
unparsed = []

for split, base in sources:

    for path in sorted(base.rglob("*.jpg")):

        filename = path.name

        m = normal_pattern.match(filename)

        filename_anomaly = False
        anomaly_type = None

        if m is None:

            m = double_underscore_pattern.match(filename)

            if m is not None:
                filename_anomaly = True
                anomaly_type = "double_underscore_after_pid"

        if m is None:

            unparsed.append({
                "split": split,
                "filename": filename,
                "path": str(path),
            })

            continue

        # Query images are directly inside query/
        if path.parent == base:
            folder_pid = None
        else:
            folder_pid = path.parent.name

        is_distractor = (
            split in {"val_gallery", "test_gallery"}
            and folder_pid == "000000"
        )

        filename_pid = int(m.group("pid"))

        folder_pid_mismatch = False

        if (
            folder_pid is not None
            and not is_distractor
            and folder_pid.isdigit()
        ):
            folder_pid_mismatch = (
                int(folder_pid) != filename_pid
            )

        rows.append({
            "filename": filename,
            "split": split,

            "pid": filename_pid,
            "image_id": int(m.group("image_id")),
            "video_id": int(m.group("video_id")),
            "frame_id": int(m.group("frame_id")),
            "bbox_id": int(m.group("bbox_id")),
            "clothes_id": int(m.group("clothes_id")),

            "folder_pid": folder_pid,

            "is_distractor": is_distractor,

            "filename_anomaly": filename_anomaly,
            "anomaly_type": anomaly_type,

            "folder_pid_mismatch": folder_pid_mismatch,

            "path": str(path),
        })

df = pd.DataFrame(rows)

print("=" * 70)
print("LAST FINAL MANIFEST AUDIT")
print("=" * 70)

print("\nParsed   :", len(df))
print("Unparsed :", len(unparsed))

print("\nSplit counts:")
print(
    df["split"]
    .value_counts()
    .reindex([
        "train",
        "val_query",
        "val_gallery",
        "test_query",
        "test_gallery",
    ])
)

print("\nFilename anomalies:")
print(
    df["filename_anomaly"]
    .value_counts()
)

print("\nAnomaly types:")
print(
    df["anomaly_type"]
    .value_counts(dropna=False)
)

print("\nFolder PID mismatches:")
print(
    df["folder_pid_mismatch"]
    .value_counts()
)

mismatches = df[
    df["folder_pid_mismatch"]
]

if len(mismatches):

    print("\nMismatch examples:")

    print(
        mismatches[
            [
                "split",
                "folder_pid",
                "pid",
                "filename",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )

print("\nDistractors:")
print(
    df.groupby("split")["is_distractor"]
    .sum()
)

print("\nNon-distractor unique PIDs:")

for split in [
    "train",
    "val_query",
    "val_gallery",
    "test_query",
    "test_gallery",
]:

    sdf = df[
        (df["split"] == split)
        & (~df["is_distractor"])
    ]

    print(
        f"{split:<15}: "
        f"{sdf['pid'].nunique()}"
    )

print("\nUnique videos:")

for split in [
    "train",
    "val_query",
    "val_gallery",
    "test_query",
    "test_gallery",
]:

    sdf = df[
        df["split"] == split
    ]

    print(
        f"{split:<15}: "
        f"{sdf['video_id'].nunique()}"
    )

print(
    "\nUnknown clothes images:",
    (df["clothes_id"] == 0).sum()
)

if unparsed:

    print("\nUNPARSED FILES:")

    for item in unparsed[:20]:
        print(item)

output = OUT / "last_manifest.csv"

df.to_csv(
    output,
    index=False
)

print("\nSaved:", output)
