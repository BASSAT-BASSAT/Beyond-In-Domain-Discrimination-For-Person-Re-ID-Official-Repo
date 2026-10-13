from pathlib import Path
import pandas as pd
import re

ROOT = Path.home() / "DG_ReID_Audit/datasets/DeepChange"
OUT = Path.home() / "DG_ReID_Audit/manifests/DeepChange"

OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": ROOT / "train-set-bbox.txt",
    "val_query": ROOT / "val-set-query-bbox.txt",
    "val_gallery": ROOT / "val-set-gallery-bbox.txt",
    "test_query": ROOT / "test-set-query-bbox.txt",
    "test_gallery": ROOT / "test-set-gallery-bbox.txt",
}

pattern = re.compile(
    r"^p(?P<pid>\d{4})_"
    r"c(?P<camera_id>\d{2})_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{6})_"
    r"(?P<frame_id>\d+)_bbox\.jpg$"
)

rows = []
invalid = []

for split, txt_path in FILES.items():

    with open(txt_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            filename, tracklet_id = line.rsplit(",", 1)

            m = pattern.match(filename)

            if not m:
                invalid.append({
                    "split": split,
                    "filename": filename,
                })
                continue

            date = m.group("date")
            time = m.group("time")

            timestamp = pd.to_datetime(
                date + time,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )

            rows.append({
                "filename": filename,
                "split": split,
                "pid": int(m.group("pid")),
                "camera_id": int(m.group("camera_id")),
                "date": date,
                "time": time,
                "timestamp": timestamp,
                "frame_id": int(m.group("frame_id")),
                "tracklet_id": int(tracklet_id),
            })

df = pd.DataFrame(rows)

print("=" * 75)
print("DEEPCHANGE MANIFEST")
print("=" * 75)

print("\nParsed :", len(df))
print("Invalid:", len(invalid))

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

print("\nUnique PIDs per split:")
print(
    df.groupby("split")["pid"]
    .nunique()
)

print("\nUnique cameras per split:")
print(
    df.groupby("split")["camera_id"]
    .nunique()
)

print("\nUnique tracklets per split:")
print(
    df.groupby("split")["tracklet_id"]
    .nunique()
)

print("\nGlobal camera IDs:")
print(
    sorted(df["camera_id"].unique())
)

print("\nGlobal date range:")
print("Earliest:", df["timestamp"].min())
print("Latest  :", df["timestamp"].max())

print(
    "\nInvalid timestamps:",
    df["timestamp"].isna().sum()
)

if invalid:
    print("\nInvalid filename examples:")
    for x in invalid[:20]:
        print(x)

output = OUT / "deepchange_manifest.csv"

df.to_csv(
    output,
    index=False,
)

print("\nSaved:", output)
