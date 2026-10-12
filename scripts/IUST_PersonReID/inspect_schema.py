from pathlib import Path
import re
import pandas as pd

ROOT = Path.home() / "DG_ReID_Audit/datasets/IUST_PersonReID/extracted/IUSTPersonReID"

pattern = re.compile(
    r"^(?P<pid>\d+)_c(?P<camera>\d+)s(?P<sequence>\d+)_(?P<frame>\d+)_(?P<suffix>\d+)\.jpg$"
)

split_map = {
    "bounding_box_train": "train",
    "query": "query",
    "bounding_box_test": "test",
}

rows = []
invalid = []

for folder_name, split in split_map.items():

    folder = ROOT / folder_name

    for path in sorted(folder.glob("*.jpg")):

        m = pattern.match(path.name)

        if not m:
            invalid.append((split, path.name))
            continue

        rows.append({
            "filename": path.name,
            "split": split,
            "pid": int(m.group("pid")),
            "camera": int(m.group("camera")),
            "sequence": int(m.group("sequence")),
            "frame": int(m.group("frame")),
            "suffix": int(m.group("suffix")),
            "path": str(path),
        })

df = pd.DataFrame(rows)

print("=" * 70)
print("FILENAME PARSING")
print("=" * 70)

print("Parsed :", len(df))
print("Invalid:", len(invalid))

print("\nUnique camera codes:")
print(sorted(df["camera"].unique()))

print("\nNumber of camera codes:")
print(df["camera"].nunique())

print("\nUnique sequence codes:")
print(sorted(df["sequence"].unique()))

print("\nUnique suffix codes:")
print(sorted(df["suffix"].unique()))

print("\nSequence counts:")
print(df["sequence"].value_counts().sort_index())

print("\nSuffix counts:")
print(df["suffix"].value_counts().sort_index())

print("\nImages per camera:")
print(df["camera"].value_counts().sort_index())

print("\nSplit summary:")
print(
    df.groupby("split").agg(
        images=("filename", "count"),
        pids=("pid", "nunique"),
        cameras=("camera", "nunique"),
        sequences=("sequence", "nunique"),
    )
)

# -------------------------------------------------
# ID relationships
# -------------------------------------------------

train_ids = set(df.loc[df["split"] == "train", "pid"])
query_ids = set(df.loc[df["split"] == "query", "pid"])
test_ids = set(df.loc[df["split"] == "test", "pid"])

print("\n" + "=" * 70)
print("PID RELATIONSHIPS")
print("=" * 70)

print("Train IDs :", len(train_ids))
print("Query IDs :", len(query_ids))
print("Test IDs  :", len(test_ids))

print("Train ∩ Query:", len(train_ids & query_ids))
print("Train ∩ Test :", len(train_ids & test_ids))
print("Query - Test :", len(query_ids - test_ids))
print("Test - Query :", len(test_ids - query_ids))

# -------------------------------------------------
# Filename overlap
# -------------------------------------------------

train_files = set(df.loc[df["split"] == "train", "filename"])
query_files = set(df.loc[df["split"] == "query", "filename"])
test_files = set(df.loc[df["split"] == "test", "filename"])

print("\n" + "=" * 70)
print("FILENAME OVERLAP")
print("=" * 70)

print("Train ∩ Query:", len(train_files & query_files))
print("Train ∩ Test :", len(train_files & test_files))
print("Query ∩ Test :", len(query_files & test_files))

print(
    "All query filenames inside test:",
    query_files.issubset(test_files)
)

if query_files & test_files:
    print("\nExample overlapping Query/Test filenames:")
    for name in sorted(query_files & test_files)[:10]:
        print(name)

if invalid:
    print("\nInvalid filename examples:")
    for item in invalid[:10]:
        print(item)

OUT = Path.home() / "DG_ReID_Audit/manifests/IUST_PersonReID"
OUT.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUT / "iust_manifest_raw.csv",
    index=False
)

print("\nSaved:", OUT / "iust_manifest_raw.csv")
