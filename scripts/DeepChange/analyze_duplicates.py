from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
from PIL import Image


MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/DeepChange/extracted"
)

REPORT = (
    Path.home()
    / "DG_ReID_Audit/reports/DeepChange"
)

REPORT.mkdir(parents=True, exist_ok=True)

SPLIT_DIRS = {
    "train": "train-set",
    "val_query": "val-set-query",
    "val_gallery": "val-set-gallery",
    "test_query": "test-set-query",
    "test_gallery": "test-set-gallery",
}


df = pd.read_csv(MANIFEST)


# ============================================================
# Hash helpers
# ============================================================

def file_sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def pixel_sha256(img):

    h = hashlib.sha256()

    h.update(
        img.tobytes()
    )

    return h.hexdigest()


def dhash64(img):

    # 9x8 -> compare adjacent horizontal pixels
    small = (
        img
        .convert("L")
        .resize((9, 8))
    )

    pixels = list(
        small.getdata()
    )

    value = 0

    bit = 0

    for row in range(8):

        offset = row * 9

        for col in range(8):

            left = pixels[offset + col]
            right = pixels[offset + col + 1]

            if left > right:
                value |= (1 << bit)

            bit += 1

    return f"{value:016x}"


# ============================================================
# Compute hashes
# ============================================================

rows = []

print("=" * 80)
print("DEEPCHANGE DUPLICATE AUDIT")
print("=" * 80)

print("\nImages:", len(df))
print("Computing hashes...")


for i, row in df.iterrows():

    path = (
        ROOT
        / SPLIT_DIRS[row["split"]]
        / row["filename"]
    )

    with Image.open(path) as img:

        img = img.convert("RGB")

        rows.append({
            "filename": row["filename"],
            "split": row["split"],
            "pid": row["pid"],
            "camera_id": row["camera_id"],
            "tracklet_id": row["tracklet_id"],
            "file_sha256": file_sha256(path),
            "pixel_sha256": pixel_sha256(img),
            "dhash64": dhash64(img),
            "path": str(path),
        })


hashes = pd.DataFrame(rows)


# ============================================================
# Generic duplicate summary
# ============================================================

def duplicate_groups(column):

    counts = hashes[column].value_counts()

    duplicate_values = set(
        counts[counts > 1].index
    )

    groups = []

    for value in duplicate_values:

        g = hashes[
            hashes[column] == value
        ].copy()

        groups.append(g)

    return groups


file_groups = duplicate_groups("file_sha256")
pixel_groups = duplicate_groups("pixel_sha256")
dhash_groups = duplicate_groups("dhash64")


# ============================================================
# Analyze exact pixel groups
# ============================================================

def classify_groups(groups, label):

    duplicate_images = sum(
        len(g)
        for g in groups
    )

    cross_pid = []
    cross_tracklet = []
    cross_split = []
    train_eval = []
    val_qg = []
    test_qg = []

    for g in groups:

        splits = set(g["split"])
        pids = set(g["pid"])
        tracks = set(g["tracklet_id"])

        if len(pids) > 1:
            cross_pid.append(g)

        if len(tracks) > 1:
            cross_tracklet.append(g)

        if len(splits) > 1:
            cross_split.append(g)

        if (
            "train" in splits
            and any(
                s != "train"
                for s in splits
            )
        ):
            train_eval.append(g)

        if (
            "val_query" in splits
            and "val_gallery" in splits
        ):
            val_qg.append(g)

        if (
            "test_query" in splits
            and "test_gallery" in splits
        ):
            test_qg.append(g)


    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print(
        "Duplicate groups       :",
        len(groups)
    )

    print(
        "Images inside groups   :",
        duplicate_images
    )

    print(
        "Cross-PID groups       :",
        len(cross_pid)
    )

    print(
        "Cross-tracklet groups  :",
        len(cross_tracklet)
    )

    print(
        "Cross-split groups     :",
        len(cross_split)
    )

    print(
        "Train ↔ Eval groups    :",
        len(train_eval)
    )

    print(
        "Val Query ↔ Gallery    :",
        len(val_qg)
    )

    print(
        "Test Query ↔ Gallery   :",
        len(test_qg)
    )

    return {
        "cross_pid": cross_pid,
        "cross_tracklet": cross_tracklet,
        "cross_split": cross_split,
        "train_eval": train_eval,
        "val_qg": val_qg,
        "test_qg": test_qg,
    }


file_summary = classify_groups(
    file_groups,
    "EXACT FILE-BYTE DUPLICATES"
)

pixel_summary = classify_groups(
    pixel_groups,
    "EXACT PIXEL DUPLICATES"
)


# ============================================================
# dHash candidate groups
#
# Same dHash ≠ guaranteed duplicate.
# These are only visual near-duplicate candidates.
# ============================================================

print("\n" + "=" * 80)
print("dHASH NEAR-DUPLICATE CANDIDATES")
print("=" * 80)

print(
    "Same-dHash groups:",
    len(dhash_groups)
)

suspicious_dhash = []

for g in dhash_groups:

    # Ignore groups that are all:
    # same PID + same tracklet
    # because near-identical frames there are expected.
    if (
        g["pid"].nunique() == 1
        and g["tracklet_id"].nunique() == 1
    ):
        continue

    suspicious_dhash.append(g)


print(
    "Groups crossing PID or tracklet:",
    len(suspicious_dhash)
)


cross_pid_dhash = [
    g
    for g in suspicious_dhash
    if g["pid"].nunique() > 1
]

cross_track_dhash = [
    g
    for g in suspicious_dhash
    if g["tracklet_id"].nunique() > 1
]


print(
    "Cross-PID dHash groups:",
    len(cross_pid_dhash)
)

print(
    "Cross-tracklet dHash groups:",
    len(cross_track_dhash)
)


# ============================================================
# Save suspicious groups as rows
# ============================================================

def flatten_groups(groups, group_type):

    output = []

    for group_id, g in enumerate(groups, start=1):

        for _, row in g.iterrows():

            output.append({
                "group_type": group_type,
                "group_id": group_id,
                **row.to_dict(),
            })

    return output


suspicious_rows = []

suspicious_rows += flatten_groups(
    pixel_summary["cross_pid"],
    "exact_pixel_cross_pid"
)

suspicious_rows += flatten_groups(
    pixel_summary["cross_tracklet"],
    "exact_pixel_cross_tracklet"
)

suspicious_rows += flatten_groups(
    pixel_summary["train_eval"],
    "exact_pixel_train_eval"
)

suspicious_rows += flatten_groups(
    cross_pid_dhash,
    "dhash_cross_pid"
)

suspicious_rows += flatten_groups(
    cross_track_dhash,
    "dhash_cross_tracklet"
)


suspicious_df = pd.DataFrame(
    suspicious_rows
)


hashes.to_csv(
    REPORT / "image_hashes.csv",
    index=False
)

suspicious_df.to_csv(
    REPORT / "duplicate_candidates.csv",
    index=False
)


print("\nSaved:")
print(REPORT / "image_hashes.csv")
print(REPORT / "duplicate_candidates.csv")


# ============================================================
# Print examples
# ============================================================

if len(suspicious_df):

    print("\n" + "=" * 80)
    print("SUSPICIOUS EXAMPLES")
    print("=" * 80)

    for group_type, x in (
        suspicious_df
        .groupby("group_type")
    ):

        print(f"\n[{group_type}]")

        for group_id, g in (
            x.groupby("group_id")
        ):

            print(
                f"\nGroup {group_id}"
            )

            for _, row in g.head(6).iterrows():

                print(
                    f"  "
                    f"{row['split']:<12} "
                    f"PID={row['pid']} "
                    f"T={row['tracklet_id']} "
                    f"C={row['camera_id']} "
                    f"{row['filename']}"
                )

            # only a few example groups
            if group_id >= 5:
                break
