from pathlib import Path
from collections import defaultdict, Counter
import hashlib

import pandas as pd


MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_manifest_v2.csv"
)

OUT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/Occluded-PoseTrack-ReID"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "video_id": str,
        "logical_camid": str,
        "posetrack_image_id": str,
        "annotation_id": str,
    },
)


# ============================================================
# Hash helper
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# Compute hashes
# ============================================================

print("=" * 90)
print("OCCLUDED-POSETRACK-REID EXACT DUPLICATE AUDIT")
print("=" * 90)

print("\nImages:", len(df))
print("Computing SHA256 hashes...")


hashes = []

for i, row in df.iterrows():

    digest = sha256_file(
        row["path"]
    )

    hashes.append(digest)

    if (i + 1) % 5000 == 0:
        print(
            f"  processed {i + 1:,} / {len(df):,}"
        )


df["sha256"] = hashes


# ============================================================
# Duplicate groups
# ============================================================

groups = (
    df.groupby("sha256")
    .filter(lambda x: len(x) > 1)
)

duplicate_hashes = (
    groups["sha256"]
    .unique()
    .tolist()
)


print("\n" + "=" * 90)
print("EXACT FILE-BYTE DUPLICATES")
print("=" * 90)

print(
    "Duplicate groups        :",
    len(duplicate_hashes)
)

print(
    "Images in dup groups    :",
    len(groups)
)

print(
    "Extra duplicate copies  :",
    len(groups)
    - len(duplicate_hashes)
)


# ============================================================
# Classify groups
# ============================================================

cross_person = 0
same_person = 0

cross_video = 0
same_video = 0

cross_split = 0

train_eval = 0
query_gallery = 0

group_summaries = []


for digest in duplicate_hashes:

    g = df[
        df["sha256"] == digest
    ].copy()

    persons = set(
        g["person_id"]
    )

    videos = set(
        g["video_id"]
    )

    splits = set(
        g["split"]
    )


    if len(persons) > 1:
        cross_person += 1
    else:
        same_person += 1


    if len(videos) > 1:
        cross_video += 1
    else:
        same_video += 1


    if len(splits) > 1:
        cross_split += 1


    has_train = (
        "train" in splits
    )

    has_eval = bool(
        splits
        & {"query", "gallery"}
    )

    if has_train and has_eval:
        train_eval += 1


    if (
        "query" in splits
        and "gallery" in splits
    ):
        query_gallery += 1


    group_summaries.append(
        {
            "sha256":
                digest,

            "n_images":
                len(g),

            "n_person_ids":
                len(persons),

            "n_videos":
                len(videos),

            "splits":
                ",".join(
                    sorted(splits)
                ),

            "person_ids":
                ",".join(
                    str(x)
                    for x in sorted(persons)
                ),

            "videos":
                ",".join(
                    sorted(videos)
                ),

            "cross_person":
                len(persons) > 1,

            "cross_video":
                len(videos) > 1,

            "cross_split":
                len(splits) > 1,

            "train_eval":
                has_train and has_eval,

            "query_gallery":
                (
                    "query" in splits
                    and "gallery" in splits
                ),
        }
    )


print("\nGROUP CHARACTERISTICS")
print("-" * 90)

print(
    "Same-person groups      :",
    same_person
)

print(
    "Cross-person groups     :",
    cross_person
)

print(
    "Same-video groups       :",
    same_video
)

print(
    "Cross-video groups      :",
    cross_video
)

print(
    "Cross-split groups      :",
    cross_split
)

print(
    "Train ↔ Eval groups     :",
    train_eval
)

print(
    "Query ↔ Gallery groups  :",
    query_gallery
)


# ============================================================
# Detailed examples
# ============================================================

if duplicate_hashes:

    print("\n" + "=" * 90)
    print("DUPLICATE GROUP EXAMPLES")
    print("=" * 90)

    for idx, digest in enumerate(
        duplicate_hashes[:20],
        start=1
    ):

        g = df[
            df["sha256"] == digest
        ]

        print(
            f"\nGroup {idx}"
        )

        print(
            "-" * 90
        )

        print(
            g[
                [
                    "split",
                    "person_id",
                    "video_id",
                    "posetrack_image_id",
                    "filename",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# Save detailed outputs
# ============================================================

HASHED_MANIFEST = (
    OUT_DIR
    / "manifest_with_sha256.csv"
)

df.to_csv(
    HASHED_MANIFEST,
    index=False
)


GROUP_FILE = (
    OUT_DIR
    / "exact_duplicate_groups.csv"
)

pd.DataFrame(
    group_summaries
).to_csv(
    GROUP_FILE,
    index=False
)


DUPLICATE_ROWS_FILE = (
    OUT_DIR
    / "exact_duplicate_rows.csv"
)

groups.to_csv(
    DUPLICATE_ROWS_FILE,
    index=False
)


print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(
    "Hashed manifest:",
    HASHED_MANIFEST
)

print(
    "Duplicate groups:",
    GROUP_FILE
)

print(
    "Duplicate rows:",
    DUPLICATE_ROWS_FILE
)

print("\nDONE")
