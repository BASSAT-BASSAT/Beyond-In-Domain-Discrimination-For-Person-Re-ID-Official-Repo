from pathlib import Path
from collections import Counter
import pandas as pd


MANIFEST_V1 = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_manifest.csv"
)

MANIFEST_V2 = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_manifest_v2.csv"
)


df = pd.read_csv(
    MANIFEST_V1,
    dtype={
        "video_id": str,
        "logical_camid": str,
        "posetrack_image_id": str,
        "annotation_id": str,
    },
)


# ============================================================
# 1. Correct terminology
# ============================================================

if "pid" in df.columns:
    df = df.rename(columns={"pid": "person_id"})

df.to_csv(
    MANIFEST_V2,
    index=False
)


train = df[df["split"] == "train"].copy()
query = df[df["split"] == "query"].copy()
gallery = df[df["split"] == "gallery"].copy()
eval_df = df[df["split"].isin(["query", "gallery"])].copy()


print("=" * 90)
print("OCCLUDED-POSETRACK-REID IDENTITY / VIDEO STRUCTURE")
print("=" * 90)


# ============================================================
# Helper
# ============================================================

def describe_images_per_id(name, data):

    counts = (
        data.groupby("person_id")
        .size()
    )

    print(f"\n{name} — IMAGES PER PERSON")
    print("-" * 90)

    print("Persons :", len(counts))
    print("Min     :", int(counts.min()))
    print("Median  :", float(counts.median()))
    print("Mean    :", round(float(counts.mean()), 2))
    print("Max     :", int(counts.max()))

    print("\nExact distribution:")

    dist = Counter(counts.tolist())

    for n_images, n_persons in sorted(dist.items()):
        print(
            f"  {n_images:>2} image(s) "
            f"→ {n_persons:>5} person(s)"
        )


def describe_videos_per_id(name, data):

    counts = (
        data.groupby("person_id")["video_id"]
        .nunique()
    )

    print(f"\n{name} — VIDEOS PER PERSON")
    print("-" * 90)

    dist = Counter(counts.tolist())

    for n_videos, n_persons in sorted(dist.items()):
        print(
            f"  {n_videos:>2} video(s) "
            f"→ {n_persons:>5} person(s)"
        )

    print(
        "Persons appearing in >1 video:",
        int((counts > 1).sum())
    )


# ============================================================
# 2. Images per identity
# ============================================================

describe_images_per_id(
    "TRAIN",
    train
)

describe_images_per_id(
    "EVAL (QUERY + GALLERY)",
    eval_df
)

describe_images_per_id(
    "QUERY",
    query
)

describe_images_per_id(
    "GALLERY",
    gallery
)


# ============================================================
# 3. Videos per identity
# ============================================================

describe_videos_per_id(
    "TRAIN",
    train
)

describe_videos_per_id(
    "EVAL (QUERY + GALLERY)",
    eval_df
)


# ============================================================
# 4. Query / gallery counts per identity
# ============================================================

q_counts = (
    query.groupby("person_id")
    .size()
)

g_counts = (
    gallery.groupby("person_id")
    .size()
)

eval_counts = (
    eval_df.groupby("person_id")
    .size()
)

all_eval_ids = sorted(
    set(eval_df["person_id"])
)

rows = []

for person_id in all_eval_ids:

    rows.append(
        {
            "person_id": person_id,
            "total": int(
                eval_counts.get(person_id, 0)
            ),
            "query": int(
                q_counts.get(person_id, 0)
            ),
            "gallery": int(
                g_counts.get(person_id, 0)
            ),
        }
    )

per_id = pd.DataFrame(rows)


print("\nQUERY / GALLERY PER PERSON")
print("-" * 90)

print("\nQuery-count distribution:")

for n_queries, n_people in sorted(
    Counter(
        per_id["query"].tolist()
    ).items()
):
    print(
        f"  {n_queries} query image(s) "
        f"→ {n_people} person(s)"
    )


# ============================================================
# 5. Gallery-only identities
# ============================================================

gallery_only = (
    per_id[
        per_id["query"] == 0
    ]
    .copy()
)


print("\nGALLERY-ONLY IDENTITIES")
print("-" * 90)

print(
    "Gallery-only persons:",
    len(gallery_only)
)

print(
    "Gallery-only images :",
    int(
        gallery_only["gallery"]
        .sum()
    )
)

print("\nTotal images per gallery-only person:")

for n_images, n_people in sorted(
    Counter(
        gallery_only["total"]
        .tolist()
    ).items()
):

    print(
        f"  {n_images} image(s) "
        f"→ {n_people} person(s)"
    )


print("\nGallery-only examples:")

print(
    gallery_only
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 6. Query validity under official default protocol
#
# mot_inter_intra_video:
# same person is sufficient;
# same video is NOT excluded.
# ============================================================

gallery_pids = set(
    gallery["person_id"]
)

valid_query_mask = (
    query["person_id"]
    .isin(gallery_pids)
)

print("\nOFFICIAL DEFAULT QUERY VALIDITY")
print("-" * 90)

print(
    "Total queries          :",
    len(query)
)

print(
    "Queries with positive  :",
    int(valid_query_mask.sum())
)

print(
    "Queries without positive:",
    int((~valid_query_mask).sum())
)


# ============================================================
# 7. Cross-video characteristic
#    Not an error under default protocol
# ============================================================

gallery_by_pid = {
    pid: group
    for pid, group
    in gallery.groupby("person_id")
}

cross_video = 0
same_video_only = 0

for _, q in query.iterrows():

    g = gallery_by_pid[
        q["person_id"]
    ]

    if (
        g["video_id"]
        != q["video_id"]
    ).any():

        cross_video += 1

    else:
        same_video_only += 1


print("\nCROSS-VIDEO CHARACTERISTIC")
print("-" * 90)

print(
    "Queries with cross-video positive:",
    cross_video
)

print(
    "Queries with same-video-only positive:",
    same_video_only
)


# ============================================================
# 8. Raw person_id overlap between source partitions
# ============================================================

train_ids = set(
    train["person_id"]
)

eval_ids = set(
    eval_df["person_id"]
)

overlap = sorted(
    train_ids
    & eval_ids
)


print("\nRAW PERSON_ID TRAIN / EVAL OVERLAP")
print("-" * 90)

print(
    "Overlap count:",
    len(overlap)
)

print(
    "Overlap person_ids:",
    overlap
)


for person_id in overlap:

    x = df[
        df["person_id"]
        == person_id
    ]

    print(
        f"\nperson_id={person_id}"
    )

    print(
        x[
            [
                "split",
                "video_id",
                "filename",
            ]
        ]
        .to_string(index=False)
    )


print("\nSaved corrected manifest:")
print(MANIFEST_V2)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
