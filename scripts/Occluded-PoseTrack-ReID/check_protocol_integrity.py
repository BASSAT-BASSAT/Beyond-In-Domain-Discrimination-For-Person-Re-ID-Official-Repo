from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd


MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/Occluded-PoseTrack-ReID"
    / "occluded_posetrack_reid_manifest.csv"
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

train = df[df["split"] == "train"].copy()
query = df[df["split"] == "query"].copy()
gallery = df[df["split"] == "gallery"].copy()
eval_df = df[df["split"].isin(["query", "gallery"])].copy()


print("=" * 90)
print("OCCLUDED-POSETRACK-REID PROTOCOL INTEGRITY")
print("=" * 90)


# ============================================================
# 1. Train / Eval PID overlap
# ============================================================

train_pids = set(train["pid"])
eval_pids = set(eval_df["pid"])

overlap = sorted(train_pids & eval_pids)

print("\nTRAIN / EVAL PID OVERLAP")
print("-" * 90)

print("Overlap PID count:", len(overlap))
print("Overlap PIDs      :", overlap)

for pid in overlap:

    print("\n" + "=" * 70)
    print(f"PID {pid}")
    print("=" * 70)

    rows = df[df["pid"] == pid]

    for split_name in ["train", "query", "gallery"]:

        x = rows[rows["split"] == split_name]

        if len(x) == 0:
            continue

        print(f"\n{split_name.upper()}")
        print("Images :", len(x))
        print(
            "Videos :",
            sorted(
                x["video_id"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        print("Examples:")

        for _, row in x.head(10).iterrows():

            print(
                f"  video={row['video_id']} | "
                f"image_id={row['posetrack_image_id']} | "
                f"{row['filename']}"
            )


# ============================================================
# 2. Query → Gallery validity
#
# ReID valid positive:
# same PID + different logical camid/video_id
# ============================================================

gallery_by_pid = {
    pid: group
    for pid, group in gallery.groupby("pid")
}

queries_with_any_positive = 0
queries_with_cross_video_positive = 0
queries_with_same_video_only = 0
queries_with_no_gallery_pid = 0

same_video_only_examples = []
no_gallery_examples = []

valid_positive_counts = []


for _, q in query.iterrows():

    pid = q["pid"]
    q_video = q["video_id"]

    g = gallery_by_pid.get(pid)

    if g is None or len(g) == 0:

        queries_with_no_gallery_pid += 1

        if len(no_gallery_examples) < 10:
            no_gallery_examples.append(
                (
                    q["filename"],
                    pid,
                    q_video,
                )
            )

        valid_positive_counts.append(0)
        continue


    queries_with_any_positive += 1

    valid = g[
        g["video_id"] != q_video
    ]

    valid_count = len(valid)

    valid_positive_counts.append(
        valid_count
    )

    if valid_count > 0:

        queries_with_cross_video_positive += 1

    else:

        queries_with_same_video_only += 1

        if len(same_video_only_examples) < 20:

            same_video_only_examples.append(
                {
                    "query_file":
                        q["filename"],

                    "pid":
                        pid,

                    "query_video":
                        q_video,

                    "gallery_videos":
                        sorted(
                            g["video_id"]
                            .unique()
                            .tolist()
                        ),

                    "gallery_images":
                        len(g),
                }
            )


print("\n\nQUERY / GALLERY VALID POSITIVES")
print("-" * 90)

print(
    "Total queries                     :",
    len(query)
)

print(
    "Queries with same-PID gallery     :",
    queries_with_any_positive
)

print(
    "Queries with cross-video positive :",
    queries_with_cross_video_positive
)

print(
    "Queries with same-video only      :",
    queries_with_same_video_only
)

print(
    "Queries with no gallery PID       :",
    queries_with_no_gallery_pid
)


if valid_positive_counts:

    s = pd.Series(
        valid_positive_counts
    )

    print("\nValid cross-video positives per query:")

    print(
        "Min    :",
        int(s.min())
    )

    print(
        "Median :",
        float(s.median())
    )

    print(
        "Mean   :",
        round(
            float(s.mean()),
            2
        )
    )

    print(
        "Max    :",
        int(s.max())
    )


if same_video_only_examples:

    print("\nSame-video-only query examples:")

    for x in same_video_only_examples:

        print(
            f"  PID={x['pid']} | "
            f"query_video={x['query_video']} | "
            f"gallery_videos={x['gallery_videos']} | "
            f"gallery_images={x['gallery_images']} | "
            f"{x['query_file']}"
        )


if no_gallery_examples:

    print("\nNo-gallery-PID examples:")

    for x in no_gallery_examples:

        print(
            " ",
            x
        )


# ============================================================
# 3. PID / Video structure
# ============================================================

def pid_video_distribution(name, data):

    pid_video_counts = (
        data.groupby("pid")["video_id"]
        .nunique()
    )

    dist = Counter(
        pid_video_counts.tolist()
    )

    print(
        f"\n{name} — videos per PID"
    )

    print("-" * 90)

    for n_videos, n_pids in sorted(
        dist.items()
    ):

        print(
            f"{n_videos:>3} video(s) "
            f"→ {n_pids:>5} PID(s)"
        )

    print(
        "PIDs in >1 video:",
        int(
            (pid_video_counts > 1)
            .sum()
        )
    )


pid_video_distribution(
    "TRAIN",
    train
)

pid_video_distribution(
    "QUERY+GALLERY",
    eval_df
)


# ============================================================
# 4. Gallery-only identities
# ============================================================

query_pids = set(
    query["pid"]
)

gallery_pids = set(
    gallery["pid"]
)

gallery_only = sorted(
    gallery_pids
    - query_pids
)


print("\nGALLERY-ONLY IDENTITIES")
print("-" * 90)

print(
    "Gallery-only PID count:",
    len(gallery_only)
)

if gallery_only:

    gallery_only_df = gallery[
        gallery["pid"].isin(
            gallery_only
        )
    ]

    print(
        "Gallery-only images   :",
        len(gallery_only_df)
    )

    print(
        "Gallery-only videos   :",
        gallery_only_df[
            "video_id"
        ].nunique()
    )

    print(
        "Example PIDs          :",
        gallery_only[:20]
    )


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
