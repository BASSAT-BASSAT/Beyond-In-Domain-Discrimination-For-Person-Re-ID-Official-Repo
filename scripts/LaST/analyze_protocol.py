from pathlib import Path
import pandas as pd

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/LaST/last_manifest.csv"
)

df = pd.read_csv(MANIFEST, low_memory=False)

train = df[df["split"] == "train"].copy()

val_q = df[df["split"] == "val_query"].copy()
val_g = df[
    (df["split"] == "val_gallery")
    & (~df["is_distractor"])
].copy()

test_q = df[df["split"] == "test_query"].copy()
test_g = df[
    (df["split"] == "test_gallery")
    & (~df["is_distractor"])
].copy()


def pid_set(x):
    return set(x["pid"].astype(int))


def print_set_check(name, a, b):
    x = a & b
    print(f"{name:<32}: {len(x)}")
    if x:
        print("  examples:", sorted(x)[:20])


print("=" * 80)
print("LaST — SPLIT / QUERY-GALLERY PROTOCOL AUDIT")
print("=" * 80)

train_ids = pid_set(train)
val_q_ids = pid_set(val_q)
val_g_ids = pid_set(val_g)
test_q_ids = pid_set(test_q)
test_g_ids = pid_set(test_g)

print("\nIDENTITY COUNTS")
print("-" * 80)

print("Train IDs       :", len(train_ids))
print("Val Query IDs   :", len(val_q_ids))
print("Val Gallery IDs :", len(val_g_ids))
print("Test Query IDs  :", len(test_q_ids))
print("Test Gallery IDs:", len(test_g_ids))


# ---------------------------------------------------------
# Cross-split leakage
# ---------------------------------------------------------

print("\nCROSS-SPLIT PID OVERLAP")
print("-" * 80)

val_ids = val_q_ids | val_g_ids
test_ids = test_q_ids | test_g_ids

print_set_check(
    "Train ∩ Validation",
    train_ids,
    val_ids
)

print_set_check(
    "Train ∩ Test",
    train_ids,
    test_ids
)

print_set_check(
    "Validation ∩ Test",
    val_ids,
    test_ids
)


# ---------------------------------------------------------
# Query / Gallery identity validity
# ---------------------------------------------------------

def query_gallery_audit(label, q, g):

    q_ids = pid_set(q)
    g_ids = pid_set(g)

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print("\nImages:")
    print("Query  :", len(q))
    print("Gallery:", len(g))

    print("\nIDs:")
    print("Query  :", len(q_ids))
    print("Gallery:", len(g_ids))

    missing = q_ids - g_ids
    gallery_only = g_ids - q_ids

    print("\nQuery IDs missing from gallery:",
          len(missing))

    if missing:
        print("Missing IDs:",
              sorted(missing)[:50])

    print("\nGallery-only IDs:",
          len(gallery_only))

    if gallery_only:
        print("Gallery-only IDs:",
              sorted(gallery_only)[:50])

    # ---------------------------------------------
    # Exact filename overlap
    # ---------------------------------------------

    q_names = set(q["filename"])
    g_names = set(g["filename"])

    filename_overlap = q_names & g_names

    print(
        "\nExact Query/Gallery filename overlap:",
        len(filename_overlap)
    )

    # ---------------------------------------------
    # Same underlying crop metadata
    # ---------------------------------------------

    key_cols = [
        "pid",
        "video_id",
        "frame_id",
        "bbox_id",
    ]

    q_keys = set(
        map(tuple, q[key_cols].to_numpy())
    )

    g_keys = set(
        map(tuple, g[key_cols].to_numpy())
    )

    metadata_overlap = q_keys & g_keys

    print(
        "Same PID/video/frame/bbox overlap:",
        len(metadata_overlap)
    )

    # ---------------------------------------------
    # Query images per PID
    # ---------------------------------------------

    q_per_pid = q.groupby("pid").size()

    print("\nQuery images per PID:")
    print(q_per_pid.describe().round(2))

    print("\nQuery images/PID distribution:")
    print(
        q_per_pid.value_counts()
        .sort_index()
    )

    # ---------------------------------------------
    # Gallery positives per query PID
    # ---------------------------------------------

    g_per_pid = g.groupby("pid").size()

    q_tmp = q.copy()

    q_tmp["gallery_positives"] = (
        q_tmp["pid"]
        .map(g_per_pid)
        .fillna(0)
        .astype(int)
    )

    print("\nGallery positives per QUERY IMAGE:")
    print(
        q_tmp["gallery_positives"]
        .describe()
        .round(2)
    )

    print(
        "Queries with ZERO gallery positives:",
        int(
            (q_tmp["gallery_positives"] == 0)
            .sum()
        )
    )

    # ---------------------------------------------
    # Different-video positives
    # ---------------------------------------------

    gallery_by_pid = {
        pid: grp
        for pid, grp in g.groupby("pid")
    }

    different_video_counts = []

    same_frame_positive_queries = 0

    for row in q.itertuples():

        positives = gallery_by_pid.get(
            row.pid
        )

        if positives is None:
            different_video_counts.append(0)
            continue

        different_video = positives[
            positives["video_id"]
            != row.video_id
        ]

        different_video_counts.append(
            len(different_video)
        )

        same_frame = positives[
            (positives["video_id"] == row.video_id)
            & (positives["frame_id"] == row.frame_id)
        ]

        if len(same_frame) > 0:
            same_frame_positive_queries += 1

    q_tmp["different_video_positives"] = (
        different_video_counts
    )

    print("\nDifferent-video positives/query:")
    print(
        q_tmp["different_video_positives"]
        .describe()
        .round(2)
    )

    print(
        "Queries with ZERO different-video positives:",
        int(
            (
                q_tmp["different_video_positives"]
                == 0
            ).sum()
        )
    )

    print(
        "Queries sharing a frame with a gallery positive:",
        same_frame_positive_queries
    )


query_gallery_audit(
    "VALIDATION PROTOCOL",
    val_q,
    val_g,
)

query_gallery_audit(
    "TEST PROTOCOL",
    test_q,
    test_g,
)


# ---------------------------------------------------------
# Special PID anomaly investigation
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("KNOWN PID 005859 / FOLDER 005895 ANOMALY")
print("=" * 80)

special = df[
    (df["pid"].isin([5859, 5895]))
    |
    (df["folder_pid"].astype(str)
     .isin(["005859", "005895"]))
]

print(
    special[
        [
            "split",
            "pid",
            "folder_pid",
            "filename",
            "filename_anomaly",
            "folder_pid_mismatch",
        ]
    ]
    .sort_values(
        ["split", "pid", "filename"]
    )
    .to_string(index=False)
)
