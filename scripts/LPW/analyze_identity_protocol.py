from pathlib import Path
from collections import Counter

import pandas as pd


MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/LPW"
    / "lpw_manifest.csv"
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "person_local_id": str,
        "global_person_id": str,
        "tracklet_id": str,
    },
)


train = df[
    df["official_role"] == "train"
].copy()

query = df[
    df["official_role"] == "query"
].copy()

gallery = df[
    df["official_role"] == "gallery"
].copy()


print("=" * 90)
print("LPW IDENTITY / TRACKLET / PROTOCOL AUDIT")
print("=" * 90)


# ============================================================
# TRAIN SUMMARY
# ============================================================

print("\nTRAIN SUMMARY")
print("-" * 90)

print(
    "Images:",
    len(train)
)

print(
    "Identities:",
    train["global_person_id"].nunique()
)

print(
    "Tracklets:",
    train["tracklet_id"].nunique()
)


# ============================================================
# IMAGES PER TRAIN ID
# ============================================================

train_id_sizes = (
    train.groupby(
        "global_person_id"
    )
    .size()
)


print("\nTRAIN IMAGES PER ID")
print("-" * 90)

print(
    train_id_sizes
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


# ============================================================
# TRACKLETS / VIEWS PER TRAIN ID
# ============================================================

train_tracklets_per_id = (
    train.groupby(
        "global_person_id"
    )["tracklet_id"]
    .nunique()
)


print("\nTRAIN TRACKLETS PER ID")
print("-" * 90)

print(
    train_tracklets_per_id
    .value_counts()
    .sort_index()
)


print(
    "\nMean:",
    round(
        float(
            train_tracklets_per_id.mean()
        ),
        2,
    )
)


# ============================================================
# TRACKLET SIZE
# ============================================================

tracklet_sizes = (
    train.groupby(
        "tracklet_id"
    )
    .size()
)


print("\nTRAIN IMAGES PER TRACKLET")
print("-" * 90)

print(
    tracklet_sizes
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


# ============================================================
# TRAIN SCENE BALANCE
# ============================================================

print("\nTRAIN SCENE DISTRIBUTION")
print("-" * 90)

for scene in [
    "scen2",
    "scen3",
]:

    x = train[
        train["scene"] == scene
    ]

    image_pct = (
        len(x)
        / len(train)
        * 100
    )

    id_count = (
        x[
            "global_person_id"
        ].nunique()
    )

    id_pct = (
        id_count
        / train[
            "global_person_id"
        ].nunique()
        * 100
    )

    print(
        f"\n{scene}"
    )

    print(
        "Images:",
        len(x),
        f"({image_pct:.2f}%)"
    )

    print(
        "IDs:",
        id_count,
        f"({id_pct:.2f}%)"
    )

    print(
        "Tracklets:",
        x[
            "tracklet_id"
        ].nunique()
    )


# ============================================================
# QUERY / GALLERY SUMMARY
# ============================================================

print("\nQUERY / GALLERY SUMMARY")
print("-" * 90)

print(
    "Query images:",
    len(query)
)

print(
    "Query IDs:",
    query[
        "global_person_id"
    ].nunique()
)

print(
    "Gallery images:",
    len(gallery)
)

print(
    "Gallery IDs:",
    gallery[
        "global_person_id"
    ].nunique()
)


# ============================================================
# QUERY IMAGES PER ID
# ============================================================

query_per_id = (
    query.groupby(
        "global_person_id"
    )
    .size()
)


print("\nQUERY IMAGES PER ID")
print("-" * 90)

print(
    query_per_id
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


# ============================================================
# GALLERY IMAGES PER ID
# ============================================================

gallery_per_id = (
    gallery.groupby(
        "global_person_id"
    )
    .size()
)


print("\nGALLERY IMAGES PER ID")
print("-" * 90)

print(
    gallery_per_id
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


# ============================================================
# GALLERY VIEW COVERAGE
# ============================================================

gallery_views = (
    gallery.groupby(
        "global_person_id"
    )["view"]
    .agg(
        lambda x:
        tuple(
            sorted(
                set(x)
            )
        )
    )
)


print("\nGALLERY VIEW COVERAGE")
print("-" * 90)

coverage = Counter(
    gallery_views.tolist()
)

for views, count in sorted(
    coverage.items(),
    key=lambda x: str(x[0])
):

    print(
        f"{views}: {count} IDs"
    )


print(
    "\nIDs with both view1 and view3:",
    int(
        (
            gallery.groupby(
                "global_person_id"
            )["view"]
            .nunique()
            == 2
        ).sum()
    )
)


# ============================================================
# OFFICIAL QUERY VALIDITY
# ============================================================

gallery_count_map = (
    gallery.groupby(
        "global_person_id"
    )
    .size()
)


query[
    "gallery_positives"
] = query[
    "global_person_id"
].map(
    gallery_count_map
)


missing_positive = (
    query[
        "gallery_positives"
    ]
    .isna()
)


print("\nOFFICIAL QUERY VALIDITY")
print("-" * 90)

print(
    "Total queries:",
    len(query)
)

print(
    "Queries with zero gallery positive:",
    int(
        missing_positive.sum()
    )
)


valid_positive_counts = (
    query.loc[
        ~missing_positive,
        "gallery_positives"
    ]
)


print(
    "\nGallery positives per query:"
)

print(
    valid_positive_counts
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


# ============================================================
# POSITIVE VIEW OPTIONS
# ============================================================

view1_ids = set(
    gallery.loc[
        gallery["view"]
        == "view1",
        "global_person_id"
    ]
)

view3_ids = set(
    gallery.loc[
        gallery["view"]
        == "view3",
        "global_person_id"
    ]
)

query_ids = set(
    query[
        "global_person_id"
    ]
)


print("\nQUERY POSITIVE VIEW OPTIONS")
print("-" * 90)

print(
    "Query IDs with View1 positive:",
    len(
        query_ids
        & view1_ids
    )
)

print(
    "Query IDs with View3 positive:",
    len(
        query_ids
        & view3_ids
    )
)

print(
    "Query IDs with BOTH:",
    len(
        query_ids
        & view1_ids
        & view3_ids
    )
)

print(
    "View1 only:",
    len(
        (
            query_ids
            & view1_ids
        )
        - view3_ids
    )
)

print(
    "View3 only:",
    len(
        (
            query_ids
            & view3_ids
        )
        - view1_ids
    )
)

print(
    "Neither:",
    len(
        query_ids
        - (
            view1_ids
            | view3_ids
        )
    )
)


# ============================================================
# IMAGE NUMBER CHARACTERISTICS
# ============================================================

print("\nIMAGE-NUMBER CHARACTERISTICS")
print("-" * 90)

print(
    "Min:",
    int(
        df[
            "image_number"
        ].min()
    )
)

print(
    "Max:",
    int(
        df[
            "image_number"
        ].max()
    )
)

print(
    "Unique numeric values:",
    df[
        "image_number"
    ].nunique()
)


# Groups sharing same numeric image number
same_number_groups = (
    df.groupby(
        [
            "scene",
            "view",
            "image_number",
        ]
    )
    .size()
)


print(
    "\n(scene, view, image_number)"
    " groups with >1 person crop:",
    int(
        (
            same_number_groups > 1
        ).sum()
    )
)

print(
    "Maximum crops sharing one"
    " (scene, view, image_number):",
    int(
        same_number_groups.max()
    )
)


print("\nDONE")
