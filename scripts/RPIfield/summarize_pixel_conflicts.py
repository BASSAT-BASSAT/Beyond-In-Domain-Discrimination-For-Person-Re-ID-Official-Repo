from pathlib import Path
from collections import Counter

import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

ROWS = (
    BASE
    / "reports/RPIfield"
    / "pixel_duplicate_rows.csv"
)

GROUPS = (
    BASE
    / "reports/RPIfield"
    / "pixel_duplicate_groups.csv"
)

OUT = (
    BASE
    / "reports/RPIfield"
    / "pixel_conflict_summary.csv"
)


rows = pd.read_csv(
    ROWS,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
        "identity_type": str,
    },
    low_memory=False,
)

groups = pd.read_csv(
    GROUPS,
    low_memory=False,
)


print("=" * 90)
print("RPIFIELD PIXEL CONFLICT SUMMARY")
print("=" * 90)


print("\nGROUP SIZE DISTRIBUTION")
print("-" * 90)

print(
    groups["images"]
    .value_counts()
    .sort_index()
)


cross_group_ids = set(
    groups.loc[
        groups["cross_person"],
        "pixel_duplicate_group",
    ]
)


cross = rows[
    rows[
        "pixel_duplicate_group"
    ].isin(
        cross_group_ids
    )
].copy()


print("\nCROSS-PERSON CONFLICTS")
print("-" * 90)

print(
    "Groups:",
    len(
        cross_group_ids
    )
)

print(
    "Rows involved:",
    len(
        cross
    )
)

print(
    "Unique persons:",
    cross[
        "person_id"
    ].nunique()
)

print(
    "Unique cameras:",
    cross[
        "camera_id"
    ].nunique()
)

print(
    "Cameras:",
    sorted(
        cross[
            "camera_id"
        ].unique()
    )
)


print("\nROWS BY IDENTITY TYPE")
print("-" * 90)

print(
    cross[
        "identity_type"
    ].value_counts()
)


# ============================================================
# Group-level identity-type classification
# ============================================================

class_rows = []


for gid, x in cross.groupby(
    "pixel_duplicate_group"
):

    types = sorted(
        x[
            "identity_type"
        ].unique()
    )

    persons = sorted(
        x[
            "person_id"
        ].unique()
    )


    if types == [
        "distractor"
    ]:

        category = (
            "distractor_vs_distractor"
        )

    elif types == [
        "known_actor"
    ]:

        category = (
            "known_vs_known"
        )

    else:

        category = (
            "known_vs_distractor"
        )


    class_rows.append(
        {
            "pixel_duplicate_group":
                gid,

            "category":
                category,

            "images":
                len(x),

            "persons":
                x[
                    "person_id"
                ].nunique(),

            "camera_id":
                x[
                    "camera_id"
                ].iloc[0],

            "frame_ids":
                ",".join(
                    map(
                        str,
                        sorted(
                            x[
                                "frame_id"
                            ].unique()
                        )
                    )
                ),

            "person_ids":
                ",".join(
                    map(
                        str,
                        persons
                    )
                ),
        }
    )


summary = pd.DataFrame(
    class_rows
)


print("\nGROUPS BY CONFLICT TYPE")
print("-" * 90)

print(
    summary[
        "category"
    ].value_counts()
)


print("\nGROUPS BY CAMERA")
print("-" * 90)

print(
    summary[
        "camera_id"
    ]
    .value_counts()
    .sort_index()
)


print("\nAFFECTED KNOWN ACTORS")
print("-" * 90)

known_pids = sorted(
    cross.loc[
        cross[
            "identity_type"
        ]
        == "known_actor",
        "person_id",
    ].unique()
)

print(
    "Count:",
    len(
        known_pids
    )
)

print(
    known_pids
)


print("\nTOP PERSON PAIRS")
print("-" * 90)

pairs = Counter()


for _, x in cross.groupby(
    "pixel_duplicate_group"
):

    pids = tuple(
        sorted(
            x[
                "person_id"
            ].unique()
        )
    )

    pairs[
        pids
    ] += 1


for pair, count in (
    pairs
    .most_common(
        30
    )
):

    print(
        f"{pair}: "
        f"{count} groups"
    )


summary.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)

print("\nDONE")
