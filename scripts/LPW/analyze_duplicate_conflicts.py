from pathlib import Path
from itertools import combinations
from collections import Counter

import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

DUP_ROWS = (
    BASE
    / "reports/LPW"
    / "exact_duplicate_rows.csv"
)

OUT_DIR = (
    BASE
    / "reports/LPW"
)

PAIR_OUT = (
    OUT_DIR
    / "duplicate_identity_pairs.csv"
)


df = pd.read_csv(
    DUP_ROWS,
    dtype={
        "global_person_id": str,
        "tracklet_id": str,
    },
)


print("=" * 90)
print("LPW CROSS-IDENTITY DUPLICATE CONFLICT ANALYSIS")
print("=" * 90)


# ============================================================
# Basic role / scene distribution
# ============================================================

print("\nDUPLICATE GROUPS BY ROLE")
print("-" * 90)

group_role = (
    df.groupby(
        "duplicate_group"
    )["official_role"]
    .first()
    .value_counts()
)

print(group_role)


print("\nDUPLICATE GROUPS BY SCENE / VIEW")
print("-" * 90)

group_scene_view = (
    df.groupby(
        "duplicate_group"
    )
    .first()
    .groupby(
        [
            "scene",
            "view",
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(group_scene_view)


# ============================================================
# Affected identities
# ============================================================

affected_ids = sorted(
    df[
        "global_person_id"
    ].unique()
)

print("\nAFFECTED IDENTITIES")
print("-" * 90)

print(
    "Affected IDs:",
    len(
        affected_ids
    )
)

print(
    "\nIDs:"
)

for pid in affected_ids:
    print(pid)


# ============================================================
# Identity-pair concentration
# ============================================================

pair_counter = Counter()


for group_id, x in df.groupby(
    "duplicate_group"
):

    persons = sorted(
        x[
            "global_person_id"
        ]
        .unique()
    )

    for a, b in combinations(
        persons,
        2,
    ):

        pair_counter[
            (
                a,
                b,
            )
        ] += 1


pair_rows = []

for (
    pid_a,
    pid_b
), count in pair_counter.most_common():

    pair_rows.append(
        {
            "person_a":
                pid_a,

            "person_b":
                pid_b,

            "duplicate_groups":
                count,
        }
    )


pairs = pd.DataFrame(
    pair_rows
)

pairs.to_csv(
    PAIR_OUT,
    index=False,
)


print("\nIDENTITY PAIRS")
print("-" * 90)

print(
    "Unique affected identity pairs:",
    len(
        pairs
    )
)

print(
    "\nTop pairs:"
)

print(
    pairs.head(
        30
    ).to_string(
        index=False
    )
)


# ============================================================
# Duplicates per affected identity
# ============================================================

dup_per_id = (
    df.groupby(
        "global_person_id"
    )[
        "duplicate_group"
    ]
    .nunique()
    .sort_values(
        ascending=False
    )
)


print("\nDUPLICATE GROUPS PER IDENTITY")
print("-" * 90)

print(
    dup_per_id.head(
        30
    )
)


# ============================================================
# Tracklet-level impact
# ============================================================

dup_per_tracklet = (
    df.groupby(
        "tracklet_id"
    )[
        "duplicate_group"
    ]
    .nunique()
    .sort_values(
        ascending=False
    )
)


print("\nAFFECTED TRACKLETS")
print("-" * 90)

print(
    "Affected tracklets:",
    len(
        dup_per_tracklet
    )
)

print(
    "\nTop affected tracklets:"
)

print(
    dup_per_tracklet.head(
        30
    )
)


# ============================================================
# Fraction of each affected tracklet duplicated
# ============================================================

MANIFEST = (
    BASE
    / "manifests/LPW"
    / "lpw_manifest.csv"
)

manifest = pd.read_csv(
    MANIFEST,
    dtype={
        "global_person_id": str,
        "tracklet_id": str,
    },
)


tracklet_sizes = (
    manifest.groupby(
        "tracklet_id"
    )
    .size()
)


tracklet_dup_rows = (
    df.groupby(
        "tracklet_id"
    )
    .size()
)


impact = pd.DataFrame(
    {
        "tracklet_images":
            tracklet_sizes,

        "duplicate_images":
            tracklet_dup_rows,
    }
).dropna()


impact[
    "duplicate_fraction"
] = (
    impact[
        "duplicate_images"
    ]
    /
    impact[
        "tracklet_images"
    ]
)


impact = impact.sort_values(
    "duplicate_fraction",
    ascending=False,
)


print("\nTRACKLET DUPLICATE FRACTION")
print("-" * 90)

print(
    impact.head(
        30
    )
    .round(
        {
            "duplicate_fraction":
                4
        }
    )
)


# ============================================================
# Do paired identities share additional non-duplicate images?
# ============================================================

print("\nPAIR SUMMARY")
print("-" * 90)

for _, row in pairs.head(
    20
).iterrows():

    a = row[
        "person_a"
    ]

    b = row[
        "person_b"
    ]

    a_rows = manifest[
        manifest[
            "global_person_id"
        ]
        == a
    ]

    b_rows = manifest[
        manifest[
            "global_person_id"
        ]
        == b
    ]

    print(
        f"\n{a} ↔ {b}"
    )

    print(
        "Duplicate exact images:",
        int(
            row[
                "duplicate_groups"
            ]
        )
    )

    print(
        f"{a} total images:",
        len(
            a_rows
        )
    )

    print(
        f"{b} total images:",
        len(
            b_rows
        )
    )

    print(
        f"{a} views:",
        sorted(
            a_rows[
                "view"
            ].unique()
        )
    )

    print(
        f"{b} views:",
        sorted(
            b_rows[
                "view"
            ].unique()
        )
    )


print("\nSaved:")
print(PAIR_OUT)

print("\nDONE")
