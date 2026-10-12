from pathlib import Path
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRW"
    / "prw_manifest.csv"
)

OUT = (
    BASE
    / "manifests/PRW"
    / "prw_audit_flags.csv"
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)


# Only official supervised-source rows matter here.
source_mask = (
    (df["frame_split"] == "train")
    &
    (df["identity_role"] == "train_id")
)


df["flag_same_pid_distinct_conflict"] = False
df["flag_exact_annotation_duplicate"] = False
df["flag_overlapping_annotation_duplicate"] = False
df["flag_out_of_bounds"] = (
    ~df["fully_inside_image"]
)

df["recommended_training_exclusion"] = False
df["exclusion_reason"] = ""


# ============================================================
# 1. DISTINCT SAME-PID / SAME-FRAME CONFLICTS
# Conservative rule:
# exclude BOTH boxes; do not manually relabel.
# ============================================================

distinct_conflicts = [
    (29,  "c1s1_005401"),
    (96,  "c1s1_019476"),
    (173, "c1s2_019916"),
    (195, "c1s1_047526"),
    (233, "c3s1_069842"),
    (297, "c1s2_024091"),
    (357, "c3s1_096067"),
    (358, "c2s1_095446"),
]


for pid, frame in distinct_conflicts:

    mask = (
        source_mask
        &
        (df["person_id"] == pid)
        &
        (df["frame_stem"] == frame)
    )

    count = int(
        mask.sum()
    )

    if count != 2:
        raise RuntimeError(
            f"Expected 2 rows for "
            f"PID={pid}, frame={frame}, "
            f"found {count}"
        )

    df.loc[
        mask,
        "flag_same_pid_distinct_conflict"
    ] = True

    df.loc[
        mask,
        "recommended_training_exclusion"
    ] = True

    df.loc[
        mask,
        "exclusion_reason"
    ] = "same_pid_same_frame_distinct_conflict"


# ============================================================
# 2. EXACT DUPLICATE
#
# PID 7 / c6s1_000476:
# box_index 2 and 3 are exactly identical.
#
# Keep lower box_index, exclude the extra row.
# ============================================================

mask = (
    source_mask
    &
    (df["person_id"] == 7)
    &
    (df["frame_stem"] == "c6s1_000476")
)

g = df[
    mask
].sort_values(
    "box_index"
)

if len(g) != 2:
    raise RuntimeError(
        "Expected 2 rows for exact duplicate "
        "PID 7 / c6s1_000476"
    )


# Flag both as belonging to a duplicate group.
df.loc[
    g.index,
    "flag_exact_annotation_duplicate"
] = True


# Keep first/lower box_index.
exclude_idx = (
    g.index[1]
)


df.loc[
    exclude_idx,
    "recommended_training_exclusion"
] = True

df.loc[
    exclude_idx,
    "exclusion_reason"
] = "extra_exact_annotation_duplicate"


# ============================================================
# 3. OVERLAPPING DUPLICATE
#
# PID 173 / c2s1_091696
#
# Both boxes describe the same person.
# Keep the larger-area annotation deterministically.
# ============================================================

mask = (
    source_mask
    &
    (df["person_id"] == 173)
    &
    (df["frame_stem"] == "c2s1_091696")
)

g = df[
    mask
].copy()

if len(g) != 2:
    raise RuntimeError(
        "Expected 2 rows for overlapping duplicate "
        "PID 173 / c2s1_091696"
    )


df.loc[
    g.index,
    "flag_overlapping_annotation_duplicate"
] = True


g["area"] = (
    g["w"]
    * g["h"]
)


# Keep larger box.
exclude_idx = (
    g.sort_values(
        "area",
        ascending=False
    )
    .index[1]
)


df.loc[
    exclude_idx,
    "recommended_training_exclusion"
] = True

df.loc[
    exclude_idx,
    "exclusion_reason"
] = "extra_overlapping_annotation_duplicate"


# ============================================================
# REPORT
# ============================================================

flags = df[
    source_mask
    &
    (
        df[
            [
                "flag_same_pid_distinct_conflict",
                "flag_exact_annotation_duplicate",
                "flag_overlapping_annotation_duplicate",
                "flag_out_of_bounds",
                "recommended_training_exclusion",
            ]
        ]
        .any(axis=1)
    )
].copy()


flags.to_csv(
    OUT,
    index=False
)


print("=" * 95)
print("PRW AUDIT FLAGS")
print("=" * 95)


print(
    "Official supervised source rows:",
    int(
        source_mask.sum()
    )
)

print(
    "\nDistinct-conflict rows:",
    int(
        df.loc[
            source_mask,
            "flag_same_pid_distinct_conflict"
        ].sum()
    )
)

print(
    "Exact-duplicate group rows:",
    int(
        df.loc[
            source_mask,
            "flag_exact_annotation_duplicate"
        ].sum()
    )
)

print(
    "Overlapping-duplicate group rows:",
    int(
        df.loc[
            source_mask,
            "flag_overlapping_annotation_duplicate"
        ].sum()
    )
)

print(
    "Out-of-bounds source rows:",
    int(
        df.loc[
            source_mask,
            "flag_out_of_bounds"
        ].sum()
    )
)

print(
    "\nRecommended hard exclusions:",
    int(
        df.loc[
            source_mask,
            "recommended_training_exclusion"
        ].sum()
    )
)


clean_rows = (
    source_mask
    &
    ~df[
        "recommended_training_exclusion"
    ]
)


print(
    "Clean source candidate rows:",
    int(
        clean_rows.sum()
    )
)


print(
    "\nExclusion reasons:"
)

print(
    df.loc[
        source_mask
        &
        df[
            "recommended_training_exclusion"
        ],
        "exclusion_reason"
    ]
    .value_counts()
)


print(
    "\nNOTE:"
)

print(
    "Out-of-bounds boxes are FLAGS ONLY."
)

print(
    "They are NOT excluded; clip them to "
    "image boundaries when generating crops."
)


print(
    "\nSaved:",
    OUT
)

print("\nDONE")
