from pathlib import Path
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/ORBench/orbench_rgb_quality_manifest.csv"
)

OUT = (
    BASE
    / "manifests/ORBench/orbench_rgb_audit_flags.csv"
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)


# ============================================================
# FLAGS
# ============================================================

df["flag_landscape"] = (
    df["width"]
    >
    df["height"]
)

df["flag_square"] = (
    df["width"]
    ==
    df["height"]
)

df["flag_extreme_narrow"] = (
    df["aspect_ratio"]
    <
    0.18
)

df["flag_exact_duplicate"] = False

df["recommended_training_exclusion"] = False

df["exclusion_reason"] = ""


def add_reason(index, reason):

    current = str(
        df.at[
            index,
            "exclusion_reason"
        ]
    )

    if (
        current
        and
        current != "nan"
    ):

        df.at[
            index,
            "exclusion_reason"
        ] = (
            current
            + "|"
            + reason
        )

    else:

        df.at[
            index,
            "exclusion_reason"
        ] = reason


# ============================================================
# 1. LANDSCAPE TRAINING CROPS
#
# Visual audit showed these are background-heavy /
# malformed pedestrian crops.
#
# Official test remains untouched.
# ============================================================

landscape_train = (
    (df["split"] == "train")
    &
    df["flag_landscape"]
)


df.loc[
    landscape_train,
    "recommended_training_exclusion"
] = True


for idx in df[
    landscape_train
].index:

    add_reason(
        idx,
        "landscape_geometry_outlier"
    )


# ============================================================
# 2. EXACT PIXEL DUPLICATES
#
# No cross-split or cross-ID duplicate groups were found.
#
# For TRAIN:
#   keep one deterministic representative per group.
#
# For TEST:
#   flag only; preserve official gallery.
# ============================================================

dup_groups = (
    df.groupby(
        "pixel_sha256"
    )
    .filter(
        lambda g:
            len(g) > 1
    )
)


train_dup_groups = 0
test_dup_groups = 0

train_dup_extra_exclusions = 0


for hash_value, g in dup_groups.groupby(
    "pixel_sha256"
):

    df.loc[
        g.index,
        "flag_exact_duplicate"
    ] = True


    train_g = (
        g[
            g["split"]
            == "train"
        ]
        .sort_values(
            "file_path"
        )
    )


    test_g = (
        g[
            g["split"]
            == "test_gallery"
        ]
    )


    if len(train_g) > 1:

        train_dup_groups += 1

        # Keep first deterministic file.
        for idx in train_g.index[1:]:

            df.at[
                idx,
                "recommended_training_exclusion"
            ] = True

            add_reason(
                idx,
                "extra_exact_duplicate"
            )

            train_dup_extra_exclusions += 1


    if len(test_g) > 1:

        test_dup_groups += 1


# ============================================================
# SUMMARY
# ============================================================

train_mask = (
    df["split"]
    == "train"
)

test_mask = (
    df["split"]
    == "test_gallery"
)


clean_train = (
    train_mask
    &
    ~df[
        "recommended_training_exclusion"
    ]
)


print("=" * 100)
print("ORBENCH RGB AUDIT FLAGS")
print("=" * 100)


print(
    "Official RGB train rows:",
    int(
        train_mask.sum()
    )
)

print(
    "Official RGB test rows:",
    int(
        test_mask.sum()
    )
)


print("\nGEOMETRY")

print(
    "Train landscape:",
    int(
        (
            train_mask
            &
            df["flag_landscape"]
        ).sum()
    )
)

print(
    "Test landscape:",
    int(
        (
            test_mask
            &
            df["flag_landscape"]
        ).sum()
    )
)

print(
    "Train square:",
    int(
        (
            train_mask
            &
            df["flag_square"]
        ).sum()
    )
)

print(
    "Test square:",
    int(
        (
            test_mask
            &
            df["flag_square"]
        ).sum()
    )
)

print(
    "Train extreme-narrow:",
    int(
        (
            train_mask
            &
            df["flag_extreme_narrow"]
        ).sum()
    )
)

print(
    "Test extreme-narrow:",
    int(
        (
            test_mask
            &
            df["flag_extreme_narrow"]
        ).sum()
    )
)


print("\nEXACT DUPLICATES")

print(
    "Train duplicate groups:",
    train_dup_groups
)

print(
    "Test duplicate groups:",
    test_dup_groups
)

print(
    "Extra train duplicate rows excluded:",
    train_dup_extra_exclusions
)


print("\nFINAL SOURCE")

hard_exclusions = int(
    (
        train_mask
        &
        df[
            "recommended_training_exclusion"
        ]
    ).sum()
)


print(
    "Recommended hard exclusions:",
    hard_exclusions
)

print(
    "Clean RGB source rows:",
    int(
        clean_train.sum()
    )
)


print("\nExclusion reasons:")

print(
    df.loc[
        train_mask
        &
        df[
            "recommended_training_exclusion"
        ],
        "exclusion_reason",
    ]
    .value_counts()
)


print(
    "\nNOTE:"
)

print(
    "Square and extreme-narrow images are FLAGS ONLY."
)

print(
    "Test/gallery images are never excluded."
)


# ============================================================
# SAVE FLAGGED ROWS
# ============================================================

flagged = df[
    df[
        [
            "flag_landscape",
            "flag_square",
            "flag_extreme_narrow",
            "flag_exact_duplicate",
            "recommended_training_exclusion",
        ]
    ]
    .any(
        axis=1
    )
].copy()


flagged.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)

print("\nDONE")
