from pathlib import Path
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

PIXEL_ROWS = (
    BASE
    / "reports/RPIfield"
    / "pixel_duplicate_rows.csv"
)

PIXEL_GROUPS = (
    BASE
    / "reports/RPIfield"
    / "pixel_duplicate_groups.csv"
)

GEOMETRY = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_geometry.csv"
)

OUT = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_audit_flags.csv"
)


rows = pd.read_csv(
    PIXEL_ROWS,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
    },
    low_memory=False,
)

groups = pd.read_csv(
    PIXEL_GROUPS,
    low_memory=False,
)

geo = pd.read_csv(
    GEOMETRY,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
    },
    low_memory=False,
)


group_info = groups[
    [
        "pixel_duplicate_group",
        "cross_person",
        "cross_sequence",
        "cross_camera",
        "cross_frame",
    ]
]


flags = rows.merge(
    group_info,
    on="pixel_duplicate_group",
    how="left",
)


flags["pixel_duplicate"] = True

flags["pixel_cross_person_conflict"] = (
    flags["cross_person"]
)

flags["pixel_same_person_redundancy"] = (
    ~flags["cross_person"]
)


# ============================================================
# Geometry anomaly
# ============================================================

geometry_bad = geo[
    geo["width"] >= geo["height"]
][
    [
        "path",
        "camera_id",
        "person_id",
        "appearance_label",
        "sequence_id",
        "frame_id",
    ]
].copy()


geometry_bad["pixel_duplicate"] = False
geometry_bad["pixel_duplicate_group"] = pd.NA
geometry_bad["cross_person"] = False
geometry_bad["cross_sequence"] = False
geometry_bad["cross_camera"] = False
geometry_bad["cross_frame"] = False

geometry_bad[
    "pixel_cross_person_conflict"
] = False

geometry_bad[
    "pixel_same_person_redundancy"
] = False


flags["geometry_anomaly"] = False
geometry_bad["geometry_anomaly"] = True


combined = pd.concat(
    [
        flags,
        geometry_bad,
    ],
    ignore_index=True,
    sort=False,
)


combined = combined.drop_duplicates(
    subset=["path"],
    keep="first",
)


# If geometry row overlaps another flag,
# restore geometry flag using path membership.

geometry_paths = set(
    geometry_bad["path"]
)

combined["geometry_anomaly"] = (
    combined["path"]
    .isin(geometry_paths)
)


# ============================================================
# Recommended handling
# ============================================================

combined["recommended_train_exclude"] = (
    combined[
        "pixel_cross_person_conflict"
    ].fillna(False)
    |
    combined[
        "geometry_anomaly"
    ].fillna(False)
)


def reason(row):

    reasons = []

    if bool(
        row.get(
            "pixel_cross_person_conflict",
            False
        )
    ):
        reasons.append(
            "cross_person_pixel_conflict"
        )

    if bool(
        row.get(
            "pixel_same_person_redundancy",
            False
        )
    ):
        reasons.append(
            "same_person_pixel_redundancy"
        )

    if bool(
        row.get(
            "geometry_anomaly",
            False
        )
    ):
        reasons.append(
            "landscape_geometry_anomaly"
        )

    return "|".join(
        reasons
    )


combined["audit_reason"] = (
    combined.apply(
        reason,
        axis=1
    )
)


columns = [
    "path",
    "camera_id",
    "person_id",
    "identity_type",
    "appearance_label",
    "sequence_id",
    "frame_id",
    "pixel_duplicate_group",
    "pixel_duplicate",
    "pixel_cross_person_conflict",
    "pixel_same_person_redundancy",
    "geometry_anomaly",
    "recommended_train_exclude",
    "audit_reason",
]


for column in columns:

    if column not in combined:
        combined[column] = pd.NA


combined = combined[
    columns
].sort_values(
    [
        "camera_id",
        "person_id",
        "frame_id",
    ]
)


combined.to_csv(
    OUT,
    index=False
)


print("=" * 90)
print("RPIFIELD AUDIT FLAGS")
print("=" * 90)

print(
    "Flagged rows:",
    len(combined)
)

print(
    "Cross-person conflict rows:",
    int(
        combined[
            "pixel_cross_person_conflict"
        ]
        .fillna(False)
        .sum()
    )
)

print(
    "Same-person redundancy rows:",
    int(
        combined[
            "pixel_same_person_redundancy"
        ]
        .fillna(False)
        .sum()
    )
)

print(
    "Geometry anomaly rows:",
    int(
        combined[
            "geometry_anomaly"
        ]
        .fillna(False)
        .sum()
    )
)

print(
    "Recommended training exclusion:",
    int(
        combined[
            "recommended_train_exclude"
        ]
        .fillna(False)
        .sum()
    )
)


print("\nSaved:")
print(OUT)

print("\nDONE")
