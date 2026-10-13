from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/RPIfield/extracted/RPIfield_v0"
)

DATA = ROOT / "Data"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
)

OUT_DIR = (
    BASE
    / "manifests/RPIfield"
)

SEQ_OUT = (
    OUT_DIR
    / "rpifield_sequence_stats.csv"
)


df = pd.read_csv(
    MANIFEST
)


print("=" * 90)
print("RPIFIELD SEQUENCE / TEMPORAL AUDIT")
print("=" * 90)

print("Images:", len(df))
print("Sequences:", df["sequence_id"].nunique())
print("Persons:", df["person_id"].nunique())


# ============================================================
# 1. FILESYSTEM FOLDER RECONCILIATION
# ============================================================

print("\n" + "=" * 90)
print("FILESYSTEM / METADATA FOLDER RECONCILIATION")
print("=" * 90)


for cam in range(1, 13):

    camera_dir = (
        DATA
        / f"Cam_{cam}"
    )

    filesystem = {
        p.name
        for p in camera_dir.iterdir()
        if p.is_dir()
    }

    metadata = set(
        df.loc[
            df["camera_id"] == cam,
            "appearance_label"
        ]
        .astype(str)
        .unique()
    )

    extra = sorted(
        filesystem - metadata
    )

    missing = sorted(
        metadata - filesystem
    )

    print(
        f"Cam_{cam:<2} | "
        f"filesystem={len(filesystem):4d} | "
        f"metadata={len(metadata):4d} | "
        f"extra={len(extra):2d} | "
        f"missing={len(missing):2d}"
    )

    if extra:
        print(
            "   Extra folders:",
            extra
        )

    if missing:
        print(
            "   Missing folders:",
            missing
        )


# ============================================================
# 2. SEQUENCE STATISTICS
# ============================================================

rows = []
all_gaps = []

duplicate_frame_sequences = 0


for sequence_id, x in df.groupby(
    "sequence_id",
    sort=False
):

    frames = np.sort(
        x["frame_id"]
        .astype(np.int64)
        .to_numpy()
    )

    unique_frames = np.unique(
        frames
    )

    duplicate_frames = (
        len(frames)
        - len(unique_frames)
    )

    if duplicate_frames > 0:
        duplicate_frame_sequences += 1


    if len(unique_frames) > 1:

        gaps = np.diff(
            unique_frames
        )

        all_gaps.extend(
            gaps.tolist()
        )

        min_gap = int(
            gaps.min()
        )

        median_gap = float(
            np.median(
                gaps
            )
        )

        mean_gap = float(
            np.mean(
                gaps
            )
        )

        max_gap = int(
            gaps.max()
        )

    else:

        min_gap = np.nan
        median_gap = np.nan
        mean_gap = np.nan
        max_gap = np.nan


    first = x.iloc[0]

    rows.append(
        {
            "sequence_id":
                sequence_id,

            "camera_id":
                int(
                    first["camera_id"]
                ),

            "person_id":
                int(
                    first["person_id"]
                ),

            "identity_type":
                first["identity_type"],

            "appearance_label":
                str(
                    first[
                        "appearance_label"
                    ]
                ),

            "reappearance_index":
                int(
                    first[
                        "reappearance_index"
                    ]
                ),

            "images":
                len(frames),

            "unique_frames":
                len(unique_frames),

            "duplicate_frames":
                duplicate_frames,

            "first_frame":
                int(
                    unique_frames[0]
                ),

            "last_frame":
                int(
                    unique_frames[-1]
                ),

            "frame_span":
                int(
                    unique_frames[-1]
                    - unique_frames[0]
                ),

            "min_gap":
                min_gap,

            "median_gap":
                median_gap,

            "mean_gap":
                mean_gap,

            "max_gap":
                max_gap,
        }
    )


seq = pd.DataFrame(
    rows
)

seq.to_csv(
    SEQ_OUT,
    index=False
)


print("\n" + "=" * 90)
print("SEQUENCE SIZE")
print("=" * 90)

print(
    seq["images"]
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


print(
    "\nSequences with duplicate frame IDs:",
    duplicate_frame_sequences
)


# ============================================================
# 3. FRAME GAP DISTRIBUTION
# ============================================================

gaps = np.asarray(
    all_gaps,
    dtype=np.int64
)


print("\n" + "=" * 90)
print("WITHIN-SEQUENCE FRAME GAPS")
print("=" * 90)

print(
    "Adjacent image pairs:",
    len(gaps)
)


if len(gaps) > 0:

    print(
        pd.Series(
            gaps
        )
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
    )


    for threshold in [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
    ]:

        pct = (
            gaps
            <= threshold
        ).mean() * 100

        print(
            f"Gap <= {threshold:<3}: "
            f"{pct:6.2f}%"
        )


# ============================================================
# 4. LARGEST SEQUENCES
# ============================================================

print("\n" + "=" * 90)
print("TOP 20 LARGEST SEQUENCES")
print("=" * 90)

largest = (
    seq
    .sort_values(
        "images",
        ascending=False
    )
    .head(20)
)

print(
    largest[
        [
            "sequence_id",
            "camera_id",
            "person_id",
            "identity_type",
            "appearance_label",
            "reappearance_index",
            "images",
            "first_frame",
            "last_frame",
            "frame_span",
            "median_gap",
            "max_gap",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 5. VERY SHORT SEQUENCES
# ============================================================

print("\n" + "=" * 90)
print("SHORT SEQUENCES")
print("=" * 90)

for threshold in [
    1,
    2,
    5,
    10,
]:

    count = int(
        (
            seq["images"]
            <= threshold
        ).sum()
    )

    print(
        f"Sequences <= {threshold:2d} images:",
        count
    )


# ============================================================
# 6. KNOWN VS DISTRACTOR SEQUENCE STATS
# ============================================================

print("\n" + "=" * 90)
print("KNOWN ACTORS VS DISTRACTORS")
print("=" * 90)

for identity_type in [
    "known_actor",
    "distractor",
]:

    x = seq[
        seq[
            "identity_type"
        ]
        == identity_type
    ]

    print(
        f"\n{identity_type}"
    )

    print(
        "Sequences:",
        len(x)
    )

    print(
        "Persons:",
        x[
            "person_id"
        ].nunique()
    )

    print(
        "Images:",
        int(
            x[
                "images"
            ].sum()
        )
    )

    print(
        "Median images / sequence:",
        float(
            x[
                "images"
            ].median()
        )
    )

    print(
        "Mean images / sequence:",
        round(
            float(
                x[
                    "images"
                ].mean()
            ),
            2
        )
    )


# ============================================================
# 7. REAPPEARANCE STATISTICS
# ============================================================

print("\n" + "=" * 90)
print("REAPPEARANCE INDEX")
print("=" * 90)

print(
    seq[
        "reappearance_index"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nSequences with reappearance_index > 0:",
    int(
        (
            seq[
                "reappearance_index"
            ]
            > 0
        ).sum()
    )
)


# ============================================================
# 8. PERSON × CAMERA SEQUENCE COUNT
# ============================================================

person_camera_sequences = (
    seq.groupby(
        [
            "person_id",
            "camera_id",
        ]
    )[
        "sequence_id"
    ]
    .nunique()
)


print("\n" + "=" * 90)
print("SEQUENCES PER PERSON-CAMERA PAIR")
print("=" * 90)

print(
    person_camera_sequences
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


# ============================================================
# 9. SAME CAMERA + SAME FRAME CROWDING
# ============================================================

frame_groups = (
    df.groupby(
        [
            "camera_id",
            "frame_id",
        ]
    )
    .size()
)


multi_person_frames = (
    frame_groups[
        frame_groups > 1
    ]
)


print("\n" + "=" * 90)
print("SAME CAMERA / SAME FRAME MULTIPLE CROPS")
print("=" * 90)

print(
    "Camera-frame groups:",
    len(
        frame_groups
    )
)

print(
    "Groups with >1 crop:",
    len(
        multi_person_frames
    )
)

if len(
    multi_person_frames
) > 0:

    print(
        "Maximum crops in one camera-frame:",
        int(
            multi_person_frames.max()
        )
    )

    print(
        "Images in multi-crop frames:",
        int(
            multi_person_frames.sum()
        )
    )


print("\nSaved:")
print(SEQ_OUT)

print("\nDONE")
