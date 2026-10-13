from pathlib import Path
import heapq
import itertools

import numpy as np
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
)

OUT_DIR = (
    BASE
    / "manifests/RPIfield"
)

LARGE_GAPS_OUT = (
    OUT_DIR
    / "rpifield_large_temporal_gaps.csv"
)

SAME_PID_FRAME_OUT = (
    OUT_DIR
    / "rpifield_same_pid_same_camera_frame.csv"
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
        "identity_type": str,
    },
    low_memory=False,
)


print("=" * 90)
print("RPIFIELD TEMPORAL GAP DIAGNOSTIC")
print("=" * 90)

print("Images:", len(df))
print("Sequences:", df["sequence_id"].nunique())


thresholds = [
    10,
    20,
    50,
    100,
    500,
    1000,
    5000,
    10000,
    50000,
]


pair_counts = {
    t: 0
    for t in thresholds
}

sequence_counts = {
    t: 0
    for t in thresholds
}


total_pairs = 0

large_gap_rows = []

heap = []
counter = itertools.count()


for sequence_id, x in df.groupby(
    "sequence_id",
    sort=False,
):

    x = x.sort_values(
        "frame_id"
    ).reset_index(
        drop=True
    )

    frames = (
        x["frame_id"]
        .astype(np.int64)
        .to_numpy()
    )

    if len(frames) <= 1:
        continue


    gaps = np.diff(
        frames
    )

    total_pairs += len(gaps)


    for threshold in thresholds:

        mask = gaps > threshold

        pair_counts[
            threshold
        ] += int(
            mask.sum()
        )

        if mask.any():

            sequence_counts[
                threshold
            ] += 1


    # Save all gaps > 100
    large_indices = np.where(
        gaps > 100
    )[0]


    for i in large_indices:

        before = x.iloc[i]
        after = x.iloc[i + 1]

        gap = int(
            gaps[i]
        )


        record = {
            "sequence_id":
                sequence_id,

            "camera_id":
                int(
                    before["camera_id"]
                ),

            "person_id":
                int(
                    before["person_id"]
                ),

            "identity_type":
                before["identity_type"],

            "appearance_label":
                before[
                    "appearance_label"
                ],

            "reappearance_index":
                int(
                    before[
                        "reappearance_index"
                    ]
                ),

            "before_frame":
                int(
                    before["frame_id"]
                ),

            "after_frame":
                int(
                    after["frame_id"]
                ),

            "gap":
                gap,

            "before_path":
                before["path"],

            "after_path":
                after["path"],
        }


        large_gap_rows.append(
            record
        )


        item = (
            gap,
            next(counter),
            record,
        )

        if len(heap) < 50:

            heapq.heappush(
                heap,
                item
            )

        else:

            heapq.heappushpop(
                heap,
                item
            )


large = pd.DataFrame(
    large_gap_rows
)

if len(large) > 0:

    large = large.sort_values(
        "gap",
        ascending=False
    )

large.to_csv(
    LARGE_GAPS_OUT,
    index=False
)


print("\n" + "=" * 90)
print("GAP THRESHOLD SUMMARY")
print("=" * 90)

print(
    "Total adjacent pairs:",
    total_pairs
)


for threshold in thresholds:

    pairs = pair_counts[
        threshold
    ]

    sequences = sequence_counts[
        threshold
    ]

    pair_pct = (
        pairs
        / total_pairs
        * 100
    )

    print(
        f"Gap > {threshold:<6} | "
        f"pairs={pairs:>7,} "
        f"({pair_pct:6.3f}%) | "
        f"sequences={sequences:>5,}"
    )


print("\n" + "=" * 90)
print("TOP 50 LARGEST GAPS")
print("=" * 90)


top = sorted(
    heap,
    key=lambda x: x[0],
    reverse=True
)


for rank, (
    gap,
    _,
    record,
) in enumerate(
    top,
    start=1,
):

    print(
        f"{rank:>2}. "
        f"gap={gap:>7} | "
        f"{record['sequence_id']:<18} | "
        f"PID={record['person_id']:<6} | "
        f"{record['before_frame']} "
        f"-> "
        f"{record['after_frame']}"
    )


# ============================================================
# SAME PERSON / CAMERA / FRAME APPEARING MORE THAN ONCE
# ============================================================

print("\n" + "=" * 90)
print("SAME PID + CAMERA + FRAME DUPLICATION")
print("=" * 90)


group_sizes = (
    df.groupby(
        [
            "camera_id",
            "person_id",
            "frame_id",
        ]
    )
    .size()
)


problem_keys = (
    group_sizes[
        group_sizes > 1
    ]
    .reset_index(
        name="count"
    )
)


print(
    "Problem groups:",
    len(
        problem_keys
    )
)


if len(problem_keys) > 0:

    problem_rows = df.merge(
        problem_keys[
            [
                "camera_id",
                "person_id",
                "frame_id",
            ]
        ],
        on=[
            "camera_id",
            "person_id",
            "frame_id",
        ],
        how="inner",
    )

    print(
        "Images involved:",
        len(
            problem_rows
        )
    )

    print(
        "\nExamples:"
    )

    print(
        problem_rows[
            [
                "camera_id",
                "person_id",
                "appearance_label",
                "frame_id",
                "filename",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

else:

    problem_rows = pd.DataFrame()


problem_rows.to_csv(
    SAME_PID_FRAME_OUT,
    index=False
)


print("\nSaved:")
print(LARGE_GAPS_OUT)
print(SAME_PID_FRAME_OUT)

print("\nDONE")
