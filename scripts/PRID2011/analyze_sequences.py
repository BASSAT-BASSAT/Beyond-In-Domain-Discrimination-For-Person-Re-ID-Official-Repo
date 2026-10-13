from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRID2011"
    / "prid2011_manifest.csv"
)

OUT = (
    BASE
    / "manifests/PRID2011"
    / "prid2011_sequence_stats.csv"
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "global_person_id": str,
        "trajectory_id": str,
    },
)


multi = df[
    df["version"] == "multi_shot"
].copy()


print("=" * 90)
print("PRID2011 MULTI-SHOT SEQUENCE AUDIT")
print("=" * 90)

print(
    "\nImages:",
    len(multi)
)

print(
    "Trajectories:",
    multi["trajectory_id"].nunique()
)


rows = []

duplicate_index_trajectories = 0
non_contiguous_trajectories = 0
start_not_one = 0

all_gaps = []


for trajectory_id, x in multi.groupby(
    "trajectory_id",
    sort=False,
):

    indices = (
        x["sequence_index"]
        .astype(int)
        .sort_values()
        .to_numpy()
    )

    unique_indices = np.unique(
        indices
    )

    duplicate_indices = (
        len(indices)
        - len(unique_indices)
    )

    if duplicate_indices > 0:
        duplicate_index_trajectories += 1


    first_index = int(
        indices[0]
    )

    last_index = int(
        indices[-1]
    )

    if first_index != 1:
        start_not_one += 1


    gaps = np.diff(
        unique_indices
    )

    if len(gaps) > 0:

        all_gaps.extend(
            gaps.tolist()
        )

        missing_count = int(
            np.maximum(
                gaps - 1,
                0
            ).sum()
        )

        max_gap = int(
            gaps.max()
        )

    else:

        missing_count = 0
        max_gap = 0


    expected_count = (
        last_index
        - first_index
        + 1
    )

    contiguous = (
        len(unique_indices)
        == expected_count
    )


    if not contiguous:
        non_contiguous_trajectories += 1


    first = x.iloc[0]


    rows.append(
        {
            "trajectory_id":
                trajectory_id,

            "camera":
                first["camera"],

            "global_person_id":
                first["global_person_id"],

            "images":
                len(indices),

            "unique_sequence_indices":
                len(unique_indices),

            "first_index":
                first_index,

            "last_index":
                last_index,

            "duplicate_indices":
                duplicate_indices,

            "missing_indices":
                missing_count,

            "max_gap":
                max_gap,

            "contiguous":
                contiguous,
        }
    )


stats = pd.DataFrame(
    rows
)

stats.to_csv(
    OUT,
    index=False
)


print("\n" + "=" * 90)
print("SEQUENCE INDEX INTEGRITY")
print("=" * 90)

print(
    "Trajectories starting at 1:",
    int(
        (
            stats["first_index"] == 1
        ).sum()
    ),
    "/",
    len(stats)
)

print(
    "Trajectories not starting at 1:",
    start_not_one
)

print(
    "Trajectories with duplicate indices:",
    duplicate_index_trajectories
)

print(
    "Non-contiguous trajectories:",
    non_contiguous_trajectories
)

print(
    "Total missing sequence indices:",
    int(
        stats["missing_indices"].sum()
    )
)


print("\n" + "=" * 90)
print("IMAGES PER TRAJECTORY")
print("=" * 90)

print(
    stats["images"]
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


print("\n" + "=" * 90)
print("BY CAMERA")
print("=" * 90)

for camera in [
    "cam_a",
    "cam_b",
]:

    x = stats[
        stats["camera"] == camera
    ]

    print(
        f"\n{camera}"
    )

    print(
        "Trajectories:",
        len(x)
    )

    print(
        "Images:",
        int(
            x["images"].sum()
        )
    )

    print(
        "Min images:",
        int(
            x["images"].min()
        )
    )

    print(
        "Median images:",
        float(
            x["images"].median()
        )
    )

    print(
        "Mean images:",
        round(
            float(
                x["images"].mean()
            ),
            2
        )
    )

    print(
        "Max images:",
        int(
            x["images"].max()
        )
    )

    print(
        "Non-contiguous:",
        int(
            (~x["contiguous"]).sum()
        )
    )


if len(all_gaps) > 0:

    gaps = np.array(
        all_gaps
    )

    print("\n" + "=" * 90)
    print("CONSECUTIVE SEQUENCE-INDEX GAPS")
    print("=" * 90)

    print(
        "Pairs:",
        len(gaps)
    )

    print(
        "Gap = 1:",
        int(
            (gaps == 1).sum()
        ),
        f"({(gaps == 1).mean() * 100:.2f}%)"
    )

    print(
        "Gap > 1:",
        int(
            (gaps > 1).sum()
        ),
        f"({(gaps > 1).mean() * 100:.2f}%)"
    )

    print(
        "Max gap:",
        int(
            gaps.max()
        )
    )


problem = stats[
    (
        stats["duplicate_indices"] > 0
    )
    |
    (
        ~stats["contiguous"]
    )
    |
    (
        stats["first_index"] != 1
    )
]


if len(problem) > 0:

    print("\n" + "=" * 90)
    print("PROBLEMATIC TRAJECTORIES")
    print("=" * 90)

    print(
        problem.head(30)
        .to_string(
            index=False
        )
    )


print("\nSaved:")
print(OUT)

print("\nDONE")
