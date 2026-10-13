from pathlib import Path
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
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


OFFICIAL = {
    1:  {"images":59230,  "known":78, "reapp":107, "distractors":297, "sequences":485},
    2:  {"images":112523, "known":83, "reapp":94,  "distractors":653, "sequences":830},
    3:  {"images":72005,  "known":74, "reapp":68,  "distractors":822, "sequences":964},
    4:  {"images":53986,  "known":70, "reapp":72,  "distractors":393, "sequences":536},
    5:  {"images":67672,  "known":63, "reapp":52,  "distractors":781, "sequences":896},
    6:  {"images":56472,  "known":64, "reapp":38,  "distractors":865, "sequences":967},
    7:  {"images":17809,  "known":48, "reapp":36,  "distractors":93,  "sequences":177},
    8:  {"images":36338,  "known":55, "reapp":40,  "distractors":266, "sequences":361},
    9:  {"images":3910,   "known":40, "reapp":16,  "distractors":32,  "sequences":88},
    10: {"images":10492,  "known":62, "reapp":51,  "distractors":105, "sequences":218},
    11: {"images":73601,  "known":76, "reapp":149, "distractors":448, "sequences":673},
    12: {"images":37543,  "known":70, "reapp":79,  "distractors":233, "sequences":382},
}


print("=" * 95)
print("RPIFIELD OFFICIAL-STATISTICS VERIFICATION")
print("=" * 95)


rows = []


for cam in range(1, 13):

    x = df[
        df["camera_id"] == cam
    ]

    known = x[
        x["is_known_actor"] == True
    ]

    distractor = x[
        x["is_distractor"] == True
    ]


    known_sequences = (
        known[
            [
                "person_id",
                "appearance_label",
            ]
        ]
        .drop_duplicates()
    )


    known_suffix_reapp = (
        known_sequences[
            "appearance_label"
        ]
        .str.contains(
            "_",
            regex=False
        )
        .sum()
    )


    known_extra_sequences = (
        known_sequences.shape[0]
        - known[
            "person_id"
        ].nunique()
    )


    dist_sequences = (
        distractor[
            [
                "person_id",
                "appearance_label",
            ]
        ]
        .drop_duplicates()
    )


    distractor_suffix_reapp = (
        dist_sequences[
            "appearance_label"
        ]
        .str.contains(
            "_",
            regex=False
        )
        .sum()
    )


    actual = {
        "images":
            len(x),

        "known":
            known[
                "person_id"
            ].nunique(),

        "distractors":
            distractor[
                "person_id"
            ].nunique(),

        "sequences":
            x[
                "appearance_label"
            ].nunique(),

        "known_suffix_reapp":
            int(
                known_suffix_reapp
            ),

        "known_extra_sequences":
            int(
                known_extra_sequences
            ),

        "distractor_suffix_reapp":
            int(
                distractor_suffix_reapp
            ),
    }


    official = OFFICIAL[
        cam
    ]


    print(
        f"\nCam_{cam}"
    )

    print(
        f" Images      : "
        f"{actual['images']} "
        f"(official {official['images']})"
    )

    print(
        f" Known IDs   : "
        f"{actual['known']} "
        f"(official {official['known']})"
    )

    print(
        f" Distractors : "
        f"{actual['distractors']} "
        f"(official {official['distractors']})"
    )

    print(
        f" Sequences   : "
        f"{actual['sequences']} "
        f"(official {official['sequences']})"
    )

    print(
        f" Known reapp suffixes : "
        f"{actual['known_suffix_reapp']} "
        f"(official {official['reapp']})"
    )

    print(
        f" Known extra sequences: "
        f"{actual['known_extra_sequences']}"
    )

    print(
        f" Distractor reapp suffixes: "
        f"{actual['distractor_suffix_reapp']}"
    )


    rows.append(
        {
            "camera_id": cam,
            **actual,
            "official_reapp":
                official["reapp"],
        }
    )


summary = pd.DataFrame(
    rows
)


print("\n" + "=" * 95)
print("REAPPEARANCE TOTALS")
print("=" * 95)

print(
    "Known suffix reappearances:",
    int(
        summary[
            "known_suffix_reapp"
        ].sum()
    )
)

print(
    "Official reappearances:",
    sum(
        v["reapp"]
        for v in OFFICIAL.values()
    )
)

print(
    "Distractor suffix reappearances:",
    int(
        summary[
            "distractor_suffix_reapp"
        ].sum()
    )
)


# ============================================================
# KNOWN ACTOR CAMERA COVERAGE
# ============================================================

known = df[
    df[
        "is_known_actor"
    ]
    == True
]


coverage = (
    known[
        [
            "person_id",
            "camera_id",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "person_id"
    )[
        "camera_id"
    ]
    .agg(
        [
            "count",
            lambda x:
                ",".join(
                    map(
                        str,
                        sorted(x)
                    )
                ),
        ]
    )
)

coverage.columns = [
    "camera_count",
    "cameras",
]


print("\n" + "=" * 95)
print("KNOWN ACTOR CAMERA COVERAGE < 3")
print("=" * 95)


problem = coverage[
    coverage[
        "camera_count"
    ]
    < 3
]


if len(problem) == 0:

    print(
        "None"
    )

else:

    for pid, row in problem.iterrows():

        x = known[
            known[
                "person_id"
            ]
            == pid
        ]

        print(
            f"\nPID {pid}"
        )

        print(
            "Cameras:",
            row[
                "cameras"
            ]
        )

        print(
            "Images:",
            len(x)
        )

        print(
            "Sequences:",
            x[
                "sequence_id"
            ].nunique()
        )

        print(
            "Appearance labels:",
            sorted(
                x[
                    "appearance_label"
                ]
                .unique()
            )
        )

        print(
            "Frame range:",
            int(
                x[
                    "frame_id"
                ].min()
            ),
            "->",
            int(
                x[
                    "frame_id"
                ].max()
            )
        )


print("\n" + "=" * 95)
print("KNOWN CAMERA-COVERAGE DISTRIBUTION")
print("=" * 95)

print(
    coverage[
        "camera_count"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# PERSON-CAMERA PAIRS WITHOUT BASE APPEARANCE
# ============================================================

print("\n" + "=" * 95)
print("PERSON-CAMERA PAIRS WITHOUT REAPPEARANCE_INDEX 0")
print("=" * 95)


problems = []


for (
    person_id,
    camera_id
), x in df.groupby(
    [
        "person_id",
        "camera_id",
    ]
):

    indices = set(
        x[
            "reappearance_index"
        ].astype(int)
    )

    if 0 not in indices:

        problems.append(
            {
                "person_id":
                    person_id,

                "camera_id":
                    camera_id,

                "identity_type":
                    x[
                        "identity_type"
                    ].iloc[0],

                "indices":
                    sorted(
                        indices
                    ),

                "appearance_labels":
                    sorted(
                        x[
                            "appearance_label"
                        ]
                        .astype(str)
                        .unique()
                    ),
            }
        )


print(
    "Count:",
    len(
        problems
    )
)


for item in problems:

    print(
        item
    )


print("\nDONE")
