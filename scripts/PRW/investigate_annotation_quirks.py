from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.io import loadmat


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/PRW/extracted"
)

ANN_DIR = (
    ROOT
    / "annotations"
)

OUT_DIR = (
    BASE
    / "manifests/PRW"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT = (
    OUT_DIR
    / "prw_annotation_boxes_v2.csv"
)


# ============================================================
# Helpers
# ============================================================

def get_int_array(path, key):

    mat = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True,
    )

    return (
        np.asarray(
            mat[key]
        )
        .reshape(-1)
        .astype(int)
    )


def get_strings(path, key):

    mat = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True,
    )

    arr = np.asarray(
        mat[key]
    ).reshape(-1)

    return {
        str(x).strip()
        for x in arr
    }


# ============================================================
# Official split files
# ============================================================

train_ids = set(
    get_int_array(
        ROOT / "ID_train.mat",
        "ID_train",
    )
)

test_ids = set(
    get_int_array(
        ROOT / "ID_test.mat",
        "ID_test2",
    )
)

train_frames = get_strings(
    ROOT / "frame_train.mat",
    "img_index_train",
)

test_frames = get_strings(
    ROOT / "frame_test.mat",
    "img_index_test",
)


print("=" * 95)
print("PRW ANNOTATION / PID QUIRK INVESTIGATION")
print("=" * 95)


print("\nPID 933 SPLIT MEMBERSHIP")
print("-" * 95)

print(
    "PID 933 in train IDs:",
    933 in train_ids
)

print(
    "PID 933 in test IDs:",
    933 in test_ids
)


# ============================================================
# Parse all annotation files
# ============================================================

candidate_keys = [
    "box_new",
    "anno_file",
    "anno_previous",
]


key_counts = Counter()

multi_candidate_files = []

unsupported_files = []

records = []

pid_counts = Counter()

pid_cameras = defaultdict(
    set
)

pid_frames = defaultdict(
    set
)


annotation_files = sorted(
    ANN_DIR.glob(
        "*.mat"
    )
)


for n, path in enumerate(
    annotation_files,
    start=1,
):

    mat = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True,
    )


    public_keys = [
        k
        for k in mat.keys()
        if not k.startswith("__")
    ]


    available = [
        key
        for key in candidate_keys
        if key in mat
    ]


    if len(available) == 0:

        unsupported_files.append(
            {
                "file":
                    path.name,

                "keys":
                    ",".join(
                        public_keys
                    ),
            }
        )

        continue


    if len(available) > 1:

        multi_candidate_files.append(
            {
                "file":
                    path.name,

                "candidate_keys":
                    ",".join(
                        available
                    ),
            }
        )


    # deterministic priority
    key = available[0]

    key_counts[
        key
    ] += 1


    arr = np.asarray(
        mat[key],
        dtype=float,
    )


    if arr.size == 0:

        arr = np.empty(
            (
                0,
                5
            )
        )

    elif arr.ndim == 1:

        arr = arr.reshape(
            1,
            -1
        )


    if (
        arr.ndim != 2
        or arr.shape[1] != 5
    ):

        raise ValueError(
            f"{path.name}: "
            f"key={key}, "
            f"shape={arr.shape}"
        )


    frame_stem = (
        path.name
        .replace(
            ".jpg.mat",
            ""
        )
    )


    # Parse camera/segment/frame
    left, frame_num = (
        frame_stem.split(
            "_",
            1
        )
    )

    camera_part, segment_part = (
        left.split(
            "s",
            1
        )
    )

    camera_id = int(
        camera_part[1:]
    )

    segment_id = int(
        segment_part
    )


    if frame_stem in train_frames:

        frame_split = "train"

    elif frame_stem in test_frames:

        frame_split = "test"

    else:

        frame_split = "unassigned"


    for box_index, row in enumerate(
        arr
    ):

        pid_raw = float(
            row[0]
        )

        pid = int(
            round(
                pid_raw
            )
        )

        if not np.isclose(
            pid_raw,
            pid
        ):

            raise ValueError(
                f"Non-integer PID "
                f"{pid_raw} in {path.name}"
            )


        x, y, w, h = map(
            float,
            row[1:]
        )


        if pid == -2:

            id_role = "unknown"

        elif pid in train_ids:

            id_role = "train_id"

        elif pid in test_ids:

            id_role = "test_id"

        else:

            id_role = "outside_split"


        pid_counts[
            pid
        ] += 1

        pid_cameras[
            pid
        ].add(
            camera_id
        )

        pid_frames[
            pid
        ].add(
            frame_stem
        )


        records.append(
            {
                "frame_stem":
                    frame_stem,

                "frame_split":
                    frame_split,

                "annotation_key":
                    key,

                "camera_id":
                    camera_id,

                "segment_id":
                    segment_id,

                "frame_number":
                    int(
                        frame_num
                    ),

                "box_index":
                    box_index,

                "person_id":
                    pid,

                "id_role":
                    id_role,

                "x":
                    x,

                "y":
                    y,

                "w":
                    w,

                "h":
                    h,
            }
        )


    if (
        n % 2000 == 0
        or n == len(
            annotation_files
        )
    ):

        print(
            f"processed "
            f"{n:,} / "
            f"{len(annotation_files):,}"
        )


boxes = pd.DataFrame(
    records
)


# ============================================================
# Annotation key summary
# ============================================================

print("\n" + "=" * 95)
print("ANNOTATION KEY SUMMARY")
print("=" * 95)

print(
    "Annotation files:",
    len(
        annotation_files
    )
)

print(
    "Key usage:"
)

for key, count in key_counts.items():

    print(
        f"  {key:<15}: "
        f"{count}"
    )


print(
    "Files with >1 candidate key:",
    len(
        multi_candidate_files
    )
)

if multi_candidate_files:

    print(
        multi_candidate_files[:20]
    )


print(
    "Unsupported files:",
    len(
        unsupported_files
    )
)

if unsupported_files:

    print(
        unsupported_files[:50]
    )


# ============================================================
# Global box counts
# ============================================================

print("\n" + "=" * 95)
print("GLOBAL BOX COUNTS")
print("=" * 95)

print(
    "Total boxes:",
    len(
        boxes
    )
)


print(
    "\nBoxes by ID role:"
)

print(
    boxes[
        "id_role"
    ].value_counts()
)


print(
    "\nUnique PIDs excluding -2:",
    boxes.loc[
        boxes[
            "person_id"
        ]
        != -2,
        "person_id",
    ].nunique()
)


outside = boxes[
    boxes[
        "id_role"
    ]
    == "outside_split"
]


print(
    "Rows outside train/test ID lists:",
    len(
        outside
    )
)

if len(outside):

    print(
        outside[
            "person_id"
        ]
        .value_counts()
        .sort_index()
    )


# ============================================================
# PID 933
# ============================================================

print("\n" + "=" * 95)
print("PID 933")
print("=" * 95)


p933 = boxes[
    boxes[
        "person_id"
    ]
    == 933
]


print(
    "Boxes:",
    len(
        p933
    )
)

print(
    "Frame split:"
)

print(
    p933[
        "frame_split"
    ].value_counts()
)


print(
    "Cameras:",
    sorted(
        p933[
            "camera_id"
        ].unique()
    )
)


print(
    "Unique frames:",
    p933[
        "frame_stem"
    ].nunique()
)


if len(p933):

    print(
        "\nFirst 30 rows:"
    )

    print(
        p933[
            [
                "frame_stem",
                "frame_split",
                "camera_id",
                "person_id",
                "x",
                "y",
                "w",
                "h",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# Split occurrence after corrected parsing
# ============================================================

print("\n" + "=" * 95)
print("IDENTITY OCCURRENCE AFTER CORRECTED PARSING")
print("=" * 95)


known_boxes = boxes[
    boxes[
        "person_id"
    ]
    != -2
]


train_frame_pids = set(
    known_boxes.loc[
        known_boxes[
            "frame_split"
        ]
        == "train",
        "person_id",
    ]
)


test_frame_pids = set(
    known_boxes.loc[
        known_boxes[
            "frame_split"
        ]
        == "test",
        "person_id",
    ]
)


print(
    "Train IDs observed in train frames:",
    len(
        train_ids
        & train_frame_pids
    ),
    "/",
    len(
        train_ids
    )
)

print(
    "Train IDs observed in test frames:",
    len(
        train_ids
        & test_frame_pids
    )
)


print(
    "Test IDs observed in train frames:",
    len(
        test_ids
        & train_frame_pids
    )
)

print(
    "Test IDs observed in test frames:",
    len(
        test_ids
        & test_frame_pids
    ),
    "/",
    len(
        test_ids
    )
)


print(
    "\nTest IDs absent from test-frame annotations:"
)

print(
    sorted(
        test_ids
        - test_frame_pids
    )
)


print(
    "\nTrain IDs absent from train-frame annotations:"
)

print(
    sorted(
        train_ids
        - train_frame_pids
    )
)


# ============================================================
# Query check for PID 933
# ============================================================

query_df = pd.read_csv(
    ROOT / "query_info.txt",
    sep=r"\s+",
    header=None,
    names=[
        "person_id",
        "x",
        "y",
        "w",
        "h",
        "frame_stem",
    ],
)


print("\n" + "=" * 95)
print("QUERY CHECK")
print("=" * 95)


print(
    "PID 933 query rows:",
    int(
        (
            query_df[
                "person_id"
            ]
            == 933
        ).sum()
    )
)


if (
    query_df[
        "person_id"
    ]
    .eq(
        933
    )
    .any()
):

    print(
        query_df[
            query_df[
                "person_id"
            ]
            == 933
        ].to_string(
            index=False
        )
    )


# ============================================================
# Camera coverage
# ============================================================

pid_summary = []


for pid in sorted(
    p
    for p in pid_counts
    if p != -2
):

    pid_summary.append(
        {
            "person_id":
                pid,

            "boxes":
                pid_counts[
                    pid
                ],

            "cameras":
                len(
                    pid_cameras[
                        pid
                    ]
                ),

            "frames":
                len(
                    pid_frames[
                        pid
                    ]
                ),

            "split_role":
                (
                    "train_id"
                    if pid in train_ids
                    else
                    "test_id"
                    if pid in test_ids
                    else
                    "outside_split"
                ),
        }
    )


pid_summary = pd.DataFrame(
    pid_summary
)


print("\n" + "=" * 95)
print("PID SUMMARY")
print("=" * 95)

print(
    "Total non--2 PIDs:",
    len(
        pid_summary
    )
)

print(
    "\nBoxes per PID:"
)

print(
    pid_summary[
        "boxes"
    ]
    .describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .95,
            .99,
        ]
    )
    .round(2)
)


# ============================================================
# Save corrected box table
# ============================================================

boxes.to_csv(
    OUT,
    index=False
)


pid_summary.to_csv(
    OUT_DIR
    / "prw_pid_summary.csv",
    index=False
)


print("\nSaved:")
print(OUT)
print(
    OUT_DIR
    / "prw_pid_summary.csv"
)

print("\nDONE")
