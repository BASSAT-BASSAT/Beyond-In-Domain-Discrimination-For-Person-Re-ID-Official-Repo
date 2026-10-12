from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/PRW/extracted"
)

FRAMES_DIR = ROOT / "frames"
ANN_DIR = ROOT / "annotations"
QUERY_DIR = ROOT / "query_box"


def flatten_int(mat_path, key):

    mat = loadmat(
        mat_path,
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


def flatten_strings(mat_path, key):

    mat = loadmat(
        mat_path,
        squeeze_me=True,
        chars_as_strings=True,
    )

    arr = np.asarray(
        mat[key]
    ).reshape(-1)

    return [
        str(x).strip()
        for x in arr
    ]


# ============================================================
# ID SPLITS
# ============================================================

train_ids_raw = flatten_int(
    ROOT / "ID_train.mat",
    "ID_train",
)

test_ids_raw = flatten_int(
    ROOT / "ID_test.mat",
    "ID_test2",
)


train_ids = set(
    train_ids_raw.tolist()
)

test_ids = set(
    test_ids_raw.tolist()
)


print("=" * 95)
print("PRW PROTOCOL / SPLIT VERIFICATION")
print("=" * 95)


print("\n" + "=" * 95)
print("IDENTITY SPLITS")
print("=" * 95)

print(
    "Train ID entries:",
    len(train_ids_raw)
)

print(
    "Train unique IDs:",
    len(train_ids)
)

print(
    "Test ID entries:",
    len(test_ids_raw)
)

print(
    "Test unique IDs:",
    len(test_ids)
)


train_counts = Counter(
    train_ids_raw.tolist()
)

train_duplicates = {
    pid: count
    for pid, count in train_counts.items()
    if count > 1
}


test_counts = Counter(
    test_ids_raw.tolist()
)

test_duplicates = {
    pid: count
    for pid, count in test_counts.items()
    if count > 1
}


print(
    "Duplicate entries in train ID file:",
    train_duplicates
)

print(
    "Duplicate entries in test ID file:",
    test_duplicates
)


id_overlap = (
    train_ids
    & test_ids
)

id_union = (
    train_ids
    | test_ids
)


print(
    "Train/Test ID overlap:",
    len(id_overlap)
)

if id_overlap:

    print(
        "Overlap IDs:",
        sorted(id_overlap)
    )


print(
    "Union IDs:",
    len(id_union)
)


expected_ids = set(
    range(
        1,
        933
    )
)

missing_ids = (
    expected_ids
    - id_union
)

unexpected_ids = (
    id_union
    - expected_ids
)


print(
    "Missing IDs from 1..932:",
    sorted(missing_ids)
)

print(
    "Unexpected split IDs:",
    sorted(unexpected_ids)
)


# ============================================================
# FRAME SPLITS
# ============================================================

train_frames = flatten_strings(
    ROOT / "frame_train.mat",
    "img_index_train",
)

test_frames = flatten_strings(
    ROOT / "frame_test.mat",
    "img_index_test",
)


train_frame_set = set(
    train_frames
)

test_frame_set = set(
    test_frames
)


filesystem_frames = {
    p.stem
    for p in FRAMES_DIR.glob(
        "*.jpg"
    )
}


print("\n" + "=" * 95)
print("FRAME SPLITS")
print("=" * 95)

print(
    "Train frame entries:",
    len(train_frames)
)

print(
    "Train unique frames:",
    len(train_frame_set)
)

print(
    "Test frame entries:",
    len(test_frames)
)

print(
    "Test unique frames:",
    len(test_frame_set)
)

print(
    "Train/Test frame overlap:",
    len(
        train_frame_set
        & test_frame_set
    )
)

print(
    "Frame union:",
    len(
        train_frame_set
        | test_frame_set
    )
)

print(
    "Filesystem frames:",
    len(
        filesystem_frames
    )
)


print(
    "Filesystem missing from split:",
    len(
        filesystem_frames
        - (
            train_frame_set
            | test_frame_set
        )
    )
)

print(
    "Split missing from filesystem:",
    len(
        (
            train_frame_set
            | test_frame_set
        )
        - filesystem_frames
    )
)


# ============================================================
# ANNOTATIONS
# ============================================================

print("\n" + "=" * 95)
print("ANNOTATION AUDIT")
print("=" * 95)


records = []

invalid_annotation_files = []

total_boxes = 0
identified_boxes = 0
unknown_boxes = 0
invalid_pid_boxes = 0

non_integer_pid_rows = 0

bad_wh = 0


annotation_files = sorted(
    ANN_DIR.glob(
        "*.mat"
    )
)


for n, path in enumerate(
    annotation_files,
    start=1,
):

    frame_stem = (
        path.name
        .replace(
            ".jpg.mat",
            ""
        )
    )


    if frame_stem in train_frame_set:

        frame_split = "train"

    elif frame_stem in test_frame_set:

        frame_split = "test"

    else:

        frame_split = "unassigned"


    mat = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True,
    )


    if "box_new" not in mat:

        invalid_annotation_files.append(
            (
                path.name,
                "missing box_new"
            )
        )

        continue


    arr = np.asarray(
        mat["box_new"],
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

        invalid_annotation_files.append(
            (
                path.name,
                f"shape={arr.shape}"
            )
        )

        continue


    for box_index, row in enumerate(
        arr
    ):

        pid_raw = row[0]

        pid = int(
            round(
                pid_raw
            )
        )


        if not np.isclose(
            pid_raw,
            pid
        ):

            non_integer_pid_rows += 1


        x, y, w, h = map(
            float,
            row[1:]
        )


        if (
            w <= 0
            or h <= 0
        ):

            bad_wh += 1


        if pid == -2:

            identity_status = (
                "unknown"
            )

            unknown_boxes += 1

            id_split = "unknown"

        elif 1 <= pid <= 932:

            identity_status = (
                "identified"
            )

            identified_boxes += 1

            if pid in train_ids:

                id_split = (
                    "train_id"
                )

            elif pid in test_ids:

                id_split = (
                    "test_id"
                )

            else:

                id_split = (
                    "unassigned_id"
                )

        else:

            identity_status = (
                "invalid"
            )

            invalid_pid_boxes += 1
            id_split = "invalid"


        records.append(
            {
                "frame_stem":
                    frame_stem,

                "frame_split":
                    frame_split,

                "box_index":
                    box_index,

                "person_id":
                    pid,

                "identity_status":
                    identity_status,

                "id_split":
                    id_split,

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


        total_boxes += 1


    if (
        n % 2000 == 0
        or n == len(
            annotation_files
        )
    ):

        print(
            f"processed "
            f"{n:,} / "
            f"{len(annotation_files):,} "
            f"annotation files"
        )


boxes = pd.DataFrame(
    records
)


print(
    "\nAnnotation files:",
    len(
        annotation_files
    )
)

print(
    "Invalid annotation files:",
    len(
        invalid_annotation_files
    )
)

if invalid_annotation_files:

    print(
        invalid_annotation_files[:20]
    )


print(
    "\nTotal pedestrian boxes:",
    total_boxes
)

print(
    "Identified boxes:",
    identified_boxes
)

print(
    "Unknown PID=-2 boxes:",
    unknown_boxes
)

print(
    "Invalid PID boxes:",
    invalid_pid_boxes
)

print(
    "Non-integer PID rows:",
    non_integer_pid_rows
)

print(
    "Boxes with w<=0 or h<=0:",
    bad_wh
)


print(
    "\nUnique identified PIDs:",
    boxes.loc[
        boxes[
            "identity_status"
        ]
        == "identified",
        "person_id",
    ].nunique()
)


# ============================================================
# BOX COUNTS BY FRAME / ID SPLIT
# ============================================================

print("\n" + "=" * 95)
print("BOX COUNTS BY FRAME SPLIT / ID ROLE")
print("=" * 95)


table = pd.crosstab(
    boxes[
        "frame_split"
    ],
    boxes[
        "id_split"
    ],
    margins=True,
)

print(
    table
)


# ============================================================
# IDENTITIES OBSERVED IN FRAME SPLITS
# ============================================================

identified = boxes[
    boxes[
        "identity_status"
    ]
    == "identified"
]


train_frame_pids = set(
    identified.loc[
        identified[
            "frame_split"
        ]
        == "train",
        "person_id",
    ]
)


test_frame_pids = set(
    identified.loc[
        identified[
            "frame_split"
        ]
        == "test",
        "person_id",
    ]
)


print("\n" + "=" * 95)
print("IDENTITY OCCURRENCE ACROSS FRAME SPLITS")
print("=" * 95)


print(
    "Unique identified PIDs in train frames:",
    len(
        train_frame_pids
    )
)

print(
    "Unique identified PIDs in test frames:",
    len(
        test_frame_pids
    )
)


print(
    "Official train IDs appearing in train frames:",
    len(
        train_ids
        & train_frame_pids
    )
)

print(
    "Official train IDs appearing in test frames:",
    len(
        train_ids
        & test_frame_pids
    )
)

print(
    "Official test IDs appearing in train frames:",
    len(
        test_ids
        & train_frame_pids
    )
)

print(
    "Official test IDs appearing in test frames:",
    len(
        test_ids
        & test_frame_pids
    )
)


train_ids_in_test = sorted(
    train_ids
    & test_frame_pids
)

test_ids_in_train = sorted(
    test_ids
    & train_frame_pids
)


print(
    "\nTrain IDs found in test frames:",
    train_ids_in_test
)

print(
    "\nTest IDs found in train frames:",
    test_ids_in_train
)


# ============================================================
# QUERY AUDIT
# ============================================================

print("\n" + "=" * 95)
print("QUERY AUDIT")
print("=" * 95)


query_rows = []


with open(
    ROOT / "query_info.txt",
    "r",
    encoding="utf-8",
) as f:

    for line_no, line in enumerate(
        f,
        start=1,
    ):

        line = line.strip()

        if not line:
            continue


        parts = line.split()

        if len(parts) != 6:

            raise ValueError(
                f"Bad query_info line "
                f"{line_no}: {line}"
            )


        pid = int(
            parts[0]
        )

        x, y, w, h = map(
            float,
            parts[1:5]
        )

        frame_stem = parts[5]


        # c1s3_016471
        left, frame_number = (
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


        expected_filename = (
            f"{pid}_"
            f"{frame_stem}.jpg"
        )


        query_rows.append(
            {
                "person_id":
                    pid,

                "x":
                    x,

                "y":
                    y,

                "w":
                    w,

                "h":
                    h,

                "frame_stem":
                    frame_stem,

                "camera_id":
                    camera_id,

                "segment_id":
                    segment_id,

                "frame_number":
                    int(
                        frame_number
                    ),

                "query_filename":
                    expected_filename,

                "query_exists":
                    (
                        QUERY_DIR
                        / expected_filename
                    ).is_file(),

                "source_frame_exists":
                    (
                        FRAMES_DIR
                        / f"{frame_stem}.jpg"
                    ).is_file(),

                "source_frame_split":
                    (
                        "train"
                        if frame_stem
                        in train_frame_set
                        else
                        "test"
                        if frame_stem
                        in test_frame_set
                        else
                        "unassigned"
                    ),
            }
        )


queries = pd.DataFrame(
    query_rows
)


filesystem_queries = {
    p.name
    for p in QUERY_DIR.glob(
        "*.jpg"
    )
}


query_info_files = set(
    queries[
        "query_filename"
    ]
)


print(
    "query_info rows:",
    len(
        queries
    )
)

print(
    "Query JPG files:",
    len(
        filesystem_queries
    )
)

print(
    "Unique query filenames:",
    len(
        query_info_files
    )
)

print(
    "Missing query JPGs:",
    int(
        (
            ~queries[
                "query_exists"
            ]
        ).sum()
    )
)

print(
    "Extra query JPGs:",
    len(
        filesystem_queries
        - query_info_files
    )
)

print(
    "Missing source frames:",
    int(
        (
            ~queries[
                "source_frame_exists"
            ]
        ).sum()
    )
)


query_ids = set(
    queries[
        "person_id"
    ]
)


print(
    "\nUnique query IDs:",
    len(
        query_ids
    )
)

print(
    "Query IDs in official test IDs:",
    len(
        query_ids
        & test_ids
    )
)

print(
    "Query IDs NOT in official test IDs:",
    sorted(
        query_ids
        - test_ids
    )
)

print(
    "Query IDs overlapping train IDs:",
    sorted(
        query_ids
        & train_ids
    )
)


print(
    "\nQuery source frame split:"
)

print(
    queries[
        "source_frame_split"
    ].value_counts()
)


query_per_pid = (
    queries.groupby(
        "person_id"
    )
    .size()
)


print(
    "\nQueries per PID:"
)

print(
    query_per_pid
    .describe()
    .round(2)
)


query_pid_camera = (
    queries.groupby(
        [
            "person_id",
            "camera_id",
        ]
    )
    .size()
)


print(
    "Maximum queries per PID-camera:",
    int(
        query_pid_camera.max()
    )
)

print(
    "PID-camera pairs with >1 query:",
    int(
        (
            query_pid_camera > 1
        ).sum()
    )
)


# ============================================================
# SAVE SUMMARY TABLES
# ============================================================

OUT_DIR = (
    BASE
    / "manifests/PRW"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


boxes.to_csv(
    OUT_DIR
    / "prw_annotation_boxes_pre_manifest.csv",
    index=False,
)


queries.to_csv(
    OUT_DIR
    / "prw_queries_pre_manifest.csv",
    index=False,
)


print("\nSaved:")
print(
    OUT_DIR
    / "prw_annotation_boxes_pre_manifest.csv"
)

print(
    OUT_DIR
    / "prw_queries_pre_manifest.csv"
)


print("\nDONE")
