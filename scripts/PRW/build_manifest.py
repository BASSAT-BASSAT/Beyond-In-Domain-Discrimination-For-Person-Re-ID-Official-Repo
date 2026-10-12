from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat


BASE = Path.home() / "DG_ReID_Audit"
ROOT = BASE / "datasets/PRW/extracted"

FRAMES = ROOT / "frames"
ANN = ROOT / "annotations"

OUT_DIR = BASE / "manifests/PRW"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "prw_manifest.csv"


def load_ints(path, key):
    return set(
        np.asarray(
            loadmat(
                path,
                squeeze_me=True
            )[key]
        )
        .reshape(-1)
        .astype(int)
        .tolist()
    )


def load_strings(path, key):
    arr = np.asarray(
        loadmat(
            path,
            squeeze_me=True,
            chars_as_strings=True
        )[key]
    ).reshape(-1)

    return {
        str(x).strip()
        for x in arr
    }


train_ids = load_ints(
    ROOT / "ID_train.mat",
    "ID_train"
)

test_ids = load_ints(
    ROOT / "ID_test.mat",
    "ID_test2"
)

train_frames = load_strings(
    ROOT / "frame_train.mat",
    "img_index_train"
)

test_frames = load_strings(
    ROOT / "frame_test.mat",
    "img_index_test"
)


candidate_keys = [
    "box_new",
    "anno_file",
    "anno_previous"
]


rows = []

bad_shapes = []
bad_dimensions = []

annotation_files = sorted(
    ANN.glob("*.mat")
)


for i, path in enumerate(annotation_files, 1):

    frame_stem = path.name.replace(
        ".jpg.mat",
        ""
    )

    frame_path = (
        FRAMES / f"{frame_stem}.jpg"
    )


    # ----------------------------------
    # frame metadata
    # ----------------------------------

    left, frame_number = frame_stem.split(
        "_",
        1
    )

    camera_part, segment_part = left.split(
        "s",
        1
    )

    camera_id = int(
        camera_part[1:]
    )

    segment_id = int(
        segment_part
    )

    frame_number = int(
        frame_number
    )


    if frame_stem in train_frames:
        frame_split = "train"

    elif frame_stem in test_frames:
        frame_split = "test"

    else:
        frame_split = "unassigned"


    # ----------------------------------
    # image dimensions
    # ----------------------------------

    with Image.open(frame_path) as im:
        image_width, image_height = im.size


    # ----------------------------------
    # annotation
    # ----------------------------------

    mat = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True
    )

    available = [
        k
        for k in candidate_keys
        if k in mat
    ]

    if len(available) != 1:
        raise RuntimeError(
            f"{path.name}: annotation keys={available}"
        )


    key = available[0]

    arr = np.asarray(
        mat[key],
        dtype=float
    )


    if arr.ndim == 1:
        arr = arr.reshape(
            1,
            -1
        )


    if arr.shape[1] != 5:
        bad_shapes.append(
            (
                path.name,
                arr.shape
            )
        )
        continue


    # ----------------------------------
    # boxes
    # ----------------------------------

    for box_index, row in enumerate(arr):

        pid = int(
            round(row[0])
        )

        x, y, w, h = map(
            float,
            row[1:]
        )


        if pid == -2:

            identity_role = (
                "unknown"
            )

        elif pid in train_ids:

            identity_role = (
                "train_id"
            )

        elif pid in test_ids:

            identity_role = (
                "test_id"
            )

        else:

            identity_role = (
                "outside_official_split"
            )


        x2 = x + w
        y2 = y + h


        positive_size = (
            w > 0
            and h > 0
        )


        fully_inside = (
            x >= 0
            and y >= 0
            and x2 <= image_width
            and y2 <= image_height
        )


        intersects_image = (
            x2 > 0
            and y2 > 0
            and x < image_width
            and y < image_height
        )


        clipped_x1 = max(
            0.0,
            x
        )

        clipped_y1 = max(
            0.0,
            y
        )

        clipped_x2 = min(
            float(image_width),
            x2
        )

        clipped_y2 = min(
            float(image_height),
            y2
        )


        clipped_w = max(
            0.0,
            clipped_x2 - clipped_x1
        )

        clipped_h = max(
            0.0,
            clipped_y2 - clipped_y1
        )


        rows.append(
            {
                "dataset":
                    "PRW",

                "frame_stem":
                    frame_stem,

                "frame_filename":
                    f"{frame_stem}.jpg",

                "frame_path":
                    str(frame_path),

                "annotation_filename":
                    path.name,

                "annotation_key":
                    key,

                "frame_split":
                    frame_split,

                "camera_id":
                    camera_id,

                "segment_id":
                    segment_id,

                "frame_number":
                    frame_number,

                "box_index":
                    box_index,

                "person_id":
                    pid,

                "identity_role":
                    identity_role,

                "x":
                    x,

                "y":
                    y,

                "w":
                    w,

                "h":
                    h,

                "x2":
                    x2,

                "y2":
                    y2,

                "image_width":
                    image_width,

                "image_height":
                    image_height,

                "positive_size":
                    positive_size,

                "fully_inside_image":
                    fully_inside,

                "intersects_image":
                    intersects_image,

                "clipped_x":
                    clipped_x1,

                "clipped_y":
                    clipped_y1,

                "clipped_w":
                    clipped_w,

                "clipped_h":
                    clipped_h,

                "recommended_source_train":
                    (
                        frame_split == "train"
                        and identity_role == "train_id"
                        and positive_size
                        and intersects_image
                    )
            }
        )


    if i % 2000 == 0 or i == len(annotation_files):
        print(
            f"processed {i:,} / "
            f"{len(annotation_files):,}"
        )


df = pd.DataFrame(rows)


print("\n" + "=" * 95)
print("PRW FINAL MANIFEST")
print("=" * 95)


print(
    "Rows:",
    len(df)
)

print(
    "Unique frames:",
    df["frame_stem"].nunique()
)

print(
    "Unique PIDs excluding -2:",
    df.loc[
        df.person_id != -2,
        "person_id"
    ].nunique()
)


print("\nFrame split × identity role:")
print(
    pd.crosstab(
        df["frame_split"],
        df["identity_role"],
        margins=True
    )
)


print("\nAnnotation keys:")
print(
    df.groupby(
        "annotation_key"
    )["frame_stem"]
    .nunique()
    .sort_values(
        ascending=False
    )
)


print("\nBBox validity:")

print(
    "Non-positive size:",
    int(
        (~df["positive_size"]).sum()
    )
)

print(
    "Not fully inside image:",
    int(
        (~df["fully_inside_image"]).sum()
    )
)

print(
    "Do not intersect image:",
    int(
        (~df["intersects_image"]).sum()
    )
)


print(
    "\nRecommended supervised source rows:",
    int(
        df[
            "recommended_source_train"
        ].sum()
    )
)


print("\nPer-camera boxes:")
print(
    df[
        "camera_id"
    ]
    .value_counts()
    .sort_index()
)


print("\nPer-camera identified PIDs:")

identified = df[
    df["person_id"] != -2
]

print(
    identified.groupby(
        "camera_id"
    )["person_id"]
    .nunique()
)


df.to_csv(
    OUT,
    index=False
)

print(
    "\nSaved:",
    OUT
)


# ============================================================
# QUERY ↔ SOURCE ANNOTATION CHECK
# ============================================================

print("\n" + "=" * 95)
print("QUERY ↔ ANNOTATION MATCH")
print("=" * 95)


query = pd.read_csv(
    ROOT / "query_info.txt",
    sep=r"\s+",
    header=None,
    names=[
        "person_id",
        "x",
        "y",
        "w",
        "h",
        "frame_stem"
    ]
)


exact = 0
near = 0
missing = []


for idx, q in query.iterrows():

    candidates = df[
        (df["frame_stem"] == q.frame_stem)
        &
        (df["person_id"] == q.person_id)
    ]


    if len(candidates) == 0:

        missing.append(
            (
                idx,
                q.person_id,
                q.frame_stem,
                "no same-PID annotation"
            )
        )

        continue


    qbox = np.array(
        [
            q.x,
            q.y,
            q.w,
            q.h
        ],
        dtype=float
    )


    distances = []

    for _, c in candidates.iterrows():

        cbox = np.array(
            [
                c.x,
                c.y,
                c.w,
                c.h
            ],
            dtype=float
        )

        distances.append(
            (
                np.max(
                    np.abs(
                        qbox - cbox
                    )
                ),
                c
            )
        )


    best_error, _ = min(
        distances,
        key=lambda z: z[0]
    )


    if best_error <= 1e-3:

        exact += 1

    elif best_error <= 0.1:

        near += 1

    else:

        missing.append(
            (
                idx,
                q.person_id,
                q.frame_stem,
                float(best_error)
            )
        )


print(
    "Queries:",
    len(query)
)

print(
    "Exact coordinate matches:",
    exact
)

print(
    "Near matches <= 0.1 px:",
    near
)

print(
    "Unmatched:",
    len(missing)
)

if missing:
    print(
        "First unmatched:",
        missing[:20]
    )


print("\nDONE")
