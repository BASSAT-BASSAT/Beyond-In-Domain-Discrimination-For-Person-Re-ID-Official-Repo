from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/ORBench/extracted/ORBench"
)

TRAIN_JSON = ROOT / "train_annos.json"
TEST_JSON = ROOT / "test_gallery_and_queries.json"

OUT_DIR = BASE / "manifests/ORBench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "orbench_rgb_manifest.csv"


# ============================================================
# LOAD
# ============================================================

with open(
    TRAIN_JSON,
    "r",
    encoding="utf-8"
) as f:
    train = json.load(f)

with open(
    TEST_JSON,
    "r",
    encoding="utf-8"
) as f:
    test = json.load(f)


# ============================================================
# PATH PARSER
# ============================================================

def parse_vis_path(path_str):

    p = Path(path_str)

    if p.parts[0] != "vis":
        raise ValueError(
            f"Non-VIS path: {path_str}"
        )

    folder_id = int(
        p.parts[1]
    )

    name = p.name

    pattern = re.compile(
        r"^(?P<orbench_id>\d+)_"
        r"(?P<source>llcm|sysu)_"
        r"(?P<source_pid>\d+)_"
        r"(?P<rest>.+)_vis\.jpg$",
        re.IGNORECASE,
    )

    m = pattern.match(name)

    if not m:
        raise ValueError(
            f"Unrecognized VIS filename: {name}"
        )


    orbench_id = int(
        m.group("orbench_id")
    )

    source = (
        m.group("source")
        .lower()
    )

    source_pid = int(
        m.group("source_pid")
    )

    rest = m.group(
        "rest"
    )


    camera = None
    sequence = None
    frame = None


    # LLCM-like:
    # c03_s184251_f31995
    cam_match = re.search(
        r"(?:^|_)c(\d+)(?:_|$)",
        rest
    )

    if cam_match:
        camera = int(
            cam_match.group(1)
        )


    # possible SYSU naming:
    # cam1 / cam2 / ...
    if camera is None:

        cam_match = re.search(
            r"(?:^|_)cam(\d+)(?:_|$)",
            rest,
            flags=re.IGNORECASE,
        )

        if cam_match:
            camera = int(
                cam_match.group(1)
            )


    seq_match = re.search(
        r"(?:^|_)s(\d+)(?:_|$)",
        rest
    )

    if seq_match:
        sequence = int(
            seq_match.group(1)
        )


    frame_match = re.search(
        r"(?:^|_)f(\d+)(?:_|$)",
        rest
    )

    if frame_match:
        frame = int(
            frame_match.group(1)
        )


    return {
        "folder_id":
            folder_id,

        "orbench_id":
            orbench_id,

        "source_dataset":
            source,

        "source_pid":
            source_pid,

        "camera_id":
            camera,

        "sequence_id":
            sequence,

        "frame_number":
            frame,

        "filename_rest":
            rest,
    }


# ============================================================
# BUILD RECORDS
# ============================================================

records = []


for item in train:

    parsed = parse_vis_path(
        item["file_path"]
    )

    records.append(
        {
            "split":
                "train",

            "json_label":
                int(
                    item["id"]
                ),

            "file_path":
                item["file_path"],

            "absolute_path":
                str(
                    ROOT
                    / item["file_path"]
                ),

            **parsed,
        }
    )


for item in test[
    "RGB_GALLERY"
]:

    label = int(
        item[0]
    )

    path_str = item[1]

    parsed = parse_vis_path(
        path_str
    )

    records.append(
        {
            "split":
                "test_gallery",

            "json_label":
                label,

            "file_path":
                path_str,

            "absolute_path":
                str(
                    ROOT
                    / path_str
                ),

            **parsed,
        }
    )


df = pd.DataFrame(
    records
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("=" * 100)
print("ORBENCH RGB MANIFEST")
print("=" * 100)


print(
    "Rows:",
    len(df)
)

print(
    "\nSplit counts:"
)

print(
    df[
        "split"
    ].value_counts()
)


print(
    "\nUnique JSON labels:"
)

print(
    df.groupby(
        "split"
    )[
        "json_label"
    ].nunique()
)


print(
    "\nUnique ORBench folder IDs:"
)

print(
    df.groupby(
        "split"
    )[
        "orbench_id"
    ].nunique()
)


# ============================================================
# FOLDER / FILENAME CONSISTENCY
# ============================================================

folder_mismatch = df[
    df[
        "folder_id"
    ]
    !=
    df[
        "orbench_id"
    ]
]


print(
    "\nFolder ID != filename ORBench ID:",
    len(
        folder_mismatch
    )
)


# ============================================================
# JSON LABEL -> ORBENCH ID MAPPING
# ============================================================

label_to_ids = (
    df.groupby(
        [
            "split",
            "json_label",
        ]
    )[
        "orbench_id"
    ]
    .nunique()
)


bad_label_mapping = (
    label_to_ids[
        label_to_ids != 1
    ]
)


id_to_labels = (
    df.groupby(
        [
            "split",
            "orbench_id",
        ]
    )[
        "json_label"
    ]
    .nunique()
)


bad_id_mapping = (
    id_to_labels[
        id_to_labels != 1
    ]
)


print("\n" + "=" * 100)
print("LABEL MAPPING VALIDATION")
print("=" * 100)


print(
    "JSON labels mapping to != 1 ORBench ID:",
    len(
        bad_label_mapping
    )
)

print(
    "ORBench IDs mapping to != 1 JSON label:",
    len(
        bad_id_mapping
    )
)


if len(
    bad_label_mapping
):

    print(
        "\nBad label mappings:"
    )

    print(
        bad_label_mapping.head(
            30
        )
    )


if len(
    bad_id_mapping
):

    print(
        "\nBad ORBench-ID mappings:"
    )

    print(
        bad_id_mapping.head(
            30
        )
    )


# ============================================================
# TRAIN / TEST OVERLAP
# ============================================================

train = df[
    df[
        "split"
    ]
    == "train"
]

test = df[
    df[
        "split"
    ]
    == "test_gallery"
]


train_orbench_ids = set(
    train[
        "orbench_id"
    ]
)

test_orbench_ids = set(
    test[
        "orbench_id"
    ]
)


train_original = set(
    zip(
        train[
            "source_dataset"
        ],
        train[
            "source_pid"
        ],
    )
)

test_original = set(
    zip(
        test[
            "source_dataset"
        ],
        test[
            "source_pid"
        ],
    )
)


print("\n" + "=" * 100)
print("TRAIN / TEST LEAKAGE")
print("=" * 100)


print(
    "ORBench-ID overlap:",
    len(
        train_orbench_ids
        &
        test_orbench_ids
    )
)

print(
    "Original-source PID overlap:",
    len(
        train_original
        &
        test_original
    )
)


# ============================================================
# COVERAGE
# ============================================================

all_vis_files = {
    str(
        p.relative_to(
            ROOT
        )
    )
    for p in (
        ROOT
        / "vis"
    ).glob(
        "*/*.jpg"
    )
}


json_vis_files = set(
    df[
        "file_path"
    ]
)


print("\n" + "=" * 100)
print("RGB FILE COVERAGE")
print("=" * 100)


print(
    "Filesystem VIS images:",
    len(
        all_vis_files
    )
)

print(
    "JSON-assigned VIS images:",
    len(
        json_vis_files
    )
)

print(
    "Filesystem VIS not in JSON:",
    len(
        all_vis_files
        -
        json_vis_files
    )
)

print(
    "JSON VIS missing from filesystem:",
    len(
        json_vis_files
        -
        all_vis_files
    )
)


# ============================================================
# SOURCE ORIGINS
# ============================================================

print("\n" + "=" * 100)
print("ORIGIN DATASET DISTRIBUTION")
print("=" * 100)


origin_table = (
    df.groupby(
        [
            "split",
            "source_dataset",
        ]
    )
    .agg(
        images=(
            "file_path",
            "size"
        ),
        identities=(
            "orbench_id",
            "nunique"
        ),
    )
)


print(
    origin_table
)


# ============================================================
# CAMERA PARSING COVERAGE
# ============================================================

print("\n" + "=" * 100)
print("CAMERA PARSING COVERAGE")
print("=" * 100)


for source in sorted(
    df[
        "source_dataset"
    ].unique()
):

    g = df[
        df[
            "source_dataset"
        ]
        == source
    ]

    parsed = int(
        g[
            "camera_id"
        ].notna().sum()
    )


    print(
        f"{source}: "
        f"{parsed:,}/{len(g):,} "
        f"({100 * parsed / len(g):.2f}%) "
        f"images with parsed camera"
    )


    print(
        "  cameras:",
        sorted(
            g[
                "camera_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    )


# ============================================================
# IMAGE COUNT PER TRAIN ID
# ============================================================

counts = (
    train.groupby(
        "json_label"
    )
    .size()
)


print("\n" + "=" * 100)
print("TRAIN IMAGES PER ID")
print("=" * 100)


print(
    counts.describe(
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


print(
    "IDs with 1 image:",
    int(
        (
            counts == 1
        ).sum()
    )
)

print(
    "IDs with < 4 images:",
    int(
        (
            counts < 4
        ).sum()
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)

print("\nDONE")
