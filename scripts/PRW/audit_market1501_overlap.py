from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

PRW_MANIFEST = (
    BASE
    / "manifests/PRW"
    / "prw_source_crops_manifest.csv"
)

MARKET_ROOT = (
    BASE
    / "datasets/Market1501/extracted/Market-1501-v15.09.15"
)

OUT_DIR = (
    BASE
    / "reports/PRW"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MARKET_DIRS = {
    "train":
        MARKET_ROOT / "bounding_box_train",

    "gallery":
        MARKET_ROOT / "bounding_box_test",

    "query":
        MARKET_ROOT / "query",
}


# ============================================================
# HEADER / PATH CHECK
# ============================================================

print("=" * 105)
print("PRW ↔ MARKET1501 CROSS-DATASET ORIGIN / LEAKAGE AUDIT")
print("=" * 105)


print("\nMarket root:")
print(MARKET_ROOT)


for split, folder in MARKET_DIRS.items():

    print(
        f"{split:<8} "
        f"exists={folder.is_dir()} "
        f"path={folder}"
    )


missing = [
    str(folder)
    for folder in MARKET_DIRS.values()
    if not folder.is_dir()
]

if missing:
    raise FileNotFoundError(
        "Missing Market1501 directories:\n"
        + "\n".join(missing)
    )


# ============================================================
# LOAD PRW CLEAN SOURCE
# ============================================================

prw = pd.read_csv(
    PRW_MANIFEST,
    low_memory=False
)


required_prw_columns = {
    "person_id",
    "camera_id",
    "segment_id",
    "frame_number",
    "frame_stem",
    "crop_path",
}

missing_cols = (
    required_prw_columns
    - set(prw.columns)
)

if missing_cols:
    raise RuntimeError(
        f"Missing PRW columns: "
        f"{sorted(missing_cols)}"
    )


# Canonical raw-video frame coordinate.
prw["origin_key"] = (
    "c"
    + prw["camera_id"]
        .astype(int)
        .astype(str)
    + "s"
    + prw["segment_id"]
        .astype(int)
        .astype(str)
    + "_"
    + prw["frame_number"]
        .astype(int)
        .astype(str)
        .str.zfill(6)
)


prw_frames = (
    prw[
        [
            "origin_key",
            "camera_id",
            "segment_id",
            "frame_number",
        ]
    ]
    .drop_duplicates(
        "origin_key"
    )
    .copy()
)


print("\n" + "=" * 105)
print("PRW CLEAN SOURCE")
print("=" * 105)

print(
    "Source crops:",
    len(prw)
)

print(
    "Source identities:",
    prw["person_id"].nunique()
)

print(
    "Unique source frames:",
    len(prw_frames)
)

print(
    "Cameras:",
    sorted(
        prw["camera_id"]
        .astype(int)
        .unique()
        .tolist()
    )
)


# ============================================================
# PARSE MARKET1501 FILENAMES
#
# Example:
# 0002_c1s1_000451_03.jpg
# -1_c1s1_000401_03.jpg
# ============================================================

PATTERN = re.compile(
    r"^(?P<pid>-?\d+)_"
    r"c(?P<camera>\d+)"
    r"s(?P<segment>\d+)_"
    r"(?P<frame>\d+)_"
    r"(?P<box>\d+)(?:\.jpg)+$",
    re.IGNORECASE,
)


market_rows = []
unparsed = []


for split, folder in MARKET_DIRS.items():

    files = sorted(
        folder.glob("*.jpg")
    )

    print(
        f"\nParsing Market {split}: "
        f"{len(files):,} JPG files"
    )


    for path in files:

        match = PATTERN.match(
            path.name
        )


        if match is None:

            unparsed.append(
                {
                    "market_split":
                        split,

                    "filename":
                        path.name,
                }
            )

            continue


        pid = int(
            match.group("pid")
        )

        camera_id = int(
            match.group("camera")
        )

        segment_id = int(
            match.group("segment")
        )

        frame_number = int(
            match.group("frame")
        )

        box_number = int(
            match.group("box")
        )


        origin_key = (
            f"c{camera_id}"
            f"s{segment_id}_"
            f"{frame_number:06d}"
        )


        market_rows.append(
            {
                "market_split":
                    split,

                "market_pid":
                    pid,

                "camera_id":
                    camera_id,

                "segment_id":
                    segment_id,

                "frame_number":
                    frame_number,

                "market_box_number":
                    box_number,

                "origin_key":
                    origin_key,

                "market_filename":
                    path.name,

                "market_path":
                    str(path),
            }
        )


market = pd.DataFrame(
    market_rows
)


print("\n" + "=" * 105)
print("MARKET1501 PARSE CHECK")
print("=" * 105)


print(
    "Parsed JPGs:",
    len(market)
)

print(
    "Unparsed JPGs:",
    len(unparsed)
)


if unparsed:

    print(
        "\nFirst unparsed filenames:"
    )

    print(
        pd.DataFrame(
            unparsed
        )
        .head(20)
        .to_string(index=False)
    )


print(
    "\nImages by split:"
)

print(
    market[
        "market_split"
    ]
    .value_counts()
)


print(
    "\nUnique underlying raw frames by split:"
)

print(
    market.groupby(
        "market_split"
    )[
        "origin_key"
    ]
    .nunique()
)


# ============================================================
# EXACT RAW-FRAME OVERLAP
# ============================================================

prw_frame_set = set(
    prw_frames[
        "origin_key"
    ]
)


summary_rows = []


for split in [
    "train",
    "gallery",
    "query",
]:

    m = market[
        market[
            "market_split"
        ]
        == split
    ].copy()


    market_frame_set = set(
        m[
            "origin_key"
        ]
    )


    shared_frames = (
        prw_frame_set
        &
        market_frame_set
    )


    market_shared = m[
        m[
            "origin_key"
        ]
        .isin(
            shared_frames
        )
    ]


    prw_shared = prw[
        prw[
            "origin_key"
        ]
        .isin(
            shared_frames
        )
    ]


    summary_rows.append(
        {
            "market_split":
                split,

            "market_images":
                len(m),

            "market_unique_frames":
                len(
                    market_frame_set
                ),

            "shared_exact_frames":
                len(
                    shared_frames
                ),

            "prw_crops_on_shared_frames":
                len(
                    prw_shared
                ),

            "prw_ids_on_shared_frames":
                prw_shared[
                    "person_id"
                ].nunique(),

            "market_images_on_shared_frames":
                len(
                    market_shared
                ),

            "market_ids_on_shared_frames":
                market_shared.loc[
                    market_shared[
                        "market_pid"
                    ]
                    >= 0,
                    "market_pid",
                ].nunique(),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print("\n" + "=" * 105)
print("EXACT RAW-FRAME OVERLAP")
print("=" * 105)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# CRITICAL TARGET OVERLAP
#
# Market unseen evaluation target:
# query + gallery
# ============================================================

market_target = market[
    market[
        "market_split"
    ]
    .isin(
        [
            "query",
            "gallery",
        ]
    )
].copy()


market_target_frame_set = set(
    market_target[
        "origin_key"
    ]
)


shared_target_frames = (
    prw_frame_set
    &
    market_target_frame_set
)


prw_target_overlap = prw[
    prw[
        "origin_key"
    ]
    .isin(
        shared_target_frames
    )
].copy()


market_target_overlap = market_target[
    market_target[
        "origin_key"
    ]
    .isin(
        shared_target_frames
    )
].copy()


print("\n" + "=" * 105)
print("CRITICAL: PRW SOURCE ↔ MARKET1501 TARGET")
print("=" * 105)


print(
    "PRW source unique frames:",
    len(
        prw_frame_set
    )
)

print(
    "Market target unique frames:",
    len(
        market_target_frame_set
    )
)

print(
    "EXACT shared raw frames:",
    len(
        shared_target_frames
    )
)


print(
    "PRW training crops from shared target frames:",
    len(
        prw_target_overlap
    )
)

print(
    "PRW source IDs affected:",
    prw_target_overlap[
        "person_id"
    ].nunique()
)


print(
    "Market query/gallery images on shared frames:",
    len(
        market_target_overlap
    )
)

print(
    "Market non-junk IDs affected:",
    market_target_overlap.loc[
        market_target_overlap[
            "market_pid"
        ]
        >= 0,
        "market_pid",
    ].nunique()
)


if shared_target_frames:

    print(
        "\nFirst 50 exact shared frames:"
    )

    for key in sorted(
        shared_target_frames
    )[:50]:

        print(key)


# ============================================================
# DETAILED EXACT OVERLAP TABLE
# ============================================================

if shared_target_frames:

    overlap_detail = (
        prw_target_overlap[
            [
                "person_id",
                "camera_id",
                "segment_id",
                "frame_number",
                "frame_stem",
                "origin_key",
                "crop_path",
            ]
        ]
        .merge(
            market_target[
                [
                    "market_split",
                    "market_pid",
                    "origin_key",
                    "market_filename",
                    "market_path",
                ]
            ],
            on="origin_key",
            how="inner",
        )
    )

else:

    overlap_detail = pd.DataFrame()


# ============================================================
# TEMPORAL PROXIMITY
#
# For every unique PRW source frame:
# find nearest Market TARGET frame
# within SAME camera + SAME segment.
# ============================================================

print("\n" + "=" * 105)
print("TEMPORAL PROXIMITY TO MARKET TARGET")
print("=" * 105)


target_frame_groups = {}


for (
    camera_id,
    segment_id
), g in market_target.groupby(
    [
        "camera_id",
        "segment_id",
    ]
):

    arr = np.sort(
        g[
            "frame_number"
        ]
        .astype(int)
        .unique()
    )


    target_frame_groups[
        (
            int(camera_id),
            int(segment_id),
        )
    ] = arr


nearest_rows = []


for _, row in prw_frames.iterrows():

    camera_id = int(
        row[
            "camera_id"
        ]
    )

    segment_id = int(
        row[
            "segment_id"
        ]
    )

    frame_number = int(
        row[
            "frame_number"
        ]
    )


    target_frames = (
        target_frame_groups.get(
            (
                camera_id,
                segment_id,
            )
        )
    )


    if (
        target_frames is None
        or len(target_frames) == 0
    ):

        nearest_rows.append(
            {
                "origin_key":
                    row[
                        "origin_key"
                    ],

                "camera_id":
                    camera_id,

                "segment_id":
                    segment_id,

                "frame_number":
                    frame_number,

                "nearest_target_frame":
                    np.nan,

                "nearest_target_gap":
                    np.nan,
            }
        )

        continue


    pos = np.searchsorted(
        target_frames,
        frame_number
    )


    candidates = []


    if pos < len(
        target_frames
    ):

        candidates.append(
            int(
                target_frames[
                    pos
                ]
            )
        )


    if pos > 0:

        candidates.append(
            int(
                target_frames[
                    pos - 1
                ]
            )
        )


    nearest_frame = min(
        candidates,
        key=lambda x:
            abs(
                x
                - frame_number
            )
    )


    nearest_gap = abs(
        nearest_frame
        - frame_number
    )


    nearest_rows.append(
        {
            "origin_key":
                row[
                    "origin_key"
                ],

            "camera_id":
                camera_id,

            "segment_id":
                segment_id,

            "frame_number":
                frame_number,

            "nearest_target_frame":
                nearest_frame,

            "nearest_target_gap":
                nearest_gap,
        }
    )


nearest = pd.DataFrame(
    nearest_rows
)


comparable = nearest[
    nearest[
        "nearest_target_gap"
    ].notna()
].copy()


print(
    "PRW source frames:",
    len(
        nearest
    )
)

print(
    "PRW frames with a Market target "
    "in same camera+segment:",
    len(
        comparable
    )
)


if len(comparable):

    print(
        "\nNearest target-frame gap statistics:"
    )

    print(
        comparable[
            "nearest_target_gap"
        ]
        .describe(
            percentiles=[
                .01,
                .05,
                .25,
                .50,
                .75,
                .90,
                .95,
                .99,
            ]
        )
        .round(2)
    )


    print(
        "\nThreshold counts:"
    )


    for threshold in [
        0,
        25,
        50,
        100,
        250,
        500,
        1000,
        5000,
        10000,
    ]:

        count = int(
            (
                comparable[
                    "nearest_target_gap"
                ]
                <= threshold
            ).sum()
        )

        percentage = (
            100
            * count
            / len(
                comparable
            )
        )


        print(
            f"gap <= {threshold:5d}: "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# CAMERA / SEGMENT OVERLAP
# ============================================================

prw_cs = set(
    zip(
        prw_frames[
            "camera_id"
        ].astype(int),

        prw_frames[
            "segment_id"
        ].astype(int),
    )
)


market_target_cs = set(
    zip(
        market_target[
            "camera_id"
        ].astype(int),

        market_target[
            "segment_id"
        ].astype(int),
    )
)


print("\n" + "=" * 105)
print("CAMERA / SEGMENT ACQUISITION OVERLAP")
print("=" * 105)


print(
    "PRW source camera-segments:",
    sorted(
        prw_cs
    )
)

print(
    "Market target camera-segments:",
    sorted(
        market_target_cs
    )
)

print(
    "Shared camera-segments:",
    sorted(
        prw_cs
        &
        market_target_cs
    )
)


# ============================================================
# SAVE
# ============================================================

summary_path = (
    OUT_DIR
    / "prw_market1501_overlap_summary.csv"
)

exact_path = (
    OUT_DIR
    / "prw_market1501_exact_target_overlap.csv"
)

temporal_path = (
    OUT_DIR
    / "prw_market1501_temporal_proximity.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


overlap_detail.to_csv(
    exact_path,
    index=False
)


nearest.to_csv(
    temporal_path,
    index=False
)


print("\n" + "=" * 105)
print("SAVED")
print("=" * 105)

print(summary_path)
print(exact_path)
print(temporal_path)


print("\nDONE")
