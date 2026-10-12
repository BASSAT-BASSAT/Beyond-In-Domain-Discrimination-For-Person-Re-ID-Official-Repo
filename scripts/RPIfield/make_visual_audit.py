from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
)

OUT_DIR = (
    BASE
    / "samples/RPIfield"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42
random.seed(SEED)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
        "identity_type": str,
    },
    low_memory=False,
)


# ============================================================
# Helpers
# ============================================================

def middle_row(x):

    x = x.sort_values(
        "frame_id"
    ).reset_index(
        drop=True
    )

    return x.iloc[
        len(x) // 2
    ]


def paste_image(
    sheet,
    draw,
    row,
    x0,
    y0,
    cell_w,
    image_h,
    label,
):

    try:

        with Image.open(
            row["path"]
        ) as img:

            img = img.convert(
                "RGB"
            )

            img.thumbnail(
                (
                    cell_w - 16,
                    image_h - 10,
                )
            )

            px = (
                x0
                + (
                    cell_w
                    - img.width
                ) // 2
            )

            py = (
                y0
                + (
                    image_h
                    - img.height
                ) // 2
            )

            sheet.paste(
                img,
                (
                    px,
                    py,
                )
            )

    except Exception as e:

        draw.text(
            (
                x0 + 5,
                y0 + 10,
            ),
            str(e),
            fill="black"
        )

    draw.text(
        (
            x0 + 5,
            y0 + image_h + 3,
        ),
        label,
        fill="black"
    )


# ============================================================
# 1. CAMERA DOMAIN OVERVIEW
# ============================================================

print(
    "Creating camera overview..."
)

cols = 6
rows_n = 12

cell_w = 170
image_h = 240
label_h = 55

sheet = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows_n
        * (
            image_h
            + label_h
        )
        + 50,
    ),
    "white"
)

draw = ImageDraw.Draw(
    sheet
)

draw.text(
    (
        10,
        15,
    ),
    "RPIfield - 12 CAMERA DOMAIN OVERVIEW",
    fill="black"
)


for rr, cam in enumerate(
    range(
        1,
        13
    )
):

    cam_df = df[
        df[
            "camera_id"
        ]
        == cam
    ]

    # sample different persons
    persons = (
        cam_df[
            "person_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    selected_pids = random.sample(
        persons,
        min(
            cols,
            len(persons)
        )
    )


    for cc, pid in enumerate(
        selected_pids
    ):

        x = cam_df[
            cam_df[
                "person_id"
            ]
            == pid
        ]

        sample = middle_row(
            x
        )

        label = (
            f"Cam {cam}\n"
            f"PID {int(pid)} "
            f"({sample['identity_type']})\n"
            f"frame={int(sample['frame_id'])}"
        )

        paste_image(
            sheet,
            draw,
            sample,
            cc * cell_w,
            50
            + rr
            * (
                image_h
                + label_h
            ),
            cell_w,
            image_h,
            label,
        )


camera_out = (
    OUT_DIR
    / "RPIfield_camera_overview.jpg"
)

sheet.save(
    camera_out,
    quality=95
)

print(
    "Saved:",
    camera_out
)


# ============================================================
# 2. SAME KNOWN ACTOR ACROSS CAMERAS
# ============================================================

print(
    "Creating cross-camera known actor sheet..."
)

known = df[
    df[
        "is_known_actor"
    ]
    == True
].copy()


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
    .nunique()
    .sort_values(
        ascending=False
    )
)


# choose 6 IDs with strong camera coverage
selected_known = (
    coverage[
        coverage >= 8
    ]
    .index
    .tolist()
)

random.shuffle(
    selected_known
)

selected_known = (
    selected_known[:6]
)


cols = 12
rows_n = len(
    selected_known
)

cell_w = 145
image_h = 220
label_h = 48


sheet2 = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows_n
        * (
            image_h
            + label_h
        )
        + 50,
    ),
    "white"
)

draw2 = ImageDraw.Draw(
    sheet2
)

draw2.text(
    (
        10,
        15,
    ),
    "RPIfield - SAME KNOWN ACTOR ACROSS CAMERAS",
    fill="black"
)


for rr, pid in enumerate(
    selected_known
):

    pid_df = known[
        known[
            "person_id"
        ]
        == pid
    ]

    for cam in range(
        1,
        13
    ):

        x = pid_df[
            pid_df[
                "camera_id"
            ]
            == cam
        ]

        x0 = (
            cam - 1
        ) * cell_w

        y0 = (
            50
            + rr
            * (
                image_h
                + label_h
            )
        )


        if len(x) == 0:

            draw2.text(
                (
                    x0 + 5,
                    y0 + 70,
                ),
                f"PID {pid}\nCam {cam}\nNOT SEEN",
                fill="black"
            )

            continue


        sample = middle_row(
            x
        )

        label = (
            f"PID {int(pid)}\n"
            f"Cam {cam}\n"
            f"f={int(sample['frame_id'])}"
        )

        paste_image(
            sheet2,
            draw2,
            sample,
            x0,
            y0,
            cell_w,
            image_h,
            label,
        )


cross_out = (
    OUT_DIR
    / "RPIfield_cross_camera_known_actors.jpg"
)

sheet2.save(
    cross_out,
    quality=95
)

print(
    "Saved:",
    cross_out
)


# ============================================================
# 3. REAPPEARANCE EXAMPLES
# ============================================================

print(
    "Creating reappearance sheet..."
)


# person-camera pairs with >1 appearance label
appearance_counts = (
    known[
        [
            "person_id",
            "camera_id",
            "appearance_label",
        ]
    ]
    .drop_duplicates()
    .groupby(
        [
            "person_id",
            "camera_id",
        ]
    )[
        "appearance_label"
    ]
    .nunique()
)

candidates = (
    appearance_counts[
        appearance_counts >= 3
    ]
    .reset_index()
)


if len(
    candidates
) < 6:

    candidates = (
        appearance_counts[
            appearance_counts >= 2
        ]
        .reset_index()
    )


selected_pairs = (
    candidates.sample(
        n=min(
            6,
            len(candidates)
        ),
        random_state=SEED,
    )
)


cols = 8
rows_n = len(
    selected_pairs
)

cell_w = 165
image_h = 235
label_h = 55


sheet3 = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows_n
        * (
            image_h
            + label_h
        )
        + 50,
    ),
    "white"
)

draw3 = ImageDraw.Draw(
    sheet3
)

draw3.text(
    (
        10,
        15,
    ),
    "RPIfield - REAPPEARANCE EXAMPLES",
    fill="black"
)


for rr, pair in selected_pairs.reset_index(
    drop=True
).iterrows():

    pid = int(
        pair[
            "person_id"
        ]
    )

    cam = int(
        pair[
            "camera_id"
        ]
    )


    x = known[
        (
            known[
                "person_id"
            ]
            == pid
        )
        &
        (
            known[
                "camera_id"
            ]
            == cam
        )
    ]


    labels = (
        x[
            "appearance_label"
        ]
        .drop_duplicates()
        .tolist()
    )


    # sort by first frame
    labels = sorted(
        labels,
        key=lambda lab:
            x[
                x[
                    "appearance_label"
                ]
                == lab
            ][
                "frame_id"
            ].min()
    )


    labels = labels[
        :cols
    ]


    for cc, label_id in enumerate(
        labels
    ):

        seq = x[
            x[
                "appearance_label"
            ]
            == label_id
        ]

        sample = middle_row(
            seq
        )

        label = (
            f"PID {pid} Cam {cam}\n"
            f"label={label_id}\n"
            f"frame={int(sample['frame_id'])}"
        )

        paste_image(
            sheet3,
            draw3,
            sample,
            cc * cell_w,
            50
            + rr
            * (
                image_h
                + label_h
            ),
            cell_w,
            image_h,
            label,
        )


reapp_out = (
    OUT_DIR
    / "RPIfield_reappearance_examples.jpg"
)

sheet3.save(
    reapp_out,
    quality=95
)

print(
    "Saved:",
    reapp_out
)


# ============================================================
# 4. CROWDING / SAME CAMERA-FRAME MULTIPLE PEOPLE
# ============================================================

print(
    "Creating crowding sheet..."
)


frame_counts = (
    df.groupby(
        [
            "camera_id",
            "frame_id",
        ]
    )
    .size()
    .reset_index(
        name="crops"
    )
)


crowded = (
    frame_counts[
        frame_counts[
            "crops"
        ]
        >= 4
    ]
    .sort_values(
        "crops",
        ascending=False
    )
)


selected_frames = (
    crowded.sample(
        n=min(
            10,
            len(crowded)
        ),
        random_state=SEED,
    )
)


cols = 6
rows_n = len(
    selected_frames
)

cell_w = 170
image_h = 235
label_h = 55


sheet4 = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows_n
        * (
            image_h
            + label_h
        )
        + 50,
    ),
    "white"
)

draw4 = ImageDraw.Draw(
    sheet4
)

draw4.text(
    (
        10,
        15,
    ),
    "RPIfield - SAME CAMERA/FRAME CROWDING EXAMPLES",
    fill="black"
)


for rr, item in selected_frames.reset_index(
    drop=True
).iterrows():

    cam = int(
        item[
            "camera_id"
        ]
    )

    frame = int(
        item[
            "frame_id"
        ]
    )


    x = df[
        (
            df[
                "camera_id"
            ]
            == cam
        )
        &
        (
            df[
                "frame_id"
            ]
            == frame
        )
    ].head(
        cols
    )


    for cc, (_, sample) in enumerate(
        x.iterrows()
    ):

        label = (
            f"Cam {cam} f={frame}\n"
            f"PID={int(sample['person_id'])}\n"
            f"{sample['identity_type']}"
        )

        paste_image(
            sheet4,
            draw4,
            sample,
            cc * cell_w,
            50
            + rr
            * (
                image_h
                + label_h
            ),
            cell_w,
            image_h,
            label,
        )


crowd_out = (
    OUT_DIR
    / "RPIfield_crowding_examples.jpg"
)

sheet4.save(
    crowd_out,
    quality=95
)

print(
    "Saved:",
    crowd_out
)


print("\nDONE")
print(OUT_DIR)
