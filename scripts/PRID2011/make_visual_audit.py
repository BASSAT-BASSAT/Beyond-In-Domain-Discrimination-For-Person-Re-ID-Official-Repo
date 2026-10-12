from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/PRID2011"
    / "prid2011_manifest.csv"
)

OUT_DIR = (
    BASE
    / "samples/PRID2011"
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
        "global_person_id": str,
        "trajectory_id": str,
    },
)

multi = df[
    df["version"] == "multi_shot"
].copy()


def middle_image(x):

    x = x.sort_values(
        "sequence_index"
    )

    return x.iloc[
        len(x) // 2
    ]


def evenly_sample(x, n=6):

    x = x.sort_values(
        "sequence_index"
    )

    if len(x) <= n:
        return x

    idxs = [
        round(
            i * (len(x) - 1) / (n - 1)
        )
        for i in range(n)
    ]

    return x.iloc[idxs]


# ============================================================
# 1. MATCHED CROSS-CAMERA IDENTITIES
# ============================================================

matched_ids = [
    f"shared-{i:04d}"
    for i in range(1, 201)
]

selected_ids = random.sample(
    matched_ids,
    12
)


cell_w = 180
cell_h = 330
label_h = 55

sheet = Image.new(
    "RGB",
    (
        2 * cell_w,
        len(selected_ids) * cell_h + 50,
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)

draw.text(
    (10, 15),
    "PRID2011 MATCHED CROSS-CAMERA IDENTITIES",
    fill="black"
)


for row_idx, pid in enumerate(
    selected_ids
):

    for col_idx, camera in enumerate(
        [
            "cam_a",
            "cam_b",
        ]
    ):

        x = multi[
            (multi["global_person_id"] == pid)
            &
            (multi["camera"] == camera)
        ]

        sample = middle_image(x)

        with Image.open(
            sample["path"]
        ) as img:

            img = img.convert("RGB")

            img.thumbnail(
                (
                    cell_w - 20,
                    cell_h - label_h - 10,
                )
            )

            px = (
                col_idx * cell_w
                + (
                    cell_w
                    - img.width
                ) // 2
            )

            py = (
                50
                + row_idx * cell_h
            )

            sheet.paste(
                img,
                (px, py)
            )


        label = (
            f"{pid}\n"
            f"{camera} | "
            f"{sample['filename']}"
        )

        draw.text(
            (
                col_idx * cell_w + 5,
                50
                + row_idx * cell_h
                + cell_h
                - label_h
            ),
            label,
            fill="black"
        )


out = (
    OUT_DIR
    / "PRID2011_matched_cross_camera.jpg"
)

sheet.save(
    out,
    quality=95
)

print("Saved:", out)


# ============================================================
# 2/3. TRAJECTORY VISUALS
# ============================================================

def make_trajectory_sheet(camera):

    camera_df = multi[
        multi["camera"] == camera
    ]

    trajectory_ids = sorted(
        camera_df[
            "trajectory_id"
        ].unique()
    )

    selected = random.sample(
        trajectory_ids,
        8
    )


    cols = 6

    cell_w = 170
    image_h = 300
    label_h = 60

    top_h = 50

    sheet = Image.new(
        "RGB",
        (
            cols * cell_w,
            top_h
            + len(selected)
            * (
                image_h
                + label_h
            ),
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        sheet
    )

    draw.text(
        (10, 15),
        f"PRID2011 {camera.upper()} TRAJECTORY SAMPLES",
        fill="black"
    )


    for row_idx, trajectory_id in enumerate(
        selected
    ):

        x = camera_df[
            camera_df[
                "trajectory_id"
            ] == trajectory_id
        ]

        samples = evenly_sample(
            x,
            n=6
        )


        for col_idx, (_, sample) in enumerate(
            samples.iterrows()
        ):

            with Image.open(
                sample["path"]
            ) as img:

                img = img.convert("RGB")

                img.thumbnail(
                    (
                        cell_w - 15,
                        image_h - 10,
                    )
                )

                px = (
                    col_idx * cell_w
                    + (
                        cell_w
                        - img.width
                    ) // 2
                )

                py = (
                    top_h
                    + row_idx
                    * (
                        image_h
                        + label_h
                    )
                )

                sheet.paste(
                    img,
                    (px, py)
                )


            label = (
                f"{trajectory_id}\n"
                f"seq={int(sample['sequence_index'])}"
            )

            draw.text(
                (
                    col_idx * cell_w + 5,
                    top_h
                    + row_idx
                    * (
                        image_h
                        + label_h
                    )
                    + image_h
                    + 3
                ),
                label,
                fill="black"
            )


    output = (
        OUT_DIR
        / f"PRID2011_{camera}_sequences.jpg"
    )

    sheet.save(
        output,
        quality=95
    )

    print(
        "Saved:",
        output
    )


make_trajectory_sheet(
    "cam_a"
)

make_trajectory_sheet(
    "cam_b"
)


print("\nDONE")
print(OUT_DIR)
