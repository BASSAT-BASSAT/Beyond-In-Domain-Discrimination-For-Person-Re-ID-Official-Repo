from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/LPW"
    / "lpw_manifest.csv"
)

OUT_DIR = (
    BASE
    / "samples/LPW/general_visual"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42
SAMPLES_PER_VIEW = 6

random.seed(SEED)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "person_local_id": str,
        "global_person_id": str,
        "tracklet_id": str,
    },
)


# Known duplicate-conflict identities.
# We exclude them from the GENERAL visual sample
# because they were already inspected separately.
EXCLUDE_IDS = {
    "scen1-1053",
    "scen1-27",
    "scen1-633",
    "scen1-644",
    "scen1-670",
    "scen1-672",
    "scen2-819",
    "scen2-826",
}


def middle_image(group):

    group = group.sort_values(
        "image_number"
    )

    return group.iloc[
        len(group) // 2
    ]


def get_view_samples(scene, view):

    x = df[
        (df["scene"] == scene)
        &
        (df["view"] == view)
        &
        (~df["global_person_id"].isin(EXCLUDE_IDS))
    ]


    persons = sorted(
        x["global_person_id"]
        .unique()
        .tolist()
    )


    selected_persons = random.sample(
        persons,
        min(
            SAMPLES_PER_VIEW,
            len(persons)
        )
    )


    rows = []

    for pid in selected_persons:

        px = x[
            x["global_person_id"] == pid
        ]

        row = middle_image(
            px
        )

        rows.append(
            row
        )


    return rows


def make_scene_sheet(scene):

    scene_df = df[
        df["scene"] == scene
    ]

    views = sorted(
        scene_df["view"]
        .unique()
    )


    all_rows = []

    for view in views:

        samples = get_view_samples(
            scene,
            view
        )

        all_rows.append(
            (
                view,
                samples,
            )
        )


    cell_w = 180
    image_h = 360
    label_h = 80

    cols = SAMPLES_PER_VIEW
    rows_count = len(all_rows)

    top_h = 55

    sheet = Image.new(
        "RGB",
        (
            cols * cell_w,
            top_h
            + rows_count
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
        f"LPW GENERAL VISUAL AUDIT - {scene.upper()}",
        fill="black"
    )


    for row_idx, (
        view,
        samples,
    ) in enumerate(all_rows):

        row_y = (
            top_h
            + row_idx
            * (
                image_h
                + label_h
            )
        )

        for col_idx, sample in enumerate(samples):

            path = Path(
                sample["path"]
            )

            with Image.open(path) as img:

                img = img.convert(
                    "RGB"
                )

                img.thumbnail(
                    (
                        cell_w - 20,
                        image_h - 10,
                    )
                )

                x = (
                    col_idx * cell_w
                    + (
                        cell_w
                        - img.width
                    ) // 2
                )

                y = (
                    row_y
                    + (
                        image_h
                        - img.height
                    ) // 2
                )

                sheet.paste(
                    img,
                    (x, y)
                )


            label = (
                f"{view}\n"
                f"{sample['global_person_id']}\n"
                f"{sample['filename']}"
            )

            draw.text(
                (
                    col_idx * cell_w + 5,
                    row_y + image_h + 5
                ),
                label,
                fill="black"
            )


    output = (
        OUT_DIR
        / f"LPW_visual_{scene}.jpg"
    )

    sheet.save(
        output,
        quality=95
    )

    print(
        "Saved:",
        output
    )


for scene in [
    "scen1",
    "scen2",
    "scen3",
]:

    make_scene_sheet(
        scene
    )


print("\nDONE")
print(OUT_DIR)
