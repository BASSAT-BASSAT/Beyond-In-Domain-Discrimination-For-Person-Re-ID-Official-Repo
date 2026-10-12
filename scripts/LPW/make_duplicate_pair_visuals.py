from pathlib import Path
import math

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
    / "samples/LPW/duplicate_pairs"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "global_person_id": str,
        "tracklet_id": str,
    },
)


PAIRS = [
    ("scen1-670", "scen1-672"),
    ("scen2-819", "scen2-826"),
    ("scen1-633", "scen1-644"),
    ("scen1-1053", "scen1-27"),
]


def sample_evenly(x, n=6):

    x = x.sort_values(
        "image_number"
    )

    if len(x) <= n:
        return x

    indices = [
        round(
            i * (len(x) - 1) / (n - 1)
        )
        for i in range(n)
    ]

    return x.iloc[
        indices
    ]


def make_pair_sheet(pid_a, pid_b):

    persons = [
        pid_a,
        pid_b,
    ]

    views = [
        "view1",
        "view2",
        "view3",
        "view4",
    ]

    samples = []

    for pid in persons:

        pdata = df[
            df["global_person_id"] == pid
        ]

        for view in views:

            x = pdata[
                pdata["view"] == view
            ]

            if len(x) == 0:
                continue

            sampled = sample_evenly(
                x,
                n=6
            )

            for _, row in sampled.iterrows():

                samples.append(
                    {
                        "pid": pid,
                        "view": view,
                        "path": row["path"],
                        "filename": row["filename"],
                        "image_number": row["image_number"],
                    }
                )


    cell_w = 170
    cell_h = 390
    label_h = 65

    cols = 6

    # each existing pid/view group gets one row
    groups = []

    for pid in persons:

        for view in views:

            g = [
                x for x in samples
                if (
                    x["pid"] == pid
                    and x["view"] == view
                )
            ]

            if g:
                groups.append(
                    (
                        pid,
                        view,
                        g,
                    )
                )


    rows = len(groups)

    sheet = Image.new(
        "RGB",
        (
            cols * cell_w,
            rows * cell_h + 50
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        sheet
    )

    draw.text(
        (10, 10),
        f"{pid_a}  <->  {pid_b}",
        fill="black"
    )


    for row_idx, (
        pid,
        view,
        group,
    ) in enumerate(groups):

        for col_idx, item in enumerate(group):

            path = Path(
                item["path"]
            )

            with Image.open(path) as img:

                img = img.convert(
                    "RGB"
                )

                img.thumbnail(
                    (
                        cell_w - 15,
                        cell_h - label_h - 15,
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
                    50
                    + row_idx * cell_h
                    + 5
                )

                sheet.paste(
                    img,
                    (x, y)
                )


            label_y = (
                50
                + row_idx * cell_h
                + cell_h
                - label_h
            )

            label = (
                f"{pid} | {view}\n"
                f"{item['filename']}"
            )

            draw.text(
                (
                    col_idx * cell_w + 4,
                    label_y
                ),
                label,
                fill="black"
            )


    safe_name = (
        f"{pid_a}__VS__{pid_b}.jpg"
        .replace("/", "_")
    )

    output = (
        OUT_DIR
        / safe_name
    )

    sheet.save(
        output,
        quality=95
    )

    print(
        "Saved:",
        output
    )


for a, b in PAIRS:

    make_pair_sheet(
        a,
        b
    )


print("\nDONE")
print(OUT_DIR)
