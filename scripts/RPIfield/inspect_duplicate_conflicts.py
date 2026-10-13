from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


BASE = Path.home() / "DG_ReID_Audit"

MANIFEST = (
    BASE
    / "manifests/RPIfield"
    / "rpifield_manifest.csv"
)

DUP_ROWS = (
    BASE
    / "reports/RPIfield"
    / "exact_duplicate_rows.csv"
)

OUT_DIR = (
    BASE
    / "samples/RPIfield"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CROSS_OUT = (
    OUT_DIR
    / "RPIfield_cross_person_duplicate_conflicts.jpg"
)

SAME_OUT = (
    OUT_DIR
    / "RPIfield_same_person_duplicates.jpg"
)


df = pd.read_csv(
    MANIFEST,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
    },
    low_memory=False,
)

dup = pd.read_csv(
    DUP_ROWS,
    dtype={
        "appearance_label": str,
        "sequence_id": str,
    },
    low_memory=False,
)


# ============================================================
# CROSS-PERSON CONFLICT IDENTITIES
# ============================================================

cross_groups = []

for group_id, x in dup.groupby(
    "duplicate_group"
):

    if x["person_id"].nunique() > 1:
        cross_groups.append(
            (
                group_id,
                x.copy(),
            )
        )


affected = sorted(
    set(
        dup.groupby(
            "duplicate_group"
        )
        .filter(
            lambda x:
                x["person_id"].nunique() > 1
        )[
            "person_id"
        ]
        .astype(int)
        .tolist()
    )
)


print("=" * 90)
print("CROSS-PERSON DUPLICATE IDENTITIES")
print("=" * 90)

print(
    "Affected PIDs:",
    affected
)


# ------------------------------------------------------------
# Build rows:
# one row per affected PID
# choose 8 evenly spaced samples from full sequence
# mark duplicate images
# ------------------------------------------------------------

rows = []

for pid in affected:

    x = df[
        (
            df["camera_id"] == 3
        )
        &
        (
            df["person_id"] == pid
        )
    ].sort_values(
        "frame_id"
    )

    if len(x) == 0:
        continue


    dup_paths = set(
        dup[
            dup["person_id"] == pid
        ]["path"]
    )


    n = min(
        10,
        len(x)
    )

    indices = [
        round(
            i
            * (len(x) - 1)
            / (n - 1)
        )
        for i in range(n)
    ] if n > 1 else [0]


    sampled = x.iloc[
        indices
    ].copy()


    # Force duplicate frames into sheet
    duplicate_rows = x[
        x["path"].isin(
            dup_paths
        )
    ]


    sampled = (
        pd.concat(
            [
                sampled,
                duplicate_rows,
            ]
        )
        .drop_duplicates(
            subset=["path"]
        )
        .sort_values(
            "frame_id"
        )
    )


    rows.append(
        (
            pid,
            sampled,
            dup_paths,
        )
    )


cell_w = 180
cell_h = 290

max_cols = max(
    len(x)
    for _, x, _ in rows
)

sheet = Image.new(
    "RGB",
    (
        max_cols * cell_w,
        len(rows) * cell_h + 50,
    ),
    "white"
)

draw = ImageDraw.Draw(
    sheet
)

draw.text(
    (10, 15),
    "RPIfield CROSS-PERSON DUPLICATE CONFLICT INSPECTION",
    fill="black"
)


for rr, (
    pid,
    samples,
    dup_paths,
) in enumerate(
    rows
):

    for cc, (_, row) in enumerate(
        samples.iterrows()
    ):

        x0 = cc * cell_w
        y0 = 50 + rr * cell_h


        with Image.open(
            row["path"]
        ) as img:

            img = img.convert(
                "RGB"
            )

            img.thumbnail(
                (
                    cell_w - 20,
                    210,
                )
            )

            px = (
                x0
                + (
                    cell_w
                    - img.width
                ) // 2
            )

            sheet.paste(
                img,
                (
                    px,
                    y0 + 5,
                )
            )


        is_dup = (
            row["path"]
            in dup_paths
        )

        label = (
            f"PID {pid}\n"
            f"frame={int(row['frame_id'])}\n"
            f"{'DUPLICATE' if is_dup else 'unique'}"
        )

        draw.text(
            (
                x0 + 5,
                y0 + 220,
            ),
            label,
            fill="black"
        )


sheet.save(
    CROSS_OUT,
    quality=95
)


# ============================================================
# SAME-PERSON DUPLICATES
# ============================================================

same_groups = []

for group_id, x in dup.groupby(
    "duplicate_group"
):

    if (
        x["person_id"].nunique() == 1
        and len(x) > 1
    ):

        same_groups.append(
            (
                group_id,
                x.copy(),
            )
        )


cols = 2
cell_w = 250
cell_h = 300

sheet2 = Image.new(
    "RGB",
    (
        cols * cell_w,
        len(same_groups)
        * cell_h
        + 50,
    ),
    "white"
)

draw2 = ImageDraw.Draw(
    sheet2
)

draw2.text(
    (10, 15),
    "RPIfield SAME-PERSON EXACT DUPLICATES",
    fill="black"
)


for rr, (
    group_id,
    x
) in enumerate(
    same_groups
):

    x = x.sort_values(
        "frame_id"
    )

    for cc, (_, row) in enumerate(
        x.iterrows()
    ):

        x0 = cc * cell_w
        y0 = 50 + rr * cell_h


        with Image.open(
            row["path"]
        ) as img:

            img = img.convert(
                "RGB"
            )

            img.thumbnail(
                (
                    cell_w - 20,
                    210,
                )
            )

            px = (
                x0
                + (
                    cell_w
                    - img.width
                ) // 2
            )

            sheet2.paste(
                img,
                (
                    px,
                    y0 + 5,
                )
            )


        label = (
            f"group={int(group_id)}\n"
            f"PID={int(row['person_id'])}\n"
            f"Cam={int(row['camera_id'])} "
            f"frame={int(row['frame_id'])}"
        )

        draw2.text(
            (
                x0 + 5,
                y0 + 220,
            ),
            label,
            fill="black"
        )


sheet2.save(
    SAME_OUT,
    quality=95
)


print("\nSaved:")
print(CROSS_OUT)
print(SAME_OUT)

print("\nCross-person groups:", len(cross_groups))
print("Affected PIDs:", len(affected))
print("Same-person duplicate groups:", len(same_groups))

print("\nDONE")
