from pathlib import Path
import hashlib
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/PRID2011/extracted/prid_2011"
)


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


print("=" * 90)
print("PRID2011 SINGLE-SHOT vs MULTI-SHOT OVERLAP")
print("=" * 90)


rows = []


for camera in [
    "cam_a",
    "cam_b",
]:

    single_dir = (
        ROOT
        / "single_shot"
        / camera
    )

    multi_dir = (
        ROOT
        / "multi_shot"
        / camera
    )


    for single_path in sorted(
        single_dir.glob("*.png")
    ):

        person_name = single_path.stem

        trajectory_dir = (
            multi_dir
            / person_name
        )

        single_hash = sha256(
            single_path
        )


        matches = []

        for multi_path in trajectory_dir.glob("*.png"):

            if sha256(multi_path) == single_hash:

                matches.append(
                    multi_path.name
                )


        rows.append(
            {
                "camera":
                    camera,

                "person":
                    person_name,

                "single_path":
                    str(single_path),

                "matches_in_multi":
                    len(matches),

                "matched_files":
                    ";".join(matches),
            }
        )


df = pd.DataFrame(rows)


print("\nTOTAL")
print("-" * 90)

print(
    "Single-shot images:",
    len(df)
)

print(
    "With exact match in multi-shot:",
    int(
        (
            df["matches_in_multi"] > 0
        ).sum()
    )
)

print(
    "Without exact match:",
    int(
        (
            df["matches_in_multi"] == 0
        ).sum()
    )
)

print(
    "With >1 exact match:",
    int(
        (
            df["matches_in_multi"] > 1
        ).sum()
    )
)


print("\nBY CAMERA")
print("-" * 90)

for camera in [
    "cam_a",
    "cam_b",
]:

    x = df[
        df["camera"] == camera
    ]

    print(
        f"\n{camera}"
    )

    print(
        "Total:",
        len(x)
    )

    print(
        "Matched:",
        int(
            (
                x["matches_in_multi"] > 0
            ).sum()
        )
    )

    print(
        "Unmatched:",
        int(
            (
                x["matches_in_multi"] == 0
            ).sum()
        )
    )


print("\nMATCH COUNT DISTRIBUTION")
print("-" * 90)

print(
    df[
        "matches_in_multi"
    ]
    .value_counts()
    .sort_index()
)


unmatched = df[
    df[
        "matches_in_multi"
    ] == 0
]


if len(unmatched) > 0:

    print("\nUNMATCHED EXAMPLES")
    print("-" * 90)

    print(
        unmatched.head(20)
        .to_string(
            index=False
        )
    )


OUT = (
    BASE
    / "manifests/PRID2011"
    / "single_multi_overlap.csv"
)

df.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)

print("\nDONE")
