from pathlib import Path
import pandas as pd


ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/LPW/extracted/pep_256x128"
)

OUT = (
    Path.home()
    / "DG_ReID_Audit/manifests/LPW"
    / "lpw_manifest.csv"
)


rows = []

invalid_filenames = []


for scene_dir in sorted(ROOT.iterdir()):

    if not scene_dir.is_dir():
        continue

    scene = scene_dir.name


    for view_dir in sorted(scene_dir.iterdir()):

        if not view_dir.is_dir():
            continue

        view = view_dir.name


        for person_dir in sorted(view_dir.iterdir()):

            if not person_dir.is_dir():
                continue

            person_local_id = person_dir.name

            global_person_id = (
                f"{scene}-{person_local_id}"
            )

            tracklet_id = (
                f"{scene}-{view}-{person_local_id}"
            )


            # ----------------------------------------
            # Official split / role
            # ----------------------------------------

            if scene in {
                "scen2",
                "scen3",
            }:

                official_split = "train"
                official_role = "train"

            elif scene == "scen1":

                official_split = "test"

                if view == "view2":
                    official_role = "query"

                elif view in {
                    "view1",
                    "view3",
                }:
                    official_role = "gallery"

                else:
                    official_role = "unknown"

            else:

                official_split = "unknown"
                official_role = "unknown"


            for image_path in sorted(
                person_dir.glob("*.jpg")
            ):

                filename = image_path.name
                stem = image_path.stem


                # Filename currently appears numeric.
                # Treat this only as image_number.
                # Do NOT call it frame_id yet.
                if not stem.isdigit():

                    invalid_filenames.append(
                        {
                            "path":
                                str(image_path),

                            "filename":
                                filename,
                        }
                    )

                    image_number = None

                else:

                    image_number = int(
                        stem
                    )


                rows.append(
                    {
                        "dataset":
                            "LPW",

                        "scene":
                            scene,

                        "view":
                            view,

                        "person_local_id":
                            person_local_id,

                        "global_person_id":
                            global_person_id,

                        "tracklet_id":
                            tracklet_id,

                        "image_number":
                            image_number,

                        "official_split":
                            official_split,

                        "official_role":
                            official_role,

                        "filename":
                            filename,

                        "path":
                            str(image_path),
                    }
                )


df = pd.DataFrame(rows)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUT,
    index=False,
)


# ============================================================
# Report
# ============================================================

print("=" * 90)
print("LPW MANIFEST BUILD")
print("=" * 90)


print("\nTOTAL")
print("-" * 90)

print(
    "Rows:",
    len(df)
)

print(
    "Invalid filenames:",
    len(invalid_filenames)
)

print(
    "Unique paths:",
    df["path"].nunique()
)

print(
    "Unique global IDs:",
    df["global_person_id"].nunique()
)

print(
    "Unique tracklets:",
    df["tracklet_id"].nunique()
)


print("\nOFFICIAL ROLES")
print("-" * 90)

print(
    df["official_role"]
    .value_counts()
)


print("\nIDENTITIES PER ROLE")
print("-" * 90)

for role in [
    "train",
    "query",
    "gallery",
]:

    x = df[
        df["official_role"]
        == role
    ]

    print(
        f"{role:<8}:",
        x["global_person_id"].nunique()
    )


print("\nSCENE / VIEW")
print("-" * 90)

print(
    df.groupby(
        [
            "scene",
            "view",
        ]
    )
    .agg(
        images=(
            "filename",
            "size"
        ),
        identities=(
            "global_person_id",
            "nunique"
        ),
        tracklets=(
            "tracklet_id",
            "nunique"
        ),
    )
)


# ============================================================
# Query / Gallery PID coverage
# ============================================================

query_ids = set(
    df.loc[
        df["official_role"]
        == "query",
        "global_person_id"
    ]
)

gallery_ids = set(
    df.loc[
        df["official_role"]
        == "gallery",
        "global_person_id"
    ]
)


print("\nQUERY / GALLERY ID COVERAGE")
print("-" * 90)

print(
    "Query IDs:",
    len(query_ids)
)

print(
    "Gallery IDs:",
    len(gallery_ids)
)

print(
    "Query IDs missing from gallery:",
    len(
        query_ids
        - gallery_ids
    )
)

print(
    "Gallery-only IDs:",
    len(
        gallery_ids
        - query_ids
    )
)


# ============================================================
# Split leakage check
# ============================================================

train_ids = set(
    df.loc[
        df["official_role"]
        == "train",
        "global_person_id"
    ]
)


eval_ids = (
    query_ids
    | gallery_ids
)


print("\nTRAIN / EVAL ID OVERLAP")
print("-" * 90)

print(
    "Train IDs:",
    len(train_ids)
)

print(
    "Eval IDs:",
    len(eval_ids)
)

print(
    "Overlap:",
    len(
        train_ids
        & eval_ids
    )
)


# ============================================================
# Tracklet image statistics
# ============================================================

tracklet_sizes = (
    df.groupby(
        "tracklet_id"
    )
    .size()
)


print("\nIMAGES PER TRACKLET")
print("-" * 90)

print(
    "Tracklets:",
    len(tracklet_sizes)
)

print(
    "Min    :",
    int(
        tracklet_sizes.min()
    )
)

print(
    "Median :",
    float(
        tracklet_sizes.median()
    )
)

print(
    "Mean   :",
    round(
        float(
            tracklet_sizes.mean()
        ),
        2
    )
)

print(
    "Max    :",
    int(
        tracklet_sizes.max()
    )
)


# ============================================================
# Filename uniqueness
# ============================================================

print("\nFILENAME CHARACTERISTICS")
print("-" * 90)

print(
    "Unique filenames globally:",
    df["filename"].nunique()
)

print(
    "Total rows:",
    len(df)
)

duplicate_filename_count = (
    df["filename"]
    .duplicated(
        keep=False
    )
    .sum()
)

print(
    "Rows sharing filename with another path:",
    int(
        duplicate_filename_count
    )
)


if invalid_filenames:

    print("\nINVALID FILENAME EXAMPLES")
    print("-" * 90)

    for item in (
        invalid_filenames[:20]
    ):

        print(item)


print("\nSaved:")
print(OUT)

print("\nDONE")
