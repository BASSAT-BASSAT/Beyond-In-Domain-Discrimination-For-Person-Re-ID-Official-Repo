from pathlib import Path
import re
import pandas as pd

ROOT = Path.home() / "DG_ReID_Audit/datasets/CUHK03_NP/extracted/cuhk03-np"
OUT = Path.home() / "DG_ReID_Audit/manifests/CUHK03_NP"
OUT.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"^(?P<pid>\d+)_c(?P<view>\d+)_(?P<image_index>\d+)\.png$"
)

split_map = {
    "bounding_box_train": "train",
    "query": "query",
    "bounding_box_test": "gallery",
}

for variant in ["detected", "labeled"]:

    rows = []
    invalid = []

    for folder_name, split_name in split_map.items():

        folder = ROOT / variant / folder_name

        for path in sorted(folder.glob("*.png")):

            match = pattern.match(path.name)

            if not match:
                invalid.append(path.name)
                continue

            rows.append({
                "filename": path.name,
                "variant": variant,
                "split": split_name,
                "pid": int(match.group("pid")),
                "view": int(match.group("view")),
                "image_index": int(match.group("image_index")),
                "path": str(path),
            })

    df = pd.DataFrame(rows)

    output_file = OUT / f"{variant}_manifest.csv"
    df.to_csv(output_file, index=False)

    print("=" * 60)
    print(variant.upper())
    print("=" * 60)

    print("Parsed images :", len(df))
    print("Invalid names :", len(invalid))
    print("Views         :", sorted(df["view"].unique().tolist()))

    print("\nSplit statistics:")
    print(
        df.groupby("split")
          .agg(
              images=("filename", "count"),
              unique_pids=("pid", "nunique"),
              views=("view", "nunique"),
          )
    )

    train_ids = set(df.loc[df["split"] == "train", "pid"])
    query_ids = set(df.loc[df["split"] == "query", "pid"])
    gallery_ids = set(df.loc[df["split"] == "gallery", "pid"])

    print("\nID integrity:")
    print("Train IDs          :", len(train_ids))
    print("Query IDs          :", len(query_ids))
    print("Gallery IDs        :", len(gallery_ids))
    print("Train ∩ Query      :", len(train_ids & query_ids))
    print("Train ∩ Gallery    :", len(train_ids & gallery_ids))
    print("Query - Gallery    :", len(query_ids - gallery_ids))
    print("Gallery - Query    :", len(gallery_ids - query_ids))

    query = df[df["split"] == "query"]

    print("\nQuery samples per ID:")
    print(query.groupby("pid").size().value_counts().sort_index())

    print("\nQuery images per view:")
    print(query["view"].value_counts().sort_index())

    if invalid:
        print("\nInvalid examples:")
        print(invalid[:10])

    print("\nSaved:", output_file)
    print()
