from pathlib import Path
import pandas as pd

MANIFEST_DIR = Path.home() / "DG_ReID_Audit/manifests/CUHK03_NP"
REPORT_DIR = Path.home() / "DG_ReID_Audit/reports/CUHK03_NP"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

for variant in ["detected", "labeled"]:

    df = pd.read_csv(
        MANIFEST_DIR / f"{variant}_manifest.csv"
    )

    train = df[df["split"] == "train"].copy()
    query = df[df["split"] == "query"].copy()
    gallery = df[df["split"] == "gallery"].copy()

    print("\n" + "=" * 70)
    print(variant.upper())
    print("=" * 70)

    # ------------------------------------------------
    # 1. Exact filename overlap between splits
    # ------------------------------------------------

    train_files = set(train["filename"])
    query_files = set(query["filename"])
    gallery_files = set(gallery["filename"])

    print("\nExact filename overlap:")

    print(
        "Train ∩ Query   :",
        len(train_files & query_files)
    )

    print(
        "Train ∩ Gallery :",
        len(train_files & gallery_files)
    )

    print(
        "Query ∩ Gallery :",
        len(query_files & gallery_files)
    )

    # ------------------------------------------------
    # 2. Positive gallery samples for every query
    # ------------------------------------------------

    rows = []

    for _, q in query.iterrows():

        positives = gallery[
            gallery["pid"] == q["pid"]
        ]

        same_view = positives[
            positives["view"] == q["view"]
        ]

        opposite_view = positives[
            positives["view"] != q["view"]
        ]

        rows.append({
            "query_filename": q["filename"],
            "pid": q["pid"],
            "query_view": q["view"],
            "all_gallery_positives": len(positives),
            "same_view_positives": len(same_view),
            "opposite_view_positives": len(opposite_view),
        })

    qstats = pd.DataFrame(rows)

    print("\nQueries checked:", len(qstats))

    print(
        "Queries with ZERO gallery positives:",
        (qstats["all_gallery_positives"] == 0).sum()
    )

    print(
        "Queries with ZERO opposite-view positives:",
        (qstats["opposite_view_positives"] == 0).sum()
    )

    print("\nOpposite-view positives per query:")
    print(
        qstats["opposite_view_positives"]
        .describe()
        .round(2)
    )

    print("\nDistribution:")
    print(
        qstats["opposite_view_positives"]
        .value_counts()
        .sort_index()
    )

    output = (
        REPORT_DIR /
        f"{variant}_query_gallery_protocol.csv"
    )

    qstats.to_csv(output, index=False)

    print("\nSaved:", output)

    # ------------------------------------------------
    # 3. Show weakest queries
    # ------------------------------------------------

    print("\nQueries with the fewest opposite-view positives:")

    print(
        qstats.sort_values(
            "opposite_view_positives"
        ).head(10).to_string(index=False)
    )
