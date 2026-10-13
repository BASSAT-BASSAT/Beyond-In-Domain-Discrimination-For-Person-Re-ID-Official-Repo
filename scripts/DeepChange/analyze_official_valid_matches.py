from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

df = pd.read_csv(
    MANIFEST,
    parse_dates=["timestamp"]
)


def analyze(name, query_split, gallery_split):

    qdf = df[df["split"] == query_split]
    gdf = df[df["split"] == gallery_split]

    gallery_by_pid = {
        pid: group
        for pid, group in gdf.groupby("pid")
    }

    rows = []

    for _, q in qdf.iterrows():

        g = gallery_by_pid[q["pid"]]

        # DeepChange valid matches:
        # same PID, DIFFERENT tracklet
        valid = g[
            g["tracklet_id"] != q["tracklet_id"]
        ].copy()

        if len(valid):

            gaps = (
                valid["timestamp"] - q["timestamp"]
            ).abs().dt.total_seconds() / 86400

            same_camera = (
                valid["camera_id"]
                == q["camera_id"]
            )

            diff_date = (
                valid["date"]
                != q["date"]
            )

            rows.append({
                "filename": q["filename"],
                "pid": q["pid"],

                "valid_positives": len(valid),

                "nearest_valid_gap_days":
                    float(gaps.min()),

                "cross_camera_valid":
                    int(
                        (
                            valid["camera_id"]
                            != q["camera_id"]
                        ).sum()
                    ),

                "diff_date_valid":
                    int(diff_date.sum()),

                "long_30d_valid":
                    int((gaps >= 30).sum()),

                "long_90d_valid":
                    int((gaps >= 90).sum()),

                "long_180d_valid":
                    int((gaps >= 180).sum()),

                "same_camera_30d_valid":
                    int(
                        (
                            same_camera
                            & (gaps >= 30)
                        ).sum()
                    ),
            })

        else:

            rows.append({
                "filename": q["filename"],
                "pid": q["pid"],
                "valid_positives": 0,
                "nearest_valid_gap_days": np.nan,
                "cross_camera_valid": 0,
                "diff_date_valid": 0,
                "long_30d_valid": 0,
                "long_90d_valid": 0,
                "long_180d_valid": 0,
                "same_camera_30d_valid": 0,
            })


    r = pd.DataFrame(rows)

    valid_queries = r[
        r["valid_positives"] > 0
    ]

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("Total queries       :", len(r))
    print(
        "Queries with valid match:",
        len(valid_queries)
    )
    print(
        "Queries with NO valid match:",
        int((r["valid_positives"] == 0).sum())
    )

    print("\nValid positives/query:")

    print(
        valid_queries["valid_positives"]
        .describe(
            percentiles=[.25, .5, .75, .95]
        )
    )

    print("\nNearest VALID positive gap (days):")

    print(
        valid_queries[
            "nearest_valid_gap_days"
        ]
        .describe(
            percentiles=[.25, .5, .75, .95]
        )
    )


    def availability(column, label):

        mask = valid_queries[column] > 0

        print(
            f"{label:<35}"
            f"{int(mask.sum()):6d} / "
            f"{len(valid_queries)} "
            f"({mask.mean()*100:6.2f}%)"
        )


    print("\nAmong queries with valid matches:")

    availability(
        "cross_camera_valid",
        "Has cross-camera valid:"
    )

    availability(
        "diff_date_valid",
        "Has different-date valid:"
    )

    availability(
        "long_30d_valid",
        "Has >=30-day valid:"
    )

    availability(
        "long_90d_valid",
        "Has >=90-day valid:"
    )

    availability(
        "long_180d_valid",
        "Has >=180-day valid:"
    )

    availability(
        "same_camera_30d_valid",
        "Has same-camera >=30d:"
    )


analyze(
    "VAL",
    "val_query",
    "val_gallery"
)

analyze(
    "TEST",
    "test_query",
    "test_gallery"
)
