from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/DeepChange"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    MANIFEST,
    parse_dates=["timestamp"]
)


def analyze(split_name, query_split, gallery_split):

    qdf = df[df["split"] == query_split].copy()
    gdf = df[df["split"] == gallery_split].copy()

    print("\n" + "=" * 80)
    print(split_name)
    print("=" * 80)

    print("Query images  :", len(qdf))
    print("Gallery images:", len(gdf))

    # Exact filename overlap
    overlap = (
        set(qdf["filename"])
        & set(gdf["filename"])
    )

    print("Exact filename overlap:", len(overlap))

    # Cache gallery by PID
    gallery_by_pid = {
        pid: group.copy()
        for pid, group in gdf.groupby("pid")
    }

    rows = []

    for _, q in qdf.iterrows():

        g = gallery_by_pid[q["pid"]]

        gap_days = (
            g["timestamp"] - q["timestamp"]
        ).abs().dt.total_seconds() / 86400

        same_tracklet = (
            g["tracklet_id"] == q["tracklet_id"]
        )

        cross_tracklet = ~same_tracklet

        same_camera = (
            g["camera_id"] == q["camera_id"]
        )

        cross_camera = ~same_camera

        same_date = (
            g["date"] == q["date"]
        )

        diff_date = ~same_date

        rows.append({
            "filename": q["filename"],
            "pid": q["pid"],
            "camera_id": q["camera_id"],
            "tracklet_id": q["tracklet_id"],

            "total_positives": len(g),

            "same_tracklet_pos":
                int(same_tracklet.sum()),

            "cross_tracklet_pos":
                int(cross_tracklet.sum()),

            "same_camera_pos":
                int(same_camera.sum()),

            "cross_camera_pos":
                int(cross_camera.sum()),

            "same_date_pos":
                int(same_date.sum()),

            "diff_date_pos":
                int(diff_date.sum()),

            "same_camera_diff_date_pos":
                int((same_camera & diff_date).sum()),

            "diff_camera_diff_date_pos":
                int((cross_camera & diff_date).sum()),

            "long_30d_pos":
                int((gap_days >= 30).sum()),

            "long_90d_pos":
                int((gap_days >= 90).sum()),

            "long_180d_pos":
                int((gap_days >= 180).sum()),

            "same_camera_30d_pos":
                int((same_camera & (gap_days >= 30)).sum()),

            "nearest_positive_gap_days":
                float(gap_days.min()),

            "nearest_cross_tracklet_gap_days":
                float(
                    gap_days[cross_tracklet].min()
                )
                if cross_tracklet.any()
                else np.nan,
        })

    result = pd.DataFrame(rows)


    def availability(column, label):

        mask = result[column] > 0

        print(
            f"{label:<38}",
            f"{int(mask.sum()):6d} / {len(result)}",
            f"({mask.mean()*100:6.2f}%)"
        )


    print("\nPositive availability per QUERY:")

    availability(
        "same_tracklet_pos",
        "Has same-tracklet positive:"
    )

    availability(
        "cross_tracklet_pos",
        "Has cross-tracklet positive:"
    )

    availability(
        "cross_camera_pos",
        "Has cross-camera positive:"
    )

    availability(
        "diff_date_pos",
        "Has different-date positive:"
    )

    availability(
        "same_camera_diff_date_pos",
        "Has same-camera/diff-date positive:"
    )

    availability(
        "diff_camera_diff_date_pos",
        "Has diff-camera/diff-date positive:"
    )

    availability(
        "long_30d_pos",
        "Has >=30-day positive:"
    )

    availability(
        "long_90d_pos",
        "Has >=90-day positive:"
    )

    availability(
        "long_180d_pos",
        "Has >=180-day positive:"
    )

    availability(
        "same_camera_30d_pos",
        "Has same-camera >=30-day positive:"
    )


    print("\nSame-tracklet positive counts/query:")
    print(
        result["same_tracklet_pos"]
        .describe(
            percentiles=[.25, .5, .75, .95]
        )
    )


    print("\nCross-tracklet positive counts/query:")
    print(
        result["cross_tracklet_pos"]
        .describe(
            percentiles=[.25, .5, .75, .95]
        )
    )


    print("\nNearest positive SESSION gap (days):")
    print(
        result["nearest_positive_gap_days"]
        .describe(
            percentiles=[.25, .5, .75, .95]
        )
    )


    valid_cross = (
        result[
            "nearest_cross_tracklet_gap_days"
        ]
        .dropna()
    )

    print("\nNearest CROSS-TRACKLET session gap (days):")

    if len(valid_cross):
        print(
            valid_cross.describe(
                percentiles=[.25, .5, .75, .95]
            )
        )


    # Camera 2 subset
    q_cam2 = result[
        result["camera_id"] == 2
    ]

    g_cam2 = gdf[
        gdf["camera_id"] == 2
    ]

    print("\nCamera 2 subset:")
    print(
        "Query images :",
        len(q_cam2)
    )
    print(
        "Query PIDs   :",
        q_cam2["pid"].nunique()
    )
    print(
        "Gallery imgs :",
        len(g_cam2)
    )
    print(
        "Gallery PIDs :",
        g_cam2["pid"].nunique()
    )

    if len(q_cam2):

        mask = (
            q_cam2["cross_camera_pos"] > 0
        )

        print(
            "Cam2 queries with cross-camera positives:",
            int(mask.sum()),
            "/",
            len(q_cam2),
            f"({mask.mean()*100:.2f}%)"
        )


    out = (
        REPORT_DIR
        / f"{split_name.lower()}_query_structure.csv"
    )

    result.to_csv(
        out,
        index=False
    )

    print("\nSaved:", out)


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
