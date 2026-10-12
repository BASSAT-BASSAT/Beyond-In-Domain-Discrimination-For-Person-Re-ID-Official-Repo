from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/LaST/last_manifest.csv"
)

df = pd.read_csv(MANIFEST, low_memory=False)


def audit(split_q, split_g, label):

    q = df[df["split"] == split_q].copy()

    g = df[
        (df["split"] == split_g)
        & (~df["is_distractor"])
    ].copy()

    gallery_by_pid = {
        pid: grp
        for pid, grp in g.groupby("pid")
    }

    rows = []

    same_frame_pairs = []

    for qr in q.itertuples():

        positives = gallery_by_pid.get(qr.pid)

        if positives is None:
            continue

        same_video = positives[
            positives["video_id"] == qr.video_id
        ].copy()

        diff_video = positives[
            positives["video_id"] != qr.video_id
        ]

        min_gap = np.nan

        if len(same_video):

            same_video["frame_gap"] = (
                same_video["frame_id"] - qr.frame_id
            ).abs()

            min_gap = same_video["frame_gap"].min()

            exact_frame = same_video[
                same_video["frame_gap"] == 0
            ]

            for gr in exact_frame.itertuples():

                same_frame_pairs.append({
                    "query": qr.filename,
                    "gallery": gr.filename,
                    "pid": qr.pid,
                    "video_id": qr.video_id,
                    "frame_id": qr.frame_id,
                    "query_bbox": qr.bbox_id,
                    "gallery_bbox": gr.bbox_id,
                })

        rows.append({
            "pid": qr.pid,
            "query": qr.filename,
            "video_id": qr.video_id,
            "frame_id": qr.frame_id,
            "same_video_positives": len(same_video),
            "different_video_positives": len(diff_video),
            "min_same_video_frame_gap": min_gap,
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print("\nQueries:", len(out))

    valid = out["min_same_video_frame_gap"].dropna()

    print("\nNearest same-video positive frame-gap:")
    print(valid.describe().round(2))

    print("\nPercentiles:")
    print(
        valid.quantile(
            [0.01, 0.05, 0.25, 0.50,
             0.75, 0.95, 0.99]
        ).round(2)
    )

    for t in [
        0, 1, 2, 3, 5, 10,
        20, 30, 60, 100
    ]:

        pct = (
            (valid <= t).mean()
            * 100
        )

        print(
            f"Nearest positive gap <= {t:3d}: "
            f"{pct:6.2f}%"
        )

    print("\nSame-frame query/gallery positive pairs:")
    print(len(same_frame_pairs))

    for x in same_frame_pairs:
        print(x)

    output_dir = (
        Path.home()
        / "DG_ReID_Audit/reports/LaST"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out.to_csv(
        output_dir
        / f"{label.lower()}_query_temporal.csv",
        index=False
    )


audit(
    "val_query",
    "val_gallery",
    "validation"
)

audit(
    "test_query",
    "test_gallery",
    "test"
)
