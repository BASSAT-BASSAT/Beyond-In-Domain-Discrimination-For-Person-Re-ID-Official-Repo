"""V3 protocol-aware cosine Rank-k, mAP, mINP.

The authoritative positive rules come from
scripts/dg_reid_data_pipeline/build_global_p3_p5.py.

Call separately for each (dataset, validation_protocol_id) group.
"""

from __future__ import annotations

import numpy as np

from reidfm.evaluation.metrics import l2_normalize


CROSS_CONTEXT_PROTOCOLS = frozenset({
    "iust_internal_cross_camera_identity_val_v1",
    "lpw_internal_cross_view_identity_val_v1",
    "orbench_rgb_internal_cross_context_val_v2",
    "prid2011_internal_cross_context_val_v2",
    "rpifield_internal_cross_context_val_v2",
})

CUHK_PROTOCOLS = frozenset({
    "cuhk03_internal_c1_to_c2_v1",
    "cuhk03_internal_c2_to_c1_v1",
})

SAMPLE_UID_PROTOCOLS = frozenset({
    "last_official_val_query_gallery_v1",
    "posetrack_internal_same_or_cross_video_v1",
})

DEEPCHANGE_PROTOCOL = (
    "deepchange_internal_cross_tracklet_v1"
)


def protocol_rule(protocol_id: str) -> str:

    if protocol_id in CUHK_PROTOCOLS:
        return "different_raw_camera"

    if protocol_id == DEEPCHANGE_PROTOCOL:
        return "different_tracklet"

    if protocol_id in SAMPLE_UID_PROTOCOLS:
        return "different_sample_uid"

    if protocol_id in CROSS_CONTEXT_PROTOCOLS:
        return "different_camera_context"

    raise ValueError(
        f"Unsupported V3 validation protocol: {protocol_id!r}"
    )


def evaluate_v3_rank_from_features(
    query_features,
    gallery_features,
    query_rows,
    gallery_rows,
    protocol_id: str,
    *,
    max_rank: int = 50,
    query_chunk: int = 128,
) -> dict:
    """Calculate protocol-aware retrieval metrics.

    Invalid same-identity gallery candidates are junk.
    They are excluded from ranking, not treated as negatives.
    """

    rule = protocol_rule(protocol_id)

    qf = np.asarray(
        query_features,
        dtype=np.float32,
    )

    gf = np.asarray(
        gallery_features,
        dtype=np.float32,
    )

    if (
        qf.ndim != 2
        or gf.ndim != 2
        or qf.shape[1] != gf.shape[1]
    ):
        raise ValueError(
            "Query/gallery features must be 2D "
            "with equal feature dimensions"
        )

    if (
        len(query_rows) != len(qf)
        or len(gallery_rows) != len(gf)
    ):
        raise ValueError(
            "Feature/metadata row count mismatch"
        )

    if not len(qf) or not len(gf):
        raise ValueError(
            "Query and gallery must be nonempty"
        )

    if (
        not np.isfinite(qf).all()
        or not np.isfinite(gf).all()
    ):
        raise ValueError(
            "Features must be finite"
        )

    if max_rank < 1 or query_chunk < 1:
        raise ValueError(
            "max_rank and query_chunk must be positive"
        )

    def field(rows, key):

        return np.asarray(
            [
                str(row.get(key, ""))
                for row in rows
            ],
            dtype=object,
        )

    gp = field(
        gallery_rows,
        "identity_key",
    )

    gu = field(
        gallery_rows,
        "sample_uid",
    )

    gc = field(
        gallery_rows,
        "raw_camera_or_view",
    )

    gx = field(
        gallery_rows,
        "camera_context_id",
    )

    gt = field(
        gallery_rows,
        "tracklet_id",
    )

    qp = field(
        query_rows,
        "identity_key",
    )

    qu = field(
        query_rows,
        "sample_uid",
    )

    qc = field(
        query_rows,
        "raw_camera_or_view",
    )

    qx = field(
        query_rows,
        "camera_context_id",
    )

    qt = field(
        query_rows,
        "tracklet_id",
    )

    query = l2_normalize(qf)

    gallery = l2_normalize(gf)

    cmcs = []
    aps = []
    inps = []

    removed_junk = 0

    k = min(
        int(max_rank),
        len(gp),
    )

    # --------------------------------------------------------
    # Chunked cosine distance computation
    # --------------------------------------------------------

    for start in range(
        0,
        len(query),
        int(query_chunk),
    ):

        end = min(
            start + int(query_chunk),
            len(query),
        )

        distances = (
            1.0
            - query[start:end] @ gallery.T
        )

        for offset, qi in enumerate(
            range(start, end)
        ):

            same = (
                gp == qp[qi]
            )

            # -----------------------------------------------
            # Protocol-aware positive eligibility
            # -----------------------------------------------

            if rule == "different_raw_camera":

                positive = (
                    same
                    & (gc != qc[qi])
                )

            elif rule == "different_tracklet":

                positive = (
                    same
                    & (qt[qi] != "")
                    & (gt != "")
                    & (gt != qt[qi])
                )

            elif rule == "different_sample_uid":

                positive = (
                    same
                    & (gu != qu[qi])
                )

            else:

                positive = (
                    same
                    & (qx[qi] != "")
                    & (gx != "")
                    & (gx != qx[qi])
                )

            # Invalid same-ID candidates are junk.
            # Identical samples are also excluded.

            junk = (
                (same & ~positive)
                | (gu == qu[qi])
            )

            ranked = np.argsort(
                distances[offset],
                kind="stable",
            )

            ranked = ranked[
                ~junk[ranked]
            ]

            removed_junk += int(
                junk.sum()
            )

            if not len(ranked):

                raise ValueError(
                    f"No non-junk gallery "
                    f"for query index {qi}"
                )

            matches = positive[
                ranked
            ]

            count = int(
                matches.sum()
            )

            if count == 0:

                raise ValueError(
                    f"No eligible positives "
                    f"for query index {qi} "
                    f"({protocol_id})"
                )

            # -----------------------------------------------
            # CMC
            # -----------------------------------------------

            running = np.cumsum(
                matches,
                dtype=np.int64,
            )

            cmc = np.minimum(
                running,
                1,
            ).astype(
                np.float64
            )[:k]

            if len(cmc) < k:

                cmc = np.pad(
                    cmc,
                    (
                        0,
                        k - len(cmc),
                    ),
                    constant_values=cmc[-1],
                )

            cmcs.append(
                cmc
            )

            # -----------------------------------------------
            # Average Precision
            # -----------------------------------------------

            precision = (
                running
                / np.arange(
                    1,
                    len(matches) + 1,
                    dtype=np.float64,
                )
            )

            ap = float(
                np.sum(
                    precision * matches
                )
                / count
            )

            aps.append(
                ap
            )

            # -----------------------------------------------
            # Inverse Negative Penalty
            # -----------------------------------------------

            last = int(
                np.flatnonzero(
                    matches
                )[-1]
            )

            inp = float(
                running[last]
                / (last + 1)
            )

            inps.append(
                inp
            )

    # --------------------------------------------------------
    # Aggregate query metrics
    # --------------------------------------------------------

    mean_cmc = np.mean(
        np.stack(cmcs),
        axis=0,
    )

    return {
        "Rank-1":
            float(mean_cmc[0] * 100.0),

        "Rank-5":
            float(mean_cmc[min(4, k - 1)] * 100.0),

        "Rank-10":
            float(mean_cmc[min(9, k - 1)] * 100.0),

        "Rank-20":
            float(mean_cmc[min(19, k - 1)] * 100.0),

        "mAP":
            float(np.mean(aps) * 100.0),

        "mINP":
            float(np.mean(inps) * 100.0),

        "num_valid_queries":
            float(len(aps)),

        "num_query":
            len(query_rows),

        "num_gallery":
            len(gallery_rows),

        "num_junk_removed":
            removed_junk,

        "protocol_id":
            protocol_id,

        "positive_rule":
            rule,
    }
