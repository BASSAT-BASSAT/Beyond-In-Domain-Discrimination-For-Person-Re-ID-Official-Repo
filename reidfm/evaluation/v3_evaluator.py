"""V3 PACK-based, protocol-aware evaluation; no legacy Pool conversion.

Evaluate distinct (dataset, validation_protocol_id) query/gallery groups.
Do not deduplicate sample_uid globally: reciprocal protocols may reuse
the same physical image with a different retrieval_role.

The unweighted "mean" is a diagnostic result contract, NOT an approved
scientific model-selection policy.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping

import numpy as np

from reidfm.evaluation.v3_metrics import (
    evaluate_v3_rank_from_features,
    protocol_rule,
)
from reidfm.evaluation.v3_packed_extractor import extract_v3_packed_features


HEADLINE = ("Rank-1", "Rank-5", "Rank-10", "Rank-20", "mAP", "mINP")
REQUIRED = frozenset({
    "dataset", "dataset_root_id", "relative_path", "sample_uid",
    "identity_key", "validation_protocol_id", "retrieval_role",
    "raw_camera_or_view", "camera_context_id", "tracklet_id",
})


def group_v3_validation_records(records: Iterable[Mapping]) -> OrderedDict:
    """Preserve input ordering and role context, even for reused image UIDs.

    Output is ordered by first-seen (dataset, protocol), then maintains
    original manifest order separately within each query/gallery role.
    """
    grouped = OrderedDict()
    seen = set()

    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Record {position} must be a mapping")
        missing = REQUIRED - record.keys()
        if missing:
            raise ValueError(f"Record {position} missing fields: {sorted(missing)}")
        row = dict(record)
        for name in ("dataset", "dataset_root_id", "relative_path",
                     "sample_uid", "identity_key", "validation_protocol_id"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"Record {position} has empty {name}")

        role = row["retrieval_role"]
        if role not in ("query", "gallery"):
            raise ValueError(f"Record {position} invalid retrieval role: {role!r}")
        proto = row["validation_protocol_id"]
        protocol_rule(proto)  # fail fast on unknown or mistyped protocols

        ds = row["dataset"]
        unique_role_key = (ds, proto, role, row["sample_uid"])
        if unique_role_key in seen:
            raise ValueError(f"Duplicate protocol-role sample: {unique_role_key}")
        seen.add(unique_role_key)

        group = grouped.setdefault((ds, proto), {"query": [], "gallery": []})
        group[role].append(row)

    if not grouped:
        raise ValueError("No V3 validation records supplied")
    for (ds, proto), group in grouped.items():
        if not group["query"] or not group["gallery"]:
            raise ValueError(f"Missing query/gallery for {ds} | {proto}")
    return grouped


def evaluate_v3_packed_groups(
    model,
    records: Iterable[Mapping],
    pack_root,
    *,
    size_hw=(384, 128),
    batch_size=4,
    num_workers=0,
    device="cpu",
    max_rank=50,
    query_chunk=128,
    reidraw_files=None,
) -> OrderedDict:
    """Extract and score all supplied protocol groups with stable role metadata.

    Returns Trainer-shaped per-protocol results plus a diagnostic unweighted
    protocol mean. Does not call ReidfmTrainer.test() or checkpoint hooks.
    """
    grouped = group_v3_validation_records(records)
    results = OrderedDict()
    for (dataset, proto), group in grouped.items():
        queries, galleries = group["query"], group["gallery"]
        qf, qr = extract_v3_packed_features(
            model, queries, pack_root, size_hw=size_hw,
            batch_size=batch_size, num_workers=num_workers, device=device,
            reidraw_files=reidraw_files,
        )
        gf, gr = extract_v3_packed_features(
            model, galleries, pack_root, size_hw=size_hw,
            batch_size=batch_size, num_workers=num_workers, device=device,
            reidraw_files=reidraw_files,
        )
        if qr != queries or gr != galleries:
            raise RuntimeError(f"Metadata order changed for {dataset} | {proto}")
        metrics = evaluate_v3_rank_from_features(
            qf, gf, qr, gr, proto, max_rank=max_rank, query_chunk=query_chunk,
        )
        key = f"{dataset} | {proto}"
        if key in results:
            raise ValueError(f"Ambiguous V3 protocol result name: {key}")
        result = {name: float(metrics[name]) for name in HEADLINE}
        if "cmc" in metrics:
            result["cmc"] = [float(x) for x in metrics["cmc"]]
        result.update({
            "metric": (result["Rank-1"] + result["mAP"]) / 2.0,
            "num_query": int(metrics["num_query"]),
            "num_gallery": int(metrics["num_gallery"]),
            "positive_rule": metrics["positive_rule"],
        })
        results[key] = result

    names = list(results)
    results["mean"] = {
        metric: float(np.mean([results[name][metric] for name in names]))
        for metric in (*HEADLINE, "metric")
    }
    return results
