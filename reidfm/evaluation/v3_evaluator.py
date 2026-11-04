"""V3 PACK-based, protocol-aware evaluation; no legacy Pool conversion.

Evaluate distinct (dataset, validation_protocol_id) query/gallery groups.
Do not deduplicate sample_uid globally: reciprocal protocols may reuse
the same physical image with a different retrieval_role.

The unweighted "mean" is a diagnostic result contract, NOT an approved
scientific model-selection policy.
"""

from __future__ import annotations

import posixpath
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping

import numpy as np

from reidfm.evaluation.sampling import sample_indices
from reidfm.evaluation.v3_metrics import (
    evaluate_v3_rank_from_features,
    protocol_rule,
)
from reidfm.evaluation.v3_packed_extractor import extract_v3_packed_features

# MARS frame names: <pid4>C<cam>T<track4>F<frame3>.jpg, e.g. 0001C1T0001F001.jpg
_MARS_TRACK = re.compile(r"^(-?\d+)C(\d+)T(\d+)F\d+", re.IGNORECASE)


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


def _tracklet_key(row: Mapping) -> str:
    tracklet = str(row.get("tracklet_id") or "").strip()
    if not tracklet:
        basename = posixpath.basename(str(row["relative_path"]).replace("\\", "/"))
        match = _MARS_TRACK.match(basename)
        if match is None:
            raise ValueError(f"Cannot derive tracklet for video row {row['relative_path']!r}")
        tracklet = f"T{match.group(3)}"
    return f"{row['identity_key']}|{row.get('raw_camera_or_view', '')}|{tracklet}"


def sample_tracklet_records(rows: list[dict], num_frames: int):
    """Group frame rows into tracklets and pick up to ``num_frames`` frames each.

    Returns ``(sampled_rows, owner_index, tracklet_rows)``: the frames to embed, the
    tracklet index each sampled frame belongs to, and one metadata row per tracklet.
    """
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        groups.setdefault(_tracklet_key(row), []).append(row)
    sampled, owners, tracklets = [], [], []
    for index, (key, members) in enumerate(groups.items()):
        members = sorted(members, key=lambda r: r["relative_path"])
        for i in sample_indices(len(members), num_frames):
            sampled.append(members[i])
            owners.append(index)
        rep = dict(members[0])
        rep["tracklet_id"] = key
        rep["sample_uid"] = f"{rep['dataset']}::{rep['retrieval_role']}::{key}"
        tracklets.append(rep)
    return sampled, np.asarray(owners, dtype=np.int64), tracklets


def mean_pool_tracklets(features: np.ndarray, owners: np.ndarray, num_tracklets: int) -> np.ndarray:
    pooled = np.zeros((num_tracklets, features.shape[1]), dtype=np.float64)
    np.add.at(pooled, owners, features)
    pooled /= np.bincount(owners, minlength=num_tracklets)[:, None]
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)


def _extract_role(model, rows, pack_root, tracklet_frames, **kwargs):
    if tracklet_frames > 0:
        sampled, owners, tracklets = sample_tracklet_records(rows, tracklet_frames)
        feats, got = extract_v3_packed_features(model, sampled, pack_root, **kwargs)
        if got != sampled:
            raise RuntimeError("Metadata order changed during tracklet extraction")
        return mean_pool_tracklets(feats, owners, len(tracklets)), tracklets
    feats, got = extract_v3_packed_features(model, rows, pack_root, **kwargs)
    if got != rows:
        raise RuntimeError("Metadata order changed during extraction")
    return feats, rows


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
    tracklet_frames: int = 0,
) -> OrderedDict:
    """Extract and score all supplied protocol groups with stable role metadata.

    ``tracklet_frames > 0`` enables the video protocol (MARS): frames are grouped into
    tracklets, up to that many frames are sampled per tracklet, and their features are
    mean-pooled into one L2-normalised descriptor per tracklet before ranking.

    Returns Trainer-shaped per-protocol results plus a diagnostic unweighted
    protocol mean. Does not call ReidfmTrainer.test() or checkpoint hooks.
    """
    grouped = group_v3_validation_records(records)
    results = OrderedDict()
    extract_kwargs = dict(
        size_hw=size_hw, batch_size=batch_size, num_workers=num_workers,
        device=device, reidraw_files=reidraw_files,
    )
    for (dataset, proto), group in grouped.items():
        qf, qr = _extract_role(model, group["query"], pack_root, tracklet_frames, **extract_kwargs)
        gf, gr = _extract_role(model, group["gallery"], pack_root, tracklet_frames, **extract_kwargs)
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
