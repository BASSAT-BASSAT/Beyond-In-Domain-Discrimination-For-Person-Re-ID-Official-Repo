"""Evaluate an extractor on manifest test sets (image and video) with the benchmark protocol.

Image sets: one feature per image. Video sets: up to ``num_frames`` frames per tracklet
(``linspace`` rule), mean-pooled and L2-normalized (benchmark ``TRACKLET_CONTRACT``).
Then cosine distance, same-pid-same-camera junk removal, CMC / mAP / mINP.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from reidfm.data.manifest import Manifest, ManifestRow, load_manifest
from reidfm.evaluation.aggregate import mean_pool_l2
from reidfm.evaluation.metrics import evaluate_rank_from_features
from reidfm.evaluation.sampling import sample_indices

logger = logging.getLogger(__name__)
HEADLINE = ("Rank-1", "Rank-5", "Rank-10", "mAP", "mINP")


@dataclass
class EvalItem:
    pid: str
    camid: str
    paths: list[Path]


def _frame_key(row: ManifestRow):
    return (int(row.frame) if row.frame.lstrip("-").isdigit() else -1, row.path)


def build_items(manifest: Manifest, split: str, num_frames: int = 8) -> list[EvalItem]:
    rows = manifest.split(split)
    if manifest.kind != "video" or not any(row.tracklet_id for row in rows):
        return [EvalItem(row.pid, row.camid, [manifest.resolve(row)]) for row in rows]
    groups: "OrderedDict[tuple[str, str], list[ManifestRow]]" = OrderedDict()
    for row in rows:
        groups.setdefault((row.pid, row.tracklet_id or row.path), []).append(row)
    items = []
    for members in groups.values():
        members.sort(key=_frame_key)
        chosen = [members[i] for i in sample_indices(len(members), num_frames)]
        items.append(EvalItem(members[0].pid, members[0].camid, [manifest.resolve(r) for r in chosen]))
    return items


def subsample_queries(items: list[EvalItem], max_queries: int) -> list[EvalItem]:
    if not max_queries or len(items) <= max_queries:
        return items
    step = len(items) / float(max_queries)
    return [items[int(i * step)] for i in range(max_queries)]


def embed_items(items: Sequence[EvalItem], extractor, desc: str | None = None) -> np.ndarray:
    flat = [path for item in items for path in item.paths]
    feats = extractor.extract(flat, desc=desc)
    out, start = [], 0
    for item in items:
        out.append(mean_pool_l2(feats[start : start + len(item.paths)]))
        start += len(item.paths)
    return np.stack(out).astype(np.float32)


def _to_ints(values: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.array([mapping[v] for v in values], dtype=np.int64), mapping


def evaluate_manifest(manifest: Manifest, extractor, num_frames: int = 8, max_queries: int = 0) -> dict:
    started = time.time()
    query = subsample_queries(build_items(manifest, "query", num_frames), max_queries)
    gallery = build_items(manifest, "gallery", num_frames)
    if not query or not gallery:
        raise ValueError(f"{manifest.name}: no query/gallery rows")
    q_feat = embed_items(query, extractor, desc=f"{manifest.name} query")
    g_feat = embed_items(gallery, extractor, desc=f"{manifest.name} gallery")
    pids, _ = _to_ints([i.pid for i in query] + [i.pid for i in gallery])
    cams, _ = _to_ints([i.camid for i in query] + [i.camid for i in gallery])
    nq = len(query)
    metrics = evaluate_rank_from_features(q_feat, g_feat, pids[:nq], pids[nq:], cams[:nq], cams[nq:])
    metrics.update(
        kind=manifest.kind,
        num_query=len(query),
        num_gallery=len(gallery),
        seconds=round(time.time() - started, 1),
    )
    return metrics


def mean_row(results: dict[str, dict]) -> dict[str, float]:
    keys = [k for k in results if k != "mean"]
    return {m: float(np.mean([results[k][m] for k in keys])) for m in HEADLINE} if keys else {}


def evaluate_sets(
    keys: Sequence[str],
    extractor,
    num_frames: int = 8,
    max_queries: int = 0,
    roots=None,
    skip_missing: bool = False,
) -> "OrderedDict[str, dict]":
    results: "OrderedDict[str, dict]" = OrderedDict()
    for key in keys:
        try:
            manifest = load_manifest(key, roots)
        except FileNotFoundError:
            if skip_missing:
                logger.warning("skipping %s: manifest not found", key)
                continue
            raise
        results[key] = evaluate_manifest(manifest, extractor, num_frames, max_queries)
        logger.info("%s: R1 %.2f mAP %.2f (%d q)", key, results[key]["Rank-1"], results[key]["mAP"], results[key]["num_query"])
    if results:
        results["mean"] = mean_row(results)
    return results


def results_markdown(results: dict[str, dict], title: str = "") -> str:
    lines = [f"### {title}", ""] if title else []
    lines.append("| set | kind | R1 | R5 | R10 | mAP | mINP | #q | #g |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, m in results.items():
        if key == "mean":
            continue
        lines.append(
            f"| {key} | {m['kind']} | {m['Rank-1']:.2f} | {m['Rank-5']:.2f} | {m['Rank-10']:.2f} | "
            f"{m['mAP']:.2f} | {m['mINP']:.2f} | {m['num_query']} | {m['num_gallery']} |"
        )
    if results.get("mean"):
        m = results["mean"]
        lines.append(f"| **mean** | | {m['Rank-1']:.2f} | {m['Rank-5']:.2f} | {m['Rank-10']:.2f} | {m['mAP']:.2f} | {m['mINP']:.2f} | | |")
    return "\n".join(lines) + "\n"
