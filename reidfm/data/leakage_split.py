"""Cluster-aware train/val splits for multi-source ReID pools."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from reidfm.data.audit import dhash
from reidfm.data.manifest import Manifest, ManifestRow, load_manifest, write_manifest
from reidfm.data.pool import PoolConfig
from reidfm.data.registry import get_dataset_spec
from reidfm.data.split_policies import (
    get_split_policy,
    identity_disjoint_train_val,
    policy_summary,
    train_pids_from_manifest,
)
from reidfm.data.teammate_splits import load_teammate_split, validate_teammate_split


@dataclass(frozen=True)
class IdentityRef:
    dataset: str
    pid: str

    def key(self) -> str:
        return f"{self.dataset}:{self.pid}"


@dataclass
class SplitArtifacts:
    train_ids: dict[str, list[str]]
    val_ids: dict[str, list[str]]
    clusters: list[list[str]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def load_train_manifests(pool: PoolConfig, roots: Sequence[Path] | None = None) -> dict[str, Manifest]:
    manifests = {}
    for key in pool.sources:
        try:
            manifests[key] = load_manifest(key, roots)
        except FileNotFoundError:
            continue
    return manifests


def sample_train_rows(manifest: Manifest, max_per_id: int, rng: random.Random) -> list[ManifestRow]:
    by_pid: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in manifest.train:
        by_pid[row.pid].append(row)
    out: list[ManifestRow] = []
    for pid, rows in by_pid.items():
        if len(rows) <= max_per_id:
            out.extend(rows)
        else:
            out.extend(rng.sample(rows, max_per_id))
    return out


def _union_find_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def build_dhash_edges(paths: list[Path], hamming_threshold: int = 6) -> list[tuple[int, int]]:
    hashes = [dhash(p) for p in paths]
    edges: list[tuple[int, int]] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if (hashes[i] ^ hashes[j]).bit_count() <= hamming_threshold:
                edges.append((i, j))
    return edges


def assign_components(
    refs: list[IdentityRef],
    component_indices: list[list[int]],
    train_fraction: float,
    rng: random.Random,
) -> SplitArtifacts:
    by_domain: dict[str, list[int]] = defaultdict(list)
    for comp_id, members in enumerate(component_indices):
        domains = {refs[i].dataset for i in members}
        # component must not span domains for hard constraint — split per domain inside comp
        for domain in domains:
            domain_members = [i for i in members if refs[i].dataset == domain]
            by_domain[domain].append(domain_members)

    train_ids: dict[str, list[str]] = defaultdict(list)
    val_ids: dict[str, list[str]] = defaultdict(list)
    clusters: list[list[str]] = []

    for domain, comps in by_domain.items():
        rng.shuffle(comps)
        n_val = max(1, int(round((1.0 - train_fraction) * len(comps)))) if len(comps) > 1 else 0
        val_comps = set(range(len(comps) - n_val, len(comps)))
        for idx, members in enumerate(comps):
            cluster_keys = [refs[i].key() for i in members]
            clusters.append(cluster_keys)
            pids = sorted({refs[i].pid for i in members})
            if idx in val_comps:
                val_ids[domain].extend(pids)
            else:
                train_ids[domain].extend(pids)

    # dedupe pids per domain
    train_ids = {k: sorted(set(v)) for k, v in train_ids.items()}
    val_ids = {k: sorted(set(v)) for k, v in val_ids.items()}
    return SplitArtifacts(train_ids=train_ids, val_ids=val_ids, clusters=clusters)


def _split_dhash_domain(
    key: str,
    manifest: Manifest,
    *,
    train_fraction: float,
    max_samples_per_id: int,
    dhash_threshold: int,
    rng: random.Random,
) -> tuple[list[str], list[str], list[list[str]], dict]:
    sampled = sample_train_rows(manifest, max_samples_per_id, rng)
    by_pid: dict[str, ManifestRow] = {}
    for row in sampled:
        by_pid.setdefault(row.pid, row)
    refs: list[IdentityRef] = []
    paths: list[Path] = []
    for pid, row in sorted(by_pid.items()):
        refs.append(IdentityRef(key, pid))
        paths.append(manifest.resolve(row))
    edges = build_dhash_edges(paths, dhash_threshold)
    components = _union_find_components(len(refs), edges)
    partial = assign_components(refs, components, train_fraction, rng)
    return (
        partial.train_ids.get(key, []),
        partial.val_ids.get(key, []),
        partial.clusters,
        {"embedder": "dhash", "identities": len(refs), "components": len(components)},
    )


def _split_teammate_or_dhash_domain(
    key: str,
    manifest: Manifest,
    *,
    train_fraction: float,
    max_samples_per_id: int,
    dhash_threshold: int,
    seed: int,
    rng: random.Random,
) -> tuple[list[str], list[str], list[list[str]], dict]:
    blob = load_teammate_split(key)
    if blob is not None:
        train, val = validate_teammate_split(key, manifest, blob)
        clusters = [[f"{key}:{p}"] for p in train + val]
        return train, val, clusters, {
            "strategy": "teammate_file",
            "policy_label": blob.get("label", f"{key}_teammate"),
            "source": blob.get("source", "teammate"),
            "path": str(blob.get("_path", "")),
            "n_train_ids": len(train),
            "n_val_ids": len(val),
        }
    tr, va, cl, meta = _split_dhash_domain(
        key,
        manifest,
        train_fraction=train_fraction,
        max_samples_per_id=max_samples_per_id,
        dhash_threshold=dhash_threshold,
        rng=rng,
    )
    meta["fallback"] = "dhash_cluster_pending_teammate_file"
    meta["expected_file"] = f"splits/datasets/{key}.json"
    return tr, va, cl, meta


def _split_identity_domain(
    key: str,
    manifest: Manifest,
    *,
    train_fraction: float,
    seed: int,
    policy_kind: str,
) -> tuple[list[str], list[str], list[list[str]], dict]:
    pids = train_pids_from_manifest(manifest)
    if not pids:
        return [], [], [], {"error": "no train identities"}
    train, val, meta = identity_disjoint_train_val(pids, train_fraction=train_fraction, seed=seed + hash(key) % 997)
    clusters = [[f"{key}:{p}"] for p in train + val]
    meta["policy_kind"] = policy_kind
    return train, val, clusters, meta


def split_pool(
    pool: PoolConfig,
    *,
    seed: int = 42,
    train_fraction: float = 0.85,
    max_samples_per_id: int = 4,
    dhash_threshold: int = 6,
    roots: Sequence[Path] | None = None,
) -> SplitArtifacts:
    """Per-source train/val identity lists (policy-aware; never mixes query/gallery test rows)."""
    rng = random.Random(seed)
    manifests = load_train_manifests(pool, roots)
    train_ids: dict[str, list[str]] = defaultdict(list)
    val_ids: dict[str, list[str]] = defaultdict(list)
    clusters: list[list[str]] = []
    per_dataset: dict[str, dict] = {}
    total_ids = 0

    for key, manifest in manifests.items():
        spec = get_dataset_spec(key)
        policy = get_split_policy(spec.split_policy or key)
        if policy.kind in ("identity_disjoint", "orbench_official_train"):
            tr, va, cl, meta = _split_identity_domain(
                key, manifest, train_fraction=train_fraction, seed=seed, policy_kind=policy.kind
            )
            meta["policy_label"] = policy.label
        elif policy.kind == "teammate_or_dhash":
            tr, va, cl, meta = _split_teammate_or_dhash_domain(
                key,
                manifest,
                train_fraction=train_fraction,
                max_samples_per_id=max_samples_per_id,
                dhash_threshold=dhash_threshold,
                seed=seed,
                rng=rng,
            )
            meta.setdefault("policy_label", policy.label)
        else:
            tr, va, cl, meta = _split_dhash_domain(
                key,
                manifest,
                train_fraction=train_fraction,
                max_samples_per_id=max_samples_per_id,
                dhash_threshold=dhash_threshold,
                rng=rng,
            )
            meta["policy_label"] = policy.label
        train_ids[key] = sorted(set(tr))
        val_ids[key] = sorted(set(va))
        clusters.extend(cl)
        per_dataset[key] = meta
        total_ids += len(train_ids[key]) + len(val_ids[key])

    return SplitArtifacts(
        train_ids=dict(train_ids),
        val_ids=dict(val_ids),
        clusters=clusters,
        meta={
            "pool": pool.name,
            "seed": seed,
            "train_fraction": train_fraction,
            "identities": total_ids,
            "per_dataset": per_dataset,
            "policies": policy_summary(pool.sources),
        },
    )


def save_split_artifacts(artifacts: SplitArtifacts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pool_v2_train_ids.json").write_text(json.dumps(artifacts.train_ids, indent=2), encoding="utf-8")
    (out_dir / "pool_v2_val_ids.json").write_text(json.dumps(artifacts.val_ids, indent=2), encoding="utf-8")
    (out_dir / "leakage_clusters.json").write_text(json.dumps(artifacts.clusters, indent=2), encoding="utf-8")
    (out_dir / "split_meta.json").write_text(json.dumps(artifacts.meta, indent=2), encoding="utf-8")
    (out_dir / "split_policies.json").write_text(
        json.dumps(artifacts.meta.get("policies", {}), indent=2), encoding="utf-8"
    )


def build_val_manifest(
    pool: PoolConfig,
    val_ids: dict[str, list[str]],
    out_dir: Path,
    roots: Sequence[Path] | None = None,
    queries_per_id: int = 1,
    seed: int = 0,
) -> Path:
    """Build a small query/gallery manifest from val identities for in-training EvalHook."""
    rng = random.Random(seed)
    rows: list[ManifestRow] = []
    for key, pids in val_ids.items():
        if not pids:
            continue
        manifest = load_manifest(key, roots)
        by_pid: dict[str, list[ManifestRow]] = defaultdict(list)
        for row in manifest.train:
            if row.pid in pids:
                by_pid[row.pid].append(row)
        for pid, train_rows in by_pid.items():
            if not train_rows:
                continue
            rng.shuffle(train_rows)
            q = train_rows[:queries_per_id]
            g = train_rows[queries_per_id:] or train_rows
            for row in q:
                rows.append(
                    ManifestRow(
                        str(manifest.resolve(row)),
                        row.pid,
                        row.camid,
                        "poolv2val",
                        "query",
                        row.tracklet_id,
                        row.frame,
                    )
                )
            for row in g:
                rows.append(
                    ManifestRow(
                        str(manifest.resolve(row)),
                        row.pid,
                        row.camid,
                        "poolv2val",
                        "gallery",
                        row.tracklet_id,
                        row.frame,
                    )
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "manifest.csv", rows)
    return out_dir / "manifest.csv"
