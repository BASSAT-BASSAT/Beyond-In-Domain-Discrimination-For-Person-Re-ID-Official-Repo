#!/usr/bin/env python3
"""Build leakage-aware train/val identity splits for a pool yaml.

Example:
    python scripts/build_train_val_split.py --pool configs/datasets/pool_v2.yaml --out splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.data.leakage_split import build_val_manifest, save_split_artifacts, split_pool  # noqa: E402
from reidfm.data.pool import load_pool  # noqa: E402
from reidfm.data.teammate_splits import list_pending_teammate_splits  # noqa: E402
from reidfm.utils.paths import repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="configs/datasets/pool_v2.yaml")
    parser.add_argument("--out", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-samples-per-id", type=int, default=4)
    parser.add_argument("--dhash-threshold", type=int, default=6)
    parser.add_argument("--embedder", default="dhash", help="dhash (default) or dinov2_vits14 (optional, requires torch)")
    args = parser.parse_args()

    pool = load_pool(args.pool)
    if args.embedder != "dhash":
        print(f"note: embedder {args.embedder!r} not wired yet; using dhash pre-filter only")

    artifacts = split_pool(
        pool,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_samples_per_id=args.max_samples_per_id,
        dhash_threshold=args.dhash_threshold,
    )
    out_dir = repo_path(args.out)
    save_split_artifacts(artifacts, out_dir)
    val_manifest_dir = repo_path("pool/poolv2val")
    try:
        build_val_manifest(pool, artifacts.val_ids, val_manifest_dir, seed=args.seed)
        print(f"val manifest: {val_manifest_dir}")
    except FileNotFoundError as exc:
        print(f"skip val manifest (missing train data): {exc}")

    report_dir = repo_path("reports/leakage")
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Leakage split audit\n\n",
        f"- Pool: `{args.pool}`\n",
        f"- Identities sampled: {artifacts.meta.get('identities')}\n",
        f"- Components: {artifacts.meta.get('components')}\n",
        f"- Train fraction: {args.train_fraction}\n",
    ]
    (report_dir / "split_audit.md").write_text("".join(lines), encoding="utf-8")
    pending = list_pending_teammate_splits(pool.sources)
    if pending:
        print("teammate split files still missing (using dhash fallback for now):", ", ".join(pending))
    print(f"wrote split artifacts under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
