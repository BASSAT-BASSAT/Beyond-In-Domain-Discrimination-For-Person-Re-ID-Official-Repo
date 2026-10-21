#!/usr/bin/env python3
"""Build train-id filters + ratio report for official native train / official-test-as-val.

Example:
    python scripts/prepare_datasets.py --pool configs/datasets/pool_v2_official_val.yaml
    python scripts/build_official_native_splits.py --pool configs/datasets/pool_v2_official_val.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.data.leakage_split import (  # noqa: E402
    build_val_manifest,
    load_train_manifests,
    save_split_artifacts,
    split_pool,
)
from reidfm.data.official_native_splits import (  # noqa: E402
    CANONICAL_REFERENCE_RATIOS,
    OFFICIAL_IN_TRAINING_VAL_KEYS,
    manifest_ratio_report,
)
from reidfm.data.pool import load_pool  # noqa: E402
from reidfm.utils.paths import repo_path  # noqa: E402


def _write_ratios_md(reports: list[dict], pool_name: str, val_sets: list[str], out: Path) -> None:
    lines = [
        "# Official native split ratios\n\n",
        f"Pool: `{pool_name}`\n\n",
        "Training: all `manifest.train` identities (`splits/pool_v2_train_ids.json`).\n",
        "In-training validation: official query/gallery for datasets in `val_sets` "
        "(not the five paper held-out benchmarks).\n\n",
        "| Dataset | Train IDs | Train imgs | Eval IDs | Query | Gallery | ID train% | Img train% | In-training val |\n",
        "|---------|-----------|------------|----------|-------|---------|-----------|------------|-----------------|\n",
    ]
    for row in reports:
        id_pct = f"{100 * row['id_ratio_train']:.1f}%" if row.get("id_ratio_train") is not None else "—"
        img_pct = f"{100 * row['image_ratio_train']:.1f}%" if row.get("image_ratio_train") is not None else "—"
        lines.append(
            f"| {row['dataset']} | {row['n_train_ids']} | {row['n_train_images']} | {row['n_eval_ids']} "
            f"| {row['n_query_images']} | {row['n_gallery_images']} | {id_pct} | {img_pct} "
            f"| {'yes' if row['in_training_val'] else 'no'} |\n"
        )
    lines.append("\n## Reference (before manifests exist)\n\n")
    for key, ref in CANONICAL_REFERENCE_RATIOS.items():
        if key not in val_sets and key not in ("mot17", "mot20", "rpifield"):
            continue
        lines.append(f"### {key}\n\n```json\n{json.dumps(ref, indent=2)}\n```\n\n")
    out.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="configs/datasets/pool_v2_official_val.yaml")
    parser.add_argument("--out", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    pool = load_pool(args.pool)
    artifacts = split_pool(pool, seed=args.seed, train_fraction=args.train_fraction)
    out_dir = repo_path(args.out)
    save_split_artifacts(artifacts, out_dir)

    supplemental = {k: v for k, v in artifacts.val_ids.items() if v}
    if supplemental:
        val_manifest_dir = repo_path("pool/poolv2val")
        try:
            build_val_manifest(pool, supplemental, val_manifest_dir, seed=args.seed)
            print(f"supplemental val manifest (MOT/RPIfield/…): {val_manifest_dir}")
        except FileNotFoundError as exc:
            print(f"skip supplemental val manifest: {exc}")

    manifests = load_train_manifests(pool)
    reports: list[dict] = []
    for key in pool.sources:
        if key in manifests:
            in_val = key in pool.val_sets
            reports.append(manifest_ratio_report(key, manifests[key], in_training_val=in_val).to_dict())
        elif key in CANONICAL_REFERENCE_RATIOS:
            ref = dict(CANONICAL_REFERENCE_RATIOS[key])
            ref["dataset"] = key
            ref.setdefault("n_train_ids", 0)
            ref.setdefault("n_train_images", 0)
            ref.setdefault("n_eval_ids", 0)
            ref.setdefault("n_query_images", 0)
            ref.setdefault("n_gallery_images", 0)
            ref["in_training_val"] = key in pool.val_sets
            reports.append(ref)

    ratios_path = out_dir / "official_native_ratios.json"
    ratios_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (out_dir / "official_val_sets.json").write_text(json.dumps(pool.val_sets, indent=2), encoding="utf-8")
    _write_ratios_md(reports, pool.name, pool.val_sets, repo_path("docs/OFFICIAL_NATIVE_SPLITS.md"))

    print(f"wrote {out_dir / 'pool_v2_train_ids.json'} (official train IDs only; val_ids empty for native sources)")
    print(f"in-training val datasets: {pool.val_sets}")
    print(f"ratios: {ratios_path} and docs/OFFICIAL_NATIVE_SPLITS.md")
    if set(pool.val_sets) != set(OFFICIAL_IN_TRAINING_VAL_KEYS):
        print("note: val_sets differ from default OFFICIAL_IN_TRAINING_VAL_KEYS; check pool yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
