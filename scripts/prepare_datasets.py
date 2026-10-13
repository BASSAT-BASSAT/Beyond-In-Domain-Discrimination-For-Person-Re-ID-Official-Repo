"""Convert raw datasets into the unified pool and audit it.

Examples:
    # everything in pool v1 that exists under $REIDFM_RAW_ROOT, copied into ./pool
    python scripts/prepare_datasets.py --pool configs/datasets/pool_v1.yaml

    # only two datasets, hard-linked instead of copied
    python scripts/prepare_datasets.py --datasets market1501 mars --materialize link

    # count what a converter would produce without writing anything
    python scripts/prepare_datasets.py --datasets deepchange --dry-run

    # audit an existing pool (e.g. Kaggle inputs) without converting
    python scripts/prepare_datasets.py --audit-only --dedup-check
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tabulate import tabulate  # noqa: E402

from reidfm.data.audit import audit_manifest, check_pool_roles, cross_dataset_duplicates  # noqa: E402
from reidfm.data.converters import build_dataset, collect_rows  # noqa: E402
from reidfm.data.manifest import load_manifest  # noqa: E402
from reidfm.data.pool import load_pool  # noqa: E402
from reidfm.utils.paths import data_roots, raw_root  # noqa: E402
from reidfm.utils.provenance import run_provenance, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="configs/datasets/pool_v2.yaml")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of pool datasets (default: all)")
    parser.add_argument("--raw-root", type=Path, default=None, help="default: $REIDFM_RAW_ROOT or ./raw")
    parser.add_argument("--out-root", type=Path, default=None, help="default: first $REIDFM_DATA_ROOT entry or ./pool")
    parser.add_argument("--materialize", choices=("copy", "link", "none"), default="copy")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-missing", action="store_true", help="skip datasets whose raw folder is absent")
    parser.add_argument("--overwrite", action="store_true", help="rebuild datasets that already have a manifest")
    parser.add_argument("--dry-run", action="store_true", help="run converters and print counts, write nothing")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dedup-check", action="store_true", help="dHash near-duplicate check on known-overlap pairs")
    parser.add_argument("--dedup-sample", type=int, default=3000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = load_pool(args.pool)
    keys = args.datasets or pool.all_datasets
    unknown = sorted(set(keys) - set(pool.all_datasets))
    if unknown:
        print(f"note: {unknown} are not in pool {pool.name}; converting with default options")
    raw = args.raw_root or raw_root()
    out_root = args.out_root or data_roots()[0]

    if args.dry_run:
        table = []
        for key in keys:
            try:
                src, rows = collect_rows(key, pool.converter_options(key), raw, include_train=key in pool.sources)
            except (FileNotFoundError, ValueError) as exc:
                table.append([key, "-", "-", "-", str(exc)[:80]])
                continue
            counts = Counter(row.split for row in rows)
            ids = len({row.pid for row in rows if row.split == "train"})
            table.append([key, counts.get("train", 0), ids, f"{counts.get('query', 0)}/{counts.get('gallery', 0)}", str(src)])
        print(tabulate(table, headers=["dataset", "train imgs", "train ids", "query/gallery", "raw / error"]))
        return 0

    if not args.audit_only:
        for key in keys:
            if (out_root / key / "manifest.csv").is_file() and not args.overwrite:
                print(f"[{key}] manifest exists, skipping (use --overwrite)")
                continue
            try:
                meta = build_dataset(
                    key,
                    out_root,
                    pool.converter_options(key),
                    raw,
                    mode=args.materialize,
                    workers=args.workers,
                    include_train=key in pool.sources or key not in pool.all_datasets,
                )
            except FileNotFoundError as exc:
                if args.skip_missing:
                    print(f"[{key}] skipped: {exc}")
                    continue
                raise
            print(f"[{key}] {meta['counts']} in {meta['seconds']}s")

    manifests = {}
    for key in keys:
        try:
            manifests[key] = load_manifest(key, roots=[out_root, *data_roots()])
        except FileNotFoundError:
            print(f"[{key}] no manifest found")

    audits = [audit_manifest(manifest) for manifest in manifests.values()]
    rows = []
    for audit in audits:
        c = audit.counts
        rows.append([
            audit.key,
            c["train"]["ids"], c["train"]["images"], c["train"]["cams"],
            c["query"]["ids"], c["query"]["images"], c["gallery"]["images"],
            c["query"]["tracklets"] + c["gallery"]["tracklets"] or "",
            "ERR" if audit.errors else ("warn" if audit.warnings else "ok"),
        ])
    print(tabulate(rows, headers=["dataset", "train ids", "train imgs", "train cams", "q ids", "q imgs", "g imgs", "eval tracklets", "status"]))
    for audit in audits:
        for message in audit.errors:
            print(f"ERROR   [{audit.key}] {message}")
        for message in audit.warnings:
            print(f"warning [{audit.key}] {message}")

    role_errors = check_pool_roles(pool, manifests)
    for message in role_errors:
        print(f"ERROR   [pool] {message}")

    duplicates = cross_dataset_duplicates(pool, manifests, sample=args.dedup_sample) if args.dedup_check else []
    for item in duplicates:
        print(f"dedup   {item['train']} train vs {item['test']} test: {item['near_duplicates']}/{item['sampled_test']} near-duplicates ({item['reason']})")

    report = {
        "pool": pool.name,
        "out_root": str(out_root),
        "datasets": {a.key: {"counts": a.counts, "errors": a.errors, "warnings": a.warnings} for a in audits},
        "pool_errors": role_errors,
        "near_duplicates": duplicates,
        "provenance": run_provenance(),
    }
    path = write_json(out_root / "_audit" / f"{pool.name}.json", report)
    print(f"audit written to {path}")
    return 1 if role_errors or any(a.errors for a in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
