"""Read-only REIDPACK3 random-I/O diagnostic (except the optional local index cache).

Example on Kaggle, after the manifest and packs are attached:

    python scripts/diagnose_v3_pack_io.py \
        --manifest /kaggle/working/Person-Re-ID-Foundational-model-main/unified/final_dataset/Global/train_manifest.csv \
        --pack-root /kaggle/input/reid-fine-tuning-datasets-v3 \
        --samples-per-domain 30 --output /kaggle/working/v3_pack_io_report.json

The PACK root must be the directory containing the .pack files; the command
above is only an example. No training and no image decoding are performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reidfm.data.v3_pack_reader import V3PackReader  # noqa: E402


def _percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * pct))]


def run(manifest: Path, pack_root: Path, samples_per_domain: int):
    rng = random.Random(20261031)
    samples = defaultdict(list)
    counts = defaultdict(int)
    # Reservoir sample with bounded memory from potentially 782k+ manifest rows.
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            domain = row["dataset_root_id"]
            path = row["relative_path"]
            counts[domain] += 1
            count = counts[domain]
            bucket = samples[domain]
            if len(bucket) < samples_per_domain:
                bucket.append(path)
            else:
                choice = rng.randrange(count)
                if choice < samples_per_domain:
                    bucket[choice] = path

    report = {
        "manifest_rows": sum(counts.values()),
        "cache_enabled": os.environ.get("REIDFM_V3_INDEX_CACHE", "1") != "0",
        "samples_per_domain": samples_per_domain,
        "domains": {},
    }
    for domain in sorted(samples):
        started = time.perf_counter()
        reader = V3PackReader(pack_root, domain)
        open_seconds = time.perf_counter() - started
        try:
            queries = {}
            path = samples[domain][0]
            for column in ("physical_path", "context_path", "leaf_path", "basename"):
                query = f"SELECT offset, bytes, sha256 FROM images WHERE {column}=?"
                plan = reader.db.execute(
                    "EXPLAIN QUERY PLAN " + query, (path,)
                ).fetchall()
                queries[column] = [str(row[-1]) for row in plan]

            elapsed, payload_times = [], []
            original_load = reader._load

            def timed_load(*args):
                now = time.perf_counter()
                try:
                    return original_load(*args)
                finally:
                    payload_times.append(time.perf_counter() - now)

            reader._load = timed_load
            for relative_path in samples[domain]:
                now = time.perf_counter()
                reader.read_bytes(relative_path)
                elapsed.append(time.perf_counter() - now)

            report["domains"][domain] = {
                "manifest_rows": counts[domain],
                "pack_index_open_s": round(open_seconds, 3),
                "verified_local_index_bytes": reader._index_path.stat().st_size,
                "samples": len(elapsed),
                "read_mean_ms": round(statistics.mean(elapsed) * 1000, 3),
                "read_p50_ms": round(_percentile(elapsed, 0.50) * 1000, 3),
                "read_p95_ms": round(_percentile(elapsed, 0.95) * 1000, 3),
                "payload_sha_and_io_mean_ms": round(
                    statistics.mean(payload_times) * 1000, 3
                ) if payload_times else None,
                "query_plans": queries,
            }
        finally:
            reader.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--samples-per-domain", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.samples_per_domain <= 1000:
        parser.error("--samples-per-domain must be between 1 and 1000")
    report = run(args.manifest, args.pack_root, args.samples_per_domain)
    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print("[PASS] Report:", args.output)
    else:
        print(output)


if __name__ == "__main__":
    main()
