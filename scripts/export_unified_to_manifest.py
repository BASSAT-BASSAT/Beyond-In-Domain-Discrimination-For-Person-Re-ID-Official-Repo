#!/usr/bin/env python3
"""Export a Unified Schema v1 CSV to reidfm manifest.csv (+ meta.json).

Examples:
    python scripts/export_unified_to_manifest.py \\
        --unified unified/v1/cuhk03_np_unified_v1.csv \\
        --out pool/cuhk03np --dataset cuhk03np
    python scripts/export_unified_to_manifest.py \\
        --unified manifests/LaST/last_unified_v1.csv --out pool/last
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.data.unified_export import export_unified_csv  # noqa: E402
from reidfm.utils.paths import repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unified", required=True, help="path to unified v1 CSV")
    parser.add_argument("--out", required=True, help="output folder for manifest.csv (relative to repo ok)")
    parser.add_argument("--dataset", default=None, help="registry key override")
    parser.add_argument("--path-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--kind", default="image", choices=("image", "video"))
    args = parser.parse_args()

    meta = export_unified_csv(
        repo_path(args.unified),
        repo_path(args.out),
        dataset_key=args.dataset,
        path_mode=args.path_mode,
        kind=args.kind,
    )
    print(f"wrote {args.out}/{meta['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
