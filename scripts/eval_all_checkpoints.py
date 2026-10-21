"""Evaluate many checkpoints listed in a JSON manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON list of {checkpoint, name, out?}")
    parser.add_argument("--pool", default="configs/datasets/pool_v2.yaml")
    parser.add_argument("--sets", default="test")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    for entry in entries:
        ckpt = entry["checkpoint"]
        name = entry.get("name") or Path(ckpt).parent.name
        out = entry.get("out", args.out)
        subprocess.check_call(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate.py"),
                "--checkpoint",
                ckpt,
                "--pool",
                args.pool,
                "--sets",
                args.sets,
                "--name",
                name,
                "--out",
                out,
            ],
            cwd=REPO_ROOT,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
