"""Aggregate held-out eval for fair BoT/AGW appendix baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.experiments.contract import FAIR_BASELINE_KEYS, HELD_OUT_TEST_KEYS
from reidfm.utils.config import load_yaml
from reidfm.utils.provenance import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiments/paper_fair_baselines.yaml")
    parser.add_argument("--pool", default="configs/datasets/pool_v2.yaml")
    parser.add_argument("--out", default="results/fair_baselines_appendix")
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_yaml(REPO_ROOT / args.manifest)
    root = manifest.get("output_root", "outputs/fair_baselines")
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in FAIR_BASELINE_KEYS:
        for seed in manifest.get("seeds") or [0]:
            run_dir = REPO_ROOT / root / key / f"seed{seed}"
            ckpt = run_dir / "model_best.pth"
            name = f"{key}_seed{seed}"
            result_json = out_dir / f"{name}.json"
            if args.eval and ckpt.is_file():
                subprocess.check_call(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "evaluate.py"),
                        "--checkpoint",
                        str(ckpt),
                        "--pool",
                        args.pool,
                        "--sets",
                        "test",
                        "--name",
                        name,
                        "--out",
                        str(out_dir),
                    ],
                    cwd=REPO_ROOT,
                )
            if not result_json.is_file():
                continue
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            per = {k: payload["results"][k] for k in HELD_OUT_TEST_KEYS if k in payload.get("results", {})}
            maps = [per[k]["mAP"] for k in per]
            rows.append({"baseline": key, "seed": seed, "macro_mAP": sum(maps) / len(maps) if maps else 0, "per_set": per})

    write_json(out_dir / "fair_baselines_appendix.json", {"rows": rows})
    lines = ["# Fair CNN baselines (appendix)", "", "| baseline | seed | macro mAP |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['baseline']} | {row['seed']} | {row['macro_mAP']:.2f} |")
    (out_dir / "fair_baselines_appendix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'fair_baselines_appendix.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
