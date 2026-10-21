"""Train fair BoT/AGW appendix baselines on pool v2 (ImageNet init)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.utils.config import load_yaml
from reidfm.utils.provenance import run_provenance, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiments/paper_fair_baselines.yaml")
    parser.add_argument("--baseline", action="append", help="bot_r50_ibn | agw_r50_ibn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_yaml(REPO_ROOT / args.manifest)
    root = manifest.get("output_root", "outputs/fair_baselines")
    keys = args.baseline or list(manifest["baselines"])
    seeds = [args.seed] if args.baseline else list(manifest.get("seeds") or [0])

    for key in keys:
        entry = manifest["baselines"][key]
        for seed in seeds:
            out = f"{root}/{key}/seed{seed}"
            opts = ["REIDFM.STUDY_MODE", "true", "REIDFM.SEED", str(seed)]
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "train.py"),
                "--config-file",
                entry["config"],
                "--model",
                entry["init_model"],
                "--output-dir",
                out,
                "--num-gpus",
                str(args.num_gpus),
                *opts,
                *args.train_args,
            ]
            if args.resume:
                cmd.append("--resume")
            if args.fresh:
                cmd.append("--fresh")
            if args.dry_run:
                print(" ".join(cmd))
                continue
            Path(out).mkdir(parents=True, exist_ok=True)
            write_json(Path(out) / "fair_baseline_meta.json", {"baseline": key, "seed": seed, "provenance": run_provenance()})
            subprocess.check_call(cmd, cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
