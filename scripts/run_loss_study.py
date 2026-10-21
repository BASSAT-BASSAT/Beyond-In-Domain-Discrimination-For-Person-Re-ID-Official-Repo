"""Run the nine-condition angular-loss study (see configs/experiments/loss_study_9.yaml)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.experiments.contract import LOSS_STUDY_CONDITIONS
from reidfm.experiments.loss_study import expand_loss_study_run, flatten_opts_for_cli, load_loss_study_manifest
from reidfm.experiments.protocol import config_fingerprint
from reidfm.utils.provenance import run_provenance, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiments/loss_study_9.yaml")
    parser.add_argument("--condition", action="append", help="e.g. R0 or B2 (repeatable)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--screening", action="store_true", help="run manifest screening list only")
    parser.add_argument("--all", action="store_true", help="all nine conditions")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default="outputs/loss_study")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="extra args passed to train.py")
    return parser.parse_args()


def _train_cmd(run, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config-file",
        run.config_file,
        "--model",
        run.init_model,
        "--output-dir",
        run.output_dir,
        "--num-gpus",
        str(args.num_gpus),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.fresh:
        cmd.append("--fresh")
    cmd.extend(flatten_opts_for_cli(run.opts))
    cmd.extend(args.train_args)
    return cmd


def main() -> int:
    args = parse_args()
    manifest = load_loss_study_manifest(args.manifest)
    if args.all:
        conditions = list(LOSS_STUDY_CONDITIONS)
    elif args.screening:
        conditions = list(manifest.get("screening") or LOSS_STUDY_CONDITIONS[:6])
    elif args.condition:
        conditions = args.condition
    else:
        print("specify --condition, --screening, or --all", file=sys.stderr)
        return 2

    seeds = [args.seed] if args.condition or not args.all else list(manifest.get("seeds") or [0])
    if args.all and args.seed >= 0 and not args.condition:
        seeds = [args.seed]

    plan = []
    for condition in conditions:
        for seed in seeds:
            run = expand_loss_study_run(condition, seed, manifest, output_root=args.output_root)
            from reidfm.config import setup_cfg

            fp = ""
            if not args.dry_run:
                cfg = setup_cfg(
                    run.config_file,
                    opts=run.opts,
                    model_key=run.init_model,
                    output_dir=run.output_dir,
                    prepare_weights=False,
                )
                fp = config_fingerprint(cfg)
            meta_path = Path(run.output_dir) / "loss_study_meta.json"
            if fp and meta_path.is_file() and args.resume and not args.fresh:
                prev = json.loads(meta_path.read_text(encoding="utf-8"))
                if prev.get("config_fingerprint") and prev["config_fingerprint"] != fp:
                    raise SystemExit(
                        f"refusing resume: config fingerprint changed for {condition} seed {seed} "
                        f"({prev['config_fingerprint']} vs {fp})"
                    )
            plan.append((run, fp))

    if args.dry_run:
        for run, fp in plan:
            print(" ".join(_train_cmd(run, args)))
            print(f"  fingerprint={fp}")
        return 0

    for run, fp in plan:
        Path(run.output_dir).mkdir(parents=True, exist_ok=True)
        write_json(
            Path(run.output_dir) / "loss_study_meta.json",
            {
                "condition": run.condition,
                "seed": run.seed,
                "config_fingerprint": fp,
                "provenance": run_provenance(manifest=args.manifest),
            },
        )
        cmd = _train_cmd(run, args)
        print("running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
