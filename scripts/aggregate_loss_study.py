"""Aggregate held-out eval results for the nine-condition loss study (main table)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.experiments.contract import HELD_OUT_TEST_KEYS, LOSS_STUDY_CONDITIONS
from reidfm.experiments.loss_study import expand_loss_study_run, load_loss_study_manifest
from reidfm.utils.provenance import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/loss_study")
    parser.add_argument("--manifest", default="configs/experiments/loss_study_9.yaml")
    parser.add_argument("--pool", default="configs/datasets/pool_v2.yaml")
    parser.add_argument("--out", default="results/loss_study_main")
    parser.add_argument("--eval", action="store_true", help="run evaluate.py before aggregating")
    parser.add_argument("--seeds", default="", help="comma-separated seeds (default: manifest seeds)")
    return parser.parse_args()


def _load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _macro(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    args = parse_args()
    manifest = load_loss_study_manifest(args.manifest)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or list(manifest.get("seeds") or [0])
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    table: list[dict] = []
    for condition in LOSS_STUDY_CONDITIONS:
        for seed in seeds:
            run = expand_loss_study_run(condition, seed, manifest, output_root=args.root)
            ckpt = Path(run.output_dir) / "model_best.pth"
            result_name = f"{condition}_seed{seed}"
            result_json = out_dir / f"{result_name}.json"
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
                        result_name,
                        "--out",
                        str(out_dir),
                    ],
                    cwd=REPO_ROOT,
                )
            if not result_json.is_file():
                continue
            payload = _load_result(result_json)
            per_set = []
            for key in HELD_OUT_TEST_KEYS:
                if key in payload.get("results", {}):
                    m = payload["results"][key]
                    per_set.append({"dataset": key, "mAP": m.get("mAP"), "Rank-1": m.get("Rank-1")})
            table.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "per_set": per_set,
                    "macro_mAP": _macro(per_set, "mAP"),
                    "macro_R1": _macro(per_set, "Rank-1"),
                }
            )

    write_json(out_dir / "loss_study_main.json", {"rows": table})
    md = ["# Loss study (main table)", "", "| condition | seed | macro mAP | macro R1 |", "|---|---:|---:|---:|"]
    for row in table:
        md.append(f"| {row['condition']} | {row['seed']} | {row['macro_mAP']:.2f} | {row['macro_R1']:.2f} |")
    (out_dir / "loss_study_main.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    csv_path = out_dir / "loss_study_main.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["condition", "seed", "macro_mAP", "macro_R1"] + list(HELD_OUT_TEST_KEYS)
        writer.writerow(header)
        for row in table:
            by = {p["dataset"]: p for p in row["per_set"]}
            writer.writerow(
                [row["condition"], row["seed"], f"{row['macro_mAP']:.4f}", f"{row['macro_R1']:.4f}"]
                + [f"{by.get(k, {}).get('mAP', '')}" for k in HELD_OUT_TEST_KEYS]
            )
    print(f"wrote {out_dir / 'loss_study_main.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
