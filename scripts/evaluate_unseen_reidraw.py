"""Evaluate a trained checkpoint on Kaggle unseen ``.reidraw`` targets (V3 protocol)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.data.unseen_targets import (  # noqa: E402
    find_target_root,
    iter_target_records,
    load_target_specs,
    reidraw_files_by_root_id,
)
from reidfm.experiments.contract import HELD_OUT_TEST_KEYS  # noqa: E402
from reidfm.experiments.loss_study import expand_loss_study_run  # noqa: E402
from reidfm.fastreid_compat import ensure_fastreid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="model_best.pth or model_final.pth")
    parser.add_argument("--condition", default="", help="loss-study condition (B6, R0, …) to rebuild cfg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-file", default=None, help="default: config.yaml beside checkpoint")
    parser.add_argument("--target-root", default="", help="folder with *.reidraw + reid_targets_index.json")
    parser.add_argument("--name", default="")
    parser.add_argument("--out", default="results/paper_eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument(
        "--num-classes", type=int, default=10036,
        help="training identities; SphereFace2 heads need the real class count to build",
    )
    parser.add_argument(
        "--datasets", default=",".join(HELD_OUT_TEST_KEYS),
        help="comma-separated subset of held-out keys",
    )
    return parser.parse_args()


def _result_key_to_battery(key: str) -> str | None:
    dataset = key.split("|", 1)[0].strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "market1501": "market1501",
        "msmt17": "msmt17",
        "mars": "mars",
        "dukemtmc": "dukemtmc",
        "occludeddukemtmc": "occludedduke",
        "occludedduke": "occludedduke",
    }
    return aliases.get(dataset)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_fastreid()
    from reidfm.config import load_cfg, setup_cfg
    from reidfm.evaluation.paper_report import build_paper_payload, format_paper_markdown
    from reidfm.evaluation.v3_evaluator import evaluate_v3_packed_groups
    from reidfm.evaluation.extractors import build_fastreid_model
    from reidfm.utils.paths import repo_path
    from reidfm.utils.provenance import run_provenance, write_json

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    target_root = Path(args.target_root) if args.target_root else find_target_root()
    specs = load_target_specs(target_root)
    reidraw_map = reidraw_files_by_root_id(specs)

    datasets = [key.strip() for key in args.datasets.split(",") if key.strip()]
    unknown = [key for key in datasets if key not in HELD_OUT_TEST_KEYS]
    if unknown:
        raise ValueError(f"Unknown held-out keys {unknown}; choose from {list(HELD_OUT_TEST_KEYS)}")

    config_file = Path(args.config_file) if args.config_file else checkpoint.parent / "config.yaml"
    device_opts = ["MODEL.DEVICE", args.device, "MODEL.HEADS.NUM_CLASSES", str(args.num_classes)]
    if args.condition:
        run = expand_loss_study_run(args.condition, args.seed)
        cfg = setup_cfg(
            run.config_file,
            opts=run.opts + device_opts,
            model_key=run.init_model,
            output_dir=str(checkpoint.parent),
            prepare_weights=False,
            validate_study=False,
        )
    elif config_file.is_file():
        cfg = load_cfg(config_file, device_opts)
    else:
        raise FileNotFoundError(
            f"Missing {config_file}; pass --condition/--seed or place config.yaml next to the checkpoint."
        )

    started = time.perf_counter()
    model, cfg = build_fastreid_model(cfg, checkpoint, args.device)
    size_hw = tuple(int(x) for x in cfg.INPUT.SIZE_TEST)
    timings = OrderedDict(model_load_s=round(time.perf_counter() - started, 2))

    raw = OrderedDict()
    for key in datasets:
        records = list(iter_target_records(specs[key], target_root))
        n_query = sum(row.get("retrieval_role") == "query" for row in records)
        print(f"[{key}] {n_query} query / {len(records) - n_query} gallery images", flush=True)
        t0 = time.perf_counter()
        part = evaluate_v3_packed_groups(
            model,
            records,
            target_root,
            size_hw=size_hw,
            batch_size=args.batch_size,
            num_workers=args.workers,
            device=args.device,
            max_rank=args.max_rank,
            reidraw_files=reidraw_map,
        )
        timings[key] = round(time.perf_counter() - t0, 2)
        print(f"[{key}] done in {timings[key]:.1f}s", flush=True)
        raw.update((k, v) for k, v in part.items() if k != "mean")
    timings["total_s"] = round(time.perf_counter() - started, 2)

    battery_results = OrderedDict()
    for proto_key, metrics in raw.items():
        if proto_key == "mean":
            continue
        battery_key = _result_key_to_battery(proto_key)
        if battery_key is None:
            continue
        battery_results[battery_key] = {
            "Rank-1": metrics["Rank-1"],
            "Rank-5": metrics["Rank-5"],
            "Rank-10": metrics.get("Rank-10", metrics["Rank-5"]),
            "Rank-20": metrics.get("Rank-20", metrics["Rank-5"]),
            "mAP": metrics["mAP"],
            "mINP": metrics.get("mINP", 0.0),
            "cmc": list(metrics.get("cmc") or []),
        }
        if len(battery_results[battery_key]["cmc"]) < 20:
            cmc = battery_results[battery_key]["cmc"]
            last = cmc[-1] if cmc else battery_results[battery_key]["Rank-20"]
            battery_results[battery_key]["cmc"] = cmc + [last] * (20 - len(cmc))

    paper = build_paper_payload(battery_results)
    name = args.name or f"{args.condition or checkpoint.parent.name}_heldout"
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = format_paper_markdown(paper, title=f"{name} — unseen targets (reidraw)")
    print(markdown)
    write_json(
        out_dir / f"{name}.json",
        {
            "name": name,
            "checkpoint": str(checkpoint),
            "target_root": str(target_root),
            "condition": args.condition,
            "seed": args.seed,
            "datasets": datasets,
            "timings_s": timings,
            "protocol_results": raw,
            "battery_results": battery_results,
            "paper": paper,
            "provenance": run_provenance(),
        },
    )
    (out_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    print("timings (s): " + ", ".join(f"{k}={v}" for k, v in timings.items()))
    print(f"wrote {out_dir / name}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
