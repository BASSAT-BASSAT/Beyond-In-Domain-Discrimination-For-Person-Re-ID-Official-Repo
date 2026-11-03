"""Evaluate a trained checkpoint on Kaggle unseen ``.reidraw`` targets (V3 protocol)."""

from __future__ import annotations

import argparse
import logging
import sys
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

    config_file = Path(args.config_file) if args.config_file else checkpoint.parent / "config.yaml"
    if args.condition:
        run = expand_loss_study_run(args.condition, args.seed)
        cfg = setup_cfg(
            run.config_file,
            opts=run.opts + ["MODEL.DEVICE", args.device],
            model_key=run.init_model,
            output_dir=str(checkpoint.parent),
            prepare_weights=False,
            validate_study=False,
        )
    elif config_file.is_file():
        cfg = load_cfg(config_file, ["MODEL.DEVICE", args.device])
    else:
        raise FileNotFoundError(
            f"Missing {config_file}; pass --condition/--seed or place config.yaml next to the checkpoint."
        )

    model, cfg = build_fastreid_model(cfg, checkpoint, args.device)
    size_hw = tuple(int(x) for x in cfg.INPUT.SIZE_TEST)

    all_records = []
    for key in HELD_OUT_TEST_KEYS:
        spec = specs[key]
        all_records.extend(iter_target_records(spec, target_root))

    raw = evaluate_v3_packed_groups(
        model,
        all_records,
        target_root,
        size_hw=size_hw,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=args.device,
        max_rank=args.max_rank,
        reidraw_files=reidraw_map,
    )

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
            "protocol_results": raw,
            "battery_results": battery_results,
            "paper": paper,
            "provenance": run_provenance(),
        },
    )
    (out_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {out_dir / name}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
