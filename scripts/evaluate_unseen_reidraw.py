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
    parser.add_argument(
        "--config-file",
        default=None,
        help="training config.yaml (default: beside checkpoint; else loss-study base if --condition)",
    )
    parser.add_argument("--target-root", default="", help="folder with *.reidraw + reid_targets_index.json")
    parser.add_argument("--name", default="")
    parser.add_argument("--out", default="results/paper_eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument(
        "--num-classes", type=int, default=10036,
        help="training identities; SphereFace2 / angular heads need the real class count",
    )
    parser.add_argument(
        "--datasets", default=",".join(HELD_OUT_TEST_KEYS),
        help="comma-separated subset of held-out keys",
    )
    parser.add_argument(
        "--video-frames", type=int, default=8,
        help="frames sampled per tracklet for video benchmarks (MARS), mean-pooled per tracklet",
    )
    return parser.parse_args()


VIDEO_TARGETS = frozenset({"mars"})


def _result_key_to_battery(key: str) -> str | None:
    dataset = key.split("|", 1)[0].strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "market1501": "market1501",
        "msmt17": "msmt17",
        "msmt17v2": "msmt17",
        "mars": "mars",
        "dukemtmc": "dukemtmc",
        "occludeddukemtmc": "occludedduke",
        "occludedduke": "occludedduke",
    }
    return aliases.get(dataset)


def _resolve_eval_config(args, checkpoint: Path):
    """Pick config path + opts without requiring pool_v2.yaml on disk."""
    from reidfm.utils.paths import repo_path

    explicit = Path(args.config_file) if args.config_file else None
    ckpt_cfg = checkpoint.parent / "config.yaml"
    opts: list[str] = ["MODEL.DEVICE", args.device, "MODEL.HEADS.NUM_CLASSES", str(args.num_classes)]
    model_key = None
    config_file: Path | None = None

    if args.condition:
        run = expand_loss_study_run(args.condition, args.seed)
        opts = run.opts + opts
        model_key = run.init_model
        if explicit and explicit.is_file():
            config_file = explicit
        elif ckpt_cfg.is_file():
            config_file = ckpt_cfg
        else:
            config_file = repo_path(run.config_file)
    elif explicit and explicit.is_file():
        config_file = explicit
    elif ckpt_cfg.is_file():
        config_file = ckpt_cfg
    else:
        raise FileNotFoundError(
            f"Missing config beside checkpoint ({ckpt_cfg}); pass --condition/--seed or --config-file."
        )
    return config_file, opts, model_key


def setup_eval_cfg_for_heldout(
    config_file,
    opts=None,
    model_key=None,
    output_dir=None,
    num_classes=10036,
):
    """Eval-only cfg: never loads ``pool_v2.yaml`` (works on old clones without ``setup_eval_cfg`` in config)."""
    import reidfm.config as config_mod

    if hasattr(config_mod, "setup_eval_cfg"):
        return config_mod.setup_eval_cfg(
            config_file,
            opts=opts,
            model_key=model_key,
            output_dir=output_dir,
            num_classes=num_classes,
        )
    from reidfm.config import apply_init_model, load_cfg, resolve_output_dir

    cfg = load_cfg(config_file, opts)
    cfg.DATASETS.NAMES = ()
    cfg.DATASETS.TESTS = ()
    if int(cfg.MODEL.HEADS.NUM_CLASSES) <= 0:
        cfg.MODEL.HEADS.NUM_CLASSES = int(num_classes)
    apply_init_model(cfg, model_key, prepare_weights=False)
    resolve_output_dir(cfg, output_dir)
    cfg.freeze()
    return cfg


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_fastreid()
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

    config_file, opts, model_key = _resolve_eval_config(args, checkpoint)
    cfg = setup_eval_cfg_for_heldout(
        config_file,
        opts=opts,
        model_key=model_key,
        output_dir=str(checkpoint.parent),
        num_classes=args.num_classes,
    )

    started = time.perf_counter()
    model, cfg = build_fastreid_model(cfg, checkpoint, args.device)
    size_hw = tuple(int(x) for x in cfg.INPUT.SIZE_TEST)
    timings = OrderedDict(model_load_s=round(time.perf_counter() - started, 2))

    raw = OrderedDict()
    for key in datasets:
        records = list(iter_target_records(specs[key], target_root))
        n_query = sum(row.get("retrieval_role") == "query" for row in records)
        tracklet_frames = args.video_frames if key in VIDEO_TARGETS else 0
        print(f"[{key}] {n_query} query / {len(records) - n_query} gallery images", flush=True)
        if tracklet_frames:
            print(f"[{key}] video protocol: tracklets, {tracklet_frames} frames each, mean-pooled", flush=True)
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
            tracklet_frames=tracklet_frames,
        )
        if tracklet_frames:
            for proto_key, metrics in part.items():
                if proto_key != "mean":
                    print(f"[{key}] {metrics['num_query']} query / {metrics['num_gallery']} gallery tracklets", flush=True)
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

    paper = build_paper_payload(battery_results, held_out_keys=tuple(datasets))
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
