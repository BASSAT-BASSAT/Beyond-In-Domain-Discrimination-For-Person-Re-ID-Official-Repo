"""Evaluate a registered model or a trained checkpoint on pool test sets (benchmark protocol).

Examples:
    # zero-shot starting point on every test split + OOD sets
    python scripts/evaluate.py --model sbs_r50_ibn_msmt --sets all
    # a trained run (config.yaml next to the checkpoint is used)
    python scripts/evaluate.py --checkpoint outputs/sbs_r50ibn_msmt_v1/model_best.pth --sets test,ood
    # explicit dataset keys
    python scripts/evaluate.py --model osnet_ain_msdc --sets market1501,mars

--sets accepts: val | test | ood | all | comma-separated dataset keys (mixable: "test,prid2011").
Writes <out>/<name>.json and <name>.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.data.pool import load_pool  # noqa: E402
from reidfm.fastreid_compat import ensure_fastreid  # noqa: E402


def resolve_sets(spec: str, pool) -> list[str]:
    keys: list[str] = []
    groups = {"val": pool.val_sets, "test": pool.test_sets, "ood": pool.ood, "all": [*pool.test_sets, *pool.ood]}
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        for key in groups.get(part, [part]):
            if key not in keys:
                keys.append(key)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="registry key (scripts/list_models.py)")
    source.add_argument("--checkpoint", help="FastReID checkpoint (.pth)")
    parser.add_argument("--config-file", default=None, help="default: config.yaml next to --checkpoint, or the model's config")
    parser.add_argument("--pool", default="configs/datasets/pool_v2.yaml")
    parser.add_argument("--sets", default="all")
    parser.add_argument("--num-frames", type=int, default=8, help="frames per tracklet for video sets")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None, help="image-decoding processes (default 2 on Windows, 4 elsewhere)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="fp32 features (bit-closer to the benchmark)")
    parser.add_argument("--skip-missing", action="store_true", help="skip sets whose manifest is absent")
    parser.add_argument("--name", default=None, help="report name (default: model key or run dir name)")
    parser.add_argument("--out", default="results", help="output folder (relative to the repo root)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="cfg overrides for the model config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_fastreid()
    from reidfm.evaluation.evaluator import evaluate_sets, results_markdown
    from reidfm.evaluation.extractors import build_extractor
    from reidfm.utils.paths import repo_path
    from reidfm.utils.provenance import run_provenance, write_json

    pool = load_pool(args.pool)
    keys = resolve_sets(args.sets, pool)
    extractor, source_info = build_extractor(
        model_key=args.model,
        checkpoint=args.checkpoint,
        config_file=args.config_file,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=not args.no_amp,
        opts=args.opts,
    )
    results = evaluate_sets(keys, extractor, num_frames=args.num_frames, max_queries=args.max_queries,
                            skip_missing=args.skip_missing)

    name = args.name or args.model or f"{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    out_dir = repo_path(args.out)
    from reidfm.evaluation.evaluator import mean_row

    ood = [k for k in results if k in pool.ood]
    held_out = {k: v for k, v in results.items() if k != "mean" and k not in pool.ood}
    summary = {
        "mean_all": results.get("mean", {}),
        "mean_held_out": mean_row(held_out),
        "mean_in_domain": mean_row(held_out),
        "mean_ood": mean_row({k: results[k] for k in ood}),
    }
    title = f"{name} ({'OOD: ' + ', '.join(ood) if ood else 'held-out test battery'})"
    markdown = results_markdown(results, title)
    for label, row in (("held-out mean", summary["mean_held_out"]), ("OOD mean", summary["mean_ood"])):
        if row:
            markdown += f"\n{label}: R1 {row['Rank-1']:.2f} / mAP {row['mAP']:.2f} / mINP {row['mINP']:.2f}"
    markdown += "\n"
    print(markdown)
    write_json(out_dir / f"{name}.json", {
        "name": name,
        "source": source_info,
        "pool": pool.name,
        "sets": keys,
        "ood_sets": ood,
        "num_frames": args.num_frames,
        "amp": not args.no_amp,
        "results": results,
        "summary": summary,
        "provenance": run_provenance(),
    })
    (out_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {out_dir / name}.json/.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
