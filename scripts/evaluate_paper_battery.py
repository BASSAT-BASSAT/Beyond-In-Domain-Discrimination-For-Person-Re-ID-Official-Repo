"""Evaluate a checkpoint on the paper held-out battery with R1, R5, mAP, and CMC curves.

Example:
    python scripts/evaluate_paper_battery.py \\
        --checkpoint outputs/loss_study/B6/seed0/model_best.pth \\
        --name B6_seed0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.data.pool import load_pool  # noqa: E402
from reidfm.experiments.contract import HELD_OUT_TEST_KEYS  # noqa: E402
from reidfm.fastreid_compat import ensure_fastreid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="registry key (scripts/list_models.py)")
    source.add_argument("--checkpoint", help="FastReID checkpoint (.pth)")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--pool", default="configs/datasets/pool_v2.yaml")
    parser.add_argument(
        "--extra-sets",
        default="",
        help="comma-separated extra dataset keys (e.g. alexandria) evaluated but excluded from macro_public",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default="results/paper_eval")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_fastreid()
    from reidfm.evaluation.evaluator import evaluate_sets
    from reidfm.evaluation.extractors import build_extractor
    from reidfm.evaluation.paper_report import build_paper_payload, format_paper_markdown
    from reidfm.utils.paths import repo_path
    from reidfm.utils.provenance import run_provenance, write_json

    pool = load_pool(args.pool)
    extra = [p.strip() for p in args.extra_sets.split(",") if p.strip()]
    keys = list(HELD_OUT_TEST_KEYS) + [k for k in extra if k not in HELD_OUT_TEST_KEYS]

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
    results = evaluate_sets(
        keys,
        extractor,
        num_frames=args.num_frames,
        max_queries=args.max_queries,
        skip_missing=args.skip_missing,
    )
    paper = build_paper_payload(results, extra_keys=extra)
    name = args.name or args.model or f"{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = format_paper_markdown(paper, title=f"{name} — held-out battery")
    print(markdown)
    payload = {
        "name": name,
        "source": source_info,
        "pool": pool.name,
        "held_out_keys": list(HELD_OUT_TEST_KEYS),
        "extra_sets": extra,
        "results_raw": results,
        "paper": paper,
        "provenance": run_provenance(),
    }
    write_json(out_dir / f"{name}.json", payload)
    (out_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {out_dir / name}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
