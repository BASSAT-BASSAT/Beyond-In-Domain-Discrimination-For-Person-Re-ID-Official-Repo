"""Train on the multi-source pool with FastReID's trainer (single GPU or DDP).

Examples:
    python scripts/train.py --config-file configs/finetune_sbs_r50ibn_msmt.yml --num-gpus 2
    python scripts/train.py --config-file configs/finetune_sbs_r50ibn_msmt.yml --num-gpus 2 --resume
    # Kaggle, next session: copy the previous session's run dir in, then resume
    python scripts/train.py --config-file configs/finetune_sbs_r50ibn_msmt.yml --num-gpus 2 \
        --resume --restore-from /kaggle/input/reidfm-run-s1
    # override the starting point / pool / anything in the cfg
    python scripts/train.py --config-file configs/finetune_sbs_r50ibn_msmt.yml --model sbs_r50_ibn_market \
        SOLVER.BASE_LR 0.0002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.fastreid_compat import ensure_fastreid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model", default=None, help="registry key of the init model (overrides REIDFM.INIT_MODEL)")
    parser.add_argument("--pool", default=None, help="pool yaml (overrides REIDFM.POOL)")
    parser.add_argument("--output-dir", default=None, help="absolute, or relative to $REIDFM_OUTPUT_ROOT")
    parser.add_argument("--resume", action="store_true", help="continue from OUTPUT_DIR/last_checkpoint")
    parser.add_argument("--restore-from", default=None, help="copy a previous run dir (read-only input) into OUTPUT_DIR first")
    parser.add_argument("--time-budget-h", type=float, default=None, help="stop cleanly after this many hours (0 = off)")
    parser.add_argument("--eval-only", action="store_true", help="evaluate MODEL.WEIGHTS / last checkpoint on the val sets")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:50152")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="FastReID cfg overrides: KEY VALUE ...")
    return parser.parse_args()


def build_cfg(args, prepare_weights: bool):
    from reidfm.config import setup_cfg

    return setup_cfg(
        args.config_file,
        opts=args.opts,
        model_key=args.model,
        pool=args.pool,
        output_dir=args.output_dir,
        time_budget_h=args.time_budget_h,
        prepare_weights=prepare_weights,
    )


def main(args) -> dict | None:
    ensure_fastreid()
    from fastreid.engine import default_setup
    from fastreid.utils import comm

    from reidfm.engine.trainer import ReidfmTrainer
    from reidfm.utils.provenance import run_provenance, write_json

    # Init weights were converted by the launcher; fresh runs reuse the cached file.
    cfg = build_cfg(args, prepare_weights=not args.resume)
    default_setup(cfg, args)
    if cfg.REIDFM.SEED >= 0:
        from fastreid.utils.env import seed_all_rng

        seed_all_rng(cfg.REIDFM.SEED + comm.get_rank())
    from fastreid.utils.logger import setup_logger

    setup_logger(cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="reidfm")

    if args.eval_only:
        from fastreid.utils.checkpoint import Checkpointer

        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()
        model = ReidfmTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return ReidfmTrainer.test(cfg, model)

    if comm.is_main_process():
        write_json(Path(cfg.OUTPUT_DIR) / "reidfm_run.json", run_provenance(config_file=args.config_file, argv=sys.argv))
    trainer = ReidfmTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def launch_main() -> int:
    args = parse_args()
    ensure_fastreid()
    from fastreid.engine import launch

    # Resolve the output dir, restore a previous session, and download/convert init weights
    # once here, so DDP workers don't race on the same files.
    cfg = build_cfg(args, prepare_weights=not args.resume)
    if args.restore_from:
        from reidfm.engine.resume import restore_run_dir

        restore_run_dir(args.restore_from, cfg.OUTPUT_DIR)
        args.resume = True
    from reidfm.utils.paths import repo_path

    args.output_dir = cfg.OUTPUT_DIR
    args.config_file = str(repo_path(args.config_file))  # default_setup logs this file
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_main())
