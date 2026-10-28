"""``ReidfmTrainer``: FastReID's ``DefaultTrainer`` with multi-source data and Kaggle-safe hooks.

Everything else (model, losses, optimizer, LR schedule, AMP, DDP, checkpoint format) is
FastReID's own SBS machinery. What changes:

* train set = all pool sources (``DATASETS.NAMES``), optional fixed virtual epoch length
  (``REIDFM.ITERS_PER_EPOCH``), and ``DomainBalancedIdentitySampler``;
* ``test`` = the benchmark-protocol evaluator on the pool's val sets (main process only);
* checkpointing keys ``model_best`` on the mean metric, a time guard stops cleanly before
  the session budget, and old epoch checkpoints are pruned.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from fastreid.data.build import _root, _train_loader_from_config, build_reid_train_loader
from fastreid.data.common import CommDataset
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.engine import DefaultTrainer, hooks
from fastreid.utils import comm
from fastreid.utils.events import EventStorage

from reidfm.config import resolve_pool
from reidfm.data.registry import spec_for_fastreid_name
from reidfm.data.split_filter import load_train_split, train_pid_allowed
from reidfm.data.samplers import DomainBalancedIdentitySampler
from reidfm.engine.hooks import CheckpointPruneHook, MeanMetricCheckpointer, TimeGuardHook

_EMERGENCY_CKPT = "model_last"

logger = logging.getLogger(__name__)


class PoolTrainSet(CommDataset):
    """CommDataset whose ``len`` can be a fixed virtual epoch (``iters * batch``).

    FastReID derives ``iters_per_epoch = len(dataset) // IMS_PER_BATCH``; the sampler is
    infinite, so the reported length only sets the epoch granularity.
    """

    def __init__(self, img_items, transform=None, relabel=True, virtual_len: int = 0):
        super().__init__(img_items, transform, relabel)
        self.virtual_len = int(virtual_len)

    def __len__(self):
        return self.virtual_len or len(self.img_items)


class ReidfmTrainer(DefaultTrainer):
    def __init__(self, cfg):
        self.stop_requested = False
        self.saved_epoch = -1
        self._resume_best_metric = -1.0
        self._v3_resume_offset = 0
        self._v3_resume_pending = None
        super().__init__(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        # FastReID's ContiguousParams relies on zero_grad() zeroing grads in place; torch >= 2.0
        # sets them to None, which leaves the flat buffer without grads (AMP then asserts).
        from fastreid.solver import build_optimizer

        return build_optimizer(cfg, model, contiguous=False)

    # ------------------------------------------------------------------ data

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.REIDFM.DATA_FORMAT == "REIDPACK3":
            return cls._build_v3_train_loader(cfg)

        allowed = load_train_split(cfg.REIDFM.TRAIN_SPLIT)
        items = []
        for name in cfg.DATASETS.NAMES:
            spec = spec_for_fastreid_name(name)
            data = DATASET_REGISTRY.get(name)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
            if comm.is_main_process():
                data.show_train()
            for path, pid, cam in data.train:
                raw_pid = pid.split("_", 1)[-1] if pid.startswith(f"{spec.key}_") else pid
                if train_pid_allowed(spec.key, raw_pid, allowed):
                    items.append((path, pid, cam))
        if not items:
            raise RuntimeError(f"no training images in {cfg.DATASETS.NAMES}")

        from fastreid.data.transforms import build_transforms

        iters = int(cfg.REIDFM.ITERS_PER_EPOCH)
        train_set = PoolTrainSet(items, build_transforms(cfg, is_train=True), relabel=True,
                                 virtual_len=iters * cfg.SOLVER.IMS_PER_BATCH if iters > 0 else 0)

        sampler = None
        if cfg.DATALOADER.SAMPLER_TRAIN == "DomainBalancedIdentitySampler":
            pool = resolve_pool(cfg)
            sampler = DomainBalancedIdentitySampler(
                train_set.img_items,
                mini_batch_size=cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
                num_instances=cfg.DATALOADER.NUM_INSTANCE,
                weight_mode=pool.weight_mode,
                multipliers=pool.domain_weights,
            )
            if comm.is_main_process():
                weights = ", ".join(f"{d}={w:.3f}" for d, w in sampler.weights.items())
                logger.info("Domain sampling weights: %s", weights)
        kwargs = _train_loader_from_config(cfg, train_set=train_set, sampler=sampler)
        return build_reid_train_loader(**kwargs)


    @classmethod
    def _build_v3_train_loader(cls, cfg):
        """Build packed Global P3/P5 batches with FastReID."""
        import math
        from pathlib import Path

        from fastreid.data.transforms import build_transforms
        from reidfm.config import parse_domain_weights
        from reidfm.data.v3_fastreid_dataset import (
            V3FastReIDTrainDataset,
        )

        manifest = Path(cfg.REIDFM.V3_TRAIN_MANIFEST)
        pack_root = Path(cfg.REIDFM.V3_PACK_ROOT)

        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        if not pack_root.is_dir():
            raise FileNotFoundError(pack_root)

        if cfg.DATALOADER.SAMPLER_TRAIN != (
            "DomainBalancedIdentitySampler"
        ):
            raise ValueError(
                "V3 currently requires "
                "DomainBalancedIdentitySampler"
            )

        iters = int(cfg.REIDFM.ITERS_PER_EPOCH)

        # Reuse the canonical configuration parser.
        multipliers = parse_domain_weights(
            cfg.REIDFM.DOMAIN_WEIGHTS
        )

        for domain, value in multipliers.items():
            if (
                not domain
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Invalid V3 domain multiplier: "
                    f"{domain}={value}"
                )

        train_set = V3FastReIDTrainDataset(
            manifest_path=manifest,
            pack_root=pack_root,
            transform=build_transforms(
                cfg, is_train=True
            ),
            virtual_len=(
                iters * cfg.SOLVER.IMS_PER_BATCH
                if iters > 0 else 0
            ),
        )

        # Detect misspelled or unavailable training domains.
        known_domains = {
            row["dataset_root_id"]
            for row in train_set.records
        }

        unknown_domains = sorted(
            set(multipliers) - known_domains
        )

        if unknown_domains:
            raise ValueError(
                "Unknown V3 domains in DOMAIN_WEIGHTS: "
                f"{unknown_domains}"
            )

        # V3: deterministic, study-seed-derived sampler AND worker RNG.
        # All DDP ranks share the global sampler seed; worker seeds vary by rank.
        import torch

        study_seed = int(cfg.REIDFM.SEED)
        if study_seed < 0:
            raise ValueError("V3 requires an explicit nonnegative REIDFM.SEED")
        v3_sampler_seed = (study_seed + 1) * 1000003 + 11
        v3_worker_seed = (study_seed + 1) * 1000003 + 97 + 1009 * comm.get_rank()

        sampler = DomainBalancedIdentitySampler(
            train_set.img_items,
            seed=v3_sampler_seed,
            mini_batch_size=(
                cfg.SOLVER.IMS_PER_BATCH
                // comm.get_world_size()
            ),
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            weight_mode=(
                cfg.REIDFM.SAMPLER_WEIGHT_MODE
                or "sqrt_ids"
            ),
            multipliers=multipliers,
        )

        if comm.is_main_process():
            logger.info(
                "V3 training: %d images, %d classes",
                len(train_set.records),
                len(sampler.pid_index),
            )
            logger.info(
                "V3 domain weights: %s",
                sampler.weights,
            )

        kwargs = _train_loader_from_config(
            cfg,
            train_set=train_set,
            sampler=sampler,
        )

        loader = build_reid_train_loader(**kwargs)
        # DefaultTrainer creates SimpleTrainer/AMPTrainer after this returns.
        # Hence no DataLoader iterator/workers exist before generator is set.
        loader.generator = torch.Generator().manual_seed(v3_worker_seed)
        loader._reidfm_v3_sampler_seed = v3_sampler_seed
        loader._reidfm_v3_worker_seed = v3_worker_seed
        loader._reidfm_v3_study_seed = study_seed
        loader._reidfm_v3_world_size = int(comm.get_world_size())
        loader._reidfm_v3_workers = int(cfg.DATALOADER.NUM_WORKERS)
        return loader

    # ------------------------------------------------------------------ eval

    @classmethod
    def test(cls, cfg, model):
        if not comm.is_main_process():
            return {}
        # V3 dispatch. Keep the Legacy evaluation body below unchanged.
        data_format = str(getattr(cfg.REIDFM, "DATA_FORMAT", "LEGACY"))
        if data_format == "REIDPACK3":
            from pathlib import Path
            from reidfm.data.v3_global_dataset import V3GlobalDataset
            from reidfm.evaluation.v3_evaluator import evaluate_v3_packed_groups

            if int(cfg.REIDFM.VAL_MAX_QUERIES) != 0:
                raise ValueError("V3 evaluation requires all queries (VAL_MAX_QUERIES=0)")
            if bool(cfg.REIDFM.EVAL_AMP):
                raise ValueError("V3 packed extractor currently requires EVAL_AMP=False")

            manifest = Path(cfg.REIDFM.V3_VAL_MANIFEST)
            pack_root = Path(cfg.REIDFM.V3_PACK_ROOT)
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            if not pack_root.is_dir():
                raise FileNotFoundError(pack_root)

            net = model.module if hasattr(model, "module") else model
            parameter = next(net.parameters(), None)
            buffer = next(net.buffers(), None)
            device = (parameter.device if parameter is not None
                      else buffer.device if buffer is not None else "cpu")
            with V3GlobalDataset(manifest, pack_root, split="validation") as dataset:
                results = evaluate_v3_packed_groups(
                    net, dataset.iter_records(), pack_root,
                    size_hw=tuple(cfg.INPUT.SIZE_TEST),
                    batch_size=int(cfg.TEST.IMS_PER_BATCH),
                    num_workers=int(cfg.REIDFM.EVAL_WORKERS),
                    device=device,
                )
            # FastReID EvalHook requires numeric-only nested results.
            # Preserve protocol metadata inside the V3 evaluator/report, but
            # never publish strings (e.g. positive_rule) as scalar metrics.
            from math import isfinite

            numeric_results = OrderedDict()
            for key, metrics in results.items():
                r1 = float(metrics["Rank-1"])
                map_score = float(metrics["mAP"])
                minp = float(metrics["mINP"])
                combined = float(metrics["metric"])
                if not all(isfinite(v) for v in (r1, map_score, minp, combined)):
                    raise ValueError(f"Non-finite V3 evaluation result: {key}")
                if abs(combined - (r1 + map_score) / 2.0) > 1e-5:
                    raise ValueError(f"Inconsistent V3 combined metric: {key}")
                numeric_results[key] = {
                    "Rank-1": r1, "mAP": map_score, "mINP": minp,
                    "metric": combined,
                }
                logger.info("V3 val %s: R1 %.2f mAP %.2f mINP %.2f",
                            key, r1, map_score, minp)
            return numeric_results
        if data_format != "LEGACY":
            raise ValueError(f"Unsupported REIDFM.DATA_FORMAT: {data_format!r}")

        from reidfm.evaluation.evaluator import evaluate_sets
        from reidfm.evaluation.extractors import FastReIDExtractor

        net = model.module if hasattr(model, "module") else model
        extractor = FastReIDExtractor(
            net, cfg, batch_size=cfg.TEST.IMS_PER_BATCH, workers=cfg.REIDFM.EVAL_WORKERS, amp=cfg.REIDFM.EVAL_AMP
        )
        keys = [spec_for_fastreid_name(name).key for name in cfg.DATASETS.TESTS]
        raw = evaluate_sets(keys, extractor, num_frames=cfg.REIDFM.EVAL_NUM_FRAMES, max_queries=cfg.REIDFM.VAL_MAX_QUERIES)
        results = OrderedDict()
        for key, metrics in raw.items():
            r1, mAP = metrics["Rank-1"], metrics["mAP"]
            results[key] = {"Rank-1": r1, "mAP": mAP, "mINP": metrics["mINP"], "metric": (r1 + mAP) / 2.0}
            logger.info("val %-14s R1 %.2f  mAP %.2f  mINP %.2f", key, r1, mAP, metrics["mINP"])
        return results

    # ------------------------------------------------------------------ hooks / resume

    def build_hooks(self):
        ret = [h for h in super().build_hooks() if not isinstance(h, hooks.PeriodicCheckpointer)]
        extra = [TimeGuardHook(self.cfg.REIDFM.TIME_BUDGET_H, self.checkpointer, self.cfg.REIDFM.TIME_GUARD_SAFETY)]
        if comm.is_main_process():
            # V3: never attach stale validation metrics to new-epoch checkpoints.
            # Legacy keeps its original MeanMetricCheckpointer behavior.
            if str(getattr(self.cfg.REIDFM, "DATA_FORMAT", "LEGACY")) == "REIDPACK3":
                from reidfm.engine.v3_fresh_checkpoint import V3FreshMeanMetricCheckpointer
                checkpoint_hook_type = V3FreshMeanMetricCheckpointer
            else:
                checkpoint_hook_type = MeanMetricCheckpointer
            extra.insert(0, checkpoint_hook_type(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD,
                                                 self._resume_best_metric))
            extra.append(CheckpointPruneHook(self.cfg.OUTPUT_DIR, self.cfg.REIDFM.KEEP_CHECKPOINTS))
        # Checkpointing/time guard must run before the writers (last hook on the main process).
        insert_at = len(ret) - 1 if comm.is_main_process() else len(ret)
        combined = ret[:insert_at] + extra + ret[insert_at:]
        if str(getattr(self.cfg.REIDFM, "DATA_FORMAT", "LEGACY")) == "REIDPACK3":
            from reidfm.engine.v3_trainer_midresume import V3TrainerMidpointHook
            combined.insert(0, V3TrainerMidpointHook(
                self.checkpointer, self.cfg, self.iters_per_epoch,
            ))
        return combined

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            v3 = str(getattr(self.cfg.REIDFM, "DATA_FORMAT", "LEGACY")) == "REIDPACK3"
            if v3 and "v3_resume_schema" in checkpoint:
                from reidfm.engine.v3_trainer_midresume import prepare_v3_midpoint_resume
                pending = prepare_v3_midpoint_resume(self, checkpoint)
                self._v3_resume_pending = pending
                self.start_epoch, self._v3_resume_offset, _ = pending["cursor"]
            else:
                if v3 and "v3_multirank_bundle" in checkpoint:
                    raise ValueError("Unexpected V3 resume bundle outside midpoint cursor")
                self.start_epoch = checkpoint.get("epoch", -1) + 1
            best = float(checkpoint.get("best_metric", checkpoint.get("metric", -1.0)))
            for hook in self._hooks:
                if isinstance(hook, MeanMetricCheckpointer):
                    hook.best_metric = best
            logger.info("Resumed at epoch %d (best mean metric so far %.2f)", self.start_epoch, best)

    def _emergency_checkpoint(self) -> None:
        if not comm.is_main_process():
            return
        if str(getattr(self.cfg.REIDFM, "DATA_FORMAT", "LEGACY")) == "REIDPACK3":
            # A mid-epoch exception is NOT an end-of-epoch checkpoint.
            # Keep the last known-good midpoint/epoch checkpoint pointer.
            logger.warning("Skipping unsafe V3 emergency save; preserving last checkpoint")
            return
        ckpt_hook = next((h for h in getattr(self, "_hooks", []) if isinstance(h, MeanMetricCheckpointer)), None)
        epoch = int(getattr(self, "epoch", max(self.start_epoch - 1, 0)))
        state = ckpt_hook.state(epoch) if ckpt_hook is not None else {"epoch": epoch, "metric": -1.0, "best_metric": -1.0}
        if ckpt_hook is not None:
            state["best_metric"] = ckpt_hook.best_metric
        self.checkpointer.save(_EMERGENCY_CKPT, **state)
        logger.warning("Saved emergency checkpoint %s (epoch %d)", _EMERGENCY_CKPT, epoch)

    # ------------------------------------------------------------------ loop

    def train(self):
        """``TrainerBase.train`` plus a clean exit when ``stop_requested`` is set."""
        start_epoch, max_epoch, iters_per_epoch = self.start_epoch, self.max_epoch, self.iters_per_epoch
        logger.info("Starting training from epoch %d / %d (%d iters per epoch)", start_epoch, max_epoch, iters_per_epoch)
        resume_offset = int(getattr(self, "_v3_resume_offset", 0))
        if resume_offset not in (0, iters_per_epoch // 2):
            raise ValueError("Invalid Trainer partial-epoch resume offset")
        self.iter = self.start_iter = start_epoch * iters_per_epoch + resume_offset
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                if resume_offset:
                    from reidfm.engine.v3_trainer_midresume import replay_v3_midpoint_loader
                    replay_v3_midpoint_loader(self, self._v3_resume_pending, resume_offset)
                    self._v3_resume_pending = None
                for self.epoch in range(start_epoch, max_epoch):
                    self.before_epoch()
                    for _ in range(
                        resume_offset if self.epoch == start_epoch else 0, iters_per_epoch
                    ):
                        self.before_step()
                        self.run_step()
                        self.after_step()
                        self.iter += 1
                    self.after_epoch()
                    if self.stop_requested:
                        break
            except Exception:
                logger.exception("Exception during training:")
                try:
                    self._emergency_checkpoint()
                except Exception:
                    logger.exception("Could not write emergency checkpoint:")
                raise
            finally:
                self.after_train()
        return getattr(self, "_last_eval_results", None)
