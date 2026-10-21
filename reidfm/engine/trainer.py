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

    # ------------------------------------------------------------------ eval

    @classmethod
    def test(cls, cfg, model):
        if not comm.is_main_process():
            return {}
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
            extra.insert(0, MeanMetricCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD,
                                                   self._resume_best_metric))
            extra.append(CheckpointPruneHook(self.cfg.OUTPUT_DIR, self.cfg.REIDFM.KEEP_CHECKPOINTS))
        # Checkpointing/time guard must run before the writers (last hook on the main process).
        insert_at = len(ret) - 1 if comm.is_main_process() else len(ret)
        return ret[:insert_at] + extra + ret[insert_at:]

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_epoch = checkpoint.get("epoch", -1) + 1
            best = float(checkpoint.get("best_metric", checkpoint.get("metric", -1.0)))
            for hook in self._hooks:
                if isinstance(hook, MeanMetricCheckpointer):
                    hook.best_metric = best
            logger.info("Resumed at epoch %d (best mean metric so far %.2f)", self.start_epoch, best)

    def _emergency_checkpoint(self) -> None:
        if not comm.is_main_process():
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
        self.iter = self.start_iter = start_epoch * iters_per_epoch
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.epoch in range(start_epoch, max_epoch):
                    self.before_epoch()
                    for _ in range(iters_per_epoch):
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
