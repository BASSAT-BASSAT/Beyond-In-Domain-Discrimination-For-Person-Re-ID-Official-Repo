"""Training hooks for Kaggle-style sessions: best-by-mean checkpointing, time guard, pruning."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import torch
from fastreid.engine.hooks import HookBase, PeriodicCheckpointer
from fastreid.utils import comm
from fastreid.utils.events import get_event_storage

logger = logging.getLogger(__name__)

_PROCESS_START = time.time()
_EPOCH_CKPT = re.compile(r"^model_(\d{4})\.pth$")


def session_start() -> float:
    """Wall-clock start of the session: ``$REIDFM_SESSION_START`` (unix seconds) or process start."""
    value = os.environ.get("REIDFM_SESSION_START")
    try:
        return float(value) if value else _PROCESS_START
    except ValueError:
        return _PROCESS_START


class MeanMetricCheckpointer(PeriodicCheckpointer):
    """FastReID's periodic checkpointer, but ``model_best`` tracks ``mean/metric`` over all
    validation sets, and the best value survives resume (stored in every checkpoint)."""

    metric_name = "mean/metric"

    def __init__(self, checkpointer, period: int, best_metric: float = -1.0):
        super().__init__(checkpointer, period)
        self.best_metric = float(best_metric)

    def before_train(self):
        self.max_epoch = self.trainer.max_epoch

    def _metric(self) -> float:
        latest = get_event_storage().latest()
        return float(latest[self.metric_name][0]) if self.metric_name in latest else -1.0

    def state(self, epoch: int) -> dict:
        return {"epoch": int(epoch), "metric": self._metric(), "best_metric": self.best_metric}

    def after_epoch(self):
        epoch = int(self.trainer.epoch)
        state = self.state(epoch)
        periodic = (epoch + 1) % self.period == 0 and epoch < self.max_epoch - 1
        final = epoch >= self.max_epoch - 1
        if (periodic or final) and state["metric"] > self.best_metric:
            self.best_metric = state["metric"]
            state["best_metric"] = self.best_metric
            self.checkpointer.save("model_best", **state)
        if periodic:
            self.checkpointer.save(f"model_{epoch:04d}", **state)
            self.checkpointer.save("model_last", **state)
        if final:
            self.checkpointer.save("model_final", **state)
            self.checkpointer.save("model_last", **state)
        self.trainer.saved_epoch = epoch if (periodic or final) else getattr(self.trainer, "saved_epoch", -1)

    def after_train(self):
        # FastReID's EvalHook evaluates the final epoch in after_train (after model_final was
        # saved) when MAX_EPOCH is not a multiple of EVAL_PERIOD; promote it if it is the best.
        state = self.state(int(self.trainer.epoch))
        if state["metric"] <= self.best_metric:
            return
        last_file = Path(self.checkpointer.save_dir) / "last_checkpoint"
        last = last_file.read_text().strip() if last_file.is_file() else ""
        self.best_metric = state["best_metric"] = state["metric"]
        self.checkpointer.save("model_best", **state)
        if last:
            self.checkpointer.tag_last_checkpoint(last)  # resume must not start from model_best


class TimeGuardHook(HookBase):
    """Stop at an epoch boundary before the session budget runs out, with a checkpoint.

    After each epoch it predicts whether one more epoch (1.25x the slowest seen) fits in
    ``budget_h``; if not, it makes sure this epoch is checkpointed and asks the trainer to
    stop. Rank 0 decides and broadcasts, so all DDP ranks leave the loop together.
    """

    def __init__(self, budget_h: float, checkpointer=None, safety: float = 1.25):
        self.budget_s = float(budget_h) * 3600.0
        self.checkpointer = checkpointer
        self.safety = safety
        self._epoch_start = None
        self._slowest = 0.0

    def before_epoch(self):
        self._epoch_start = time.time()

    def _decide(self) -> bool:
        now = time.time()
        self._slowest = max(self._slowest, now - (self._epoch_start or now))
        remaining = self.budget_s - (now - session_start())
        stop = remaining < self._slowest * self.safety
        if comm.get_world_size() > 1:
            flag = torch.tensor([1 if stop else 0], device=torch.device("cuda", comm.get_local_rank()))
            torch.distributed.broadcast(flag, src=0)
            stop = bool(flag.item())
        return stop

    def after_epoch(self):
        if self.budget_s <= 0:
            return
        epoch = int(self.trainer.epoch)
        if epoch >= self.trainer.max_epoch - 1 or not self._decide():
            return
        if getattr(self.trainer, "saved_epoch", -1) != epoch and self.checkpointer is not None:
            state = {"epoch": epoch, "metric": -1.0, "best_metric": -1.0}
            ckpt_hook = next((h for h in self.trainer._hooks if isinstance(h, MeanMetricCheckpointer)), None)
            if ckpt_hook is not None:
                state = ckpt_hook.state(epoch)
            self.checkpointer.save(f"model_{epoch:04d}", **state)
            self.checkpointer.save("model_last", **state)
        logger.warning(
            "Time guard: stopping after epoch %d (%.2f h used of %.2f h). Resume with --resume.",
            epoch, (time.time() - session_start()) / 3600.0, self.budget_s / 3600.0,
        )
        self.trainer.stop_requested = True


class CheckpointPruneHook(HookBase):
    """Keep only the newest ``keep`` ``model_XXXX.pth`` files (Kaggle /kaggle/working is 20 GB)."""

    def __init__(self, output_dir: str, keep: int):
        self.output_dir = Path(output_dir)
        self.keep = int(keep)

    def after_epoch(self):
        if self.keep <= 0 or not comm.is_main_process():
            return
        last_file = self.output_dir / "last_checkpoint"
        protected = last_file.read_text().strip() if last_file.is_file() else ""
        epoch_ckpts = sorted(
            (p for p in self.output_dir.glob("model_*.pth") if _EPOCH_CKPT.match(p.name)),
            key=lambda p: int(_EPOCH_CKPT.match(p.name).group(1)),
        )
        for path in epoch_ckpts[: -self.keep]:
            if path.name != protected:
                path.unlink(missing_ok=True)
                logger.info("pruned %s", path.name)
