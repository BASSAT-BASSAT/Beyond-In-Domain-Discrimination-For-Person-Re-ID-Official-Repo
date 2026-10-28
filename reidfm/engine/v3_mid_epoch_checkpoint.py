"""V3-only midpoint checkpoint hook and strict resume cursor metadata.

Not registered automatically: the Trainer integration must explicitly opt in.
The hook fires after a committed training step (at 50% of a virtual epoch).
It writes a single rotating model_mid checkpoint via the supplied checkpointer.

NOTE: This module alone does NOT make mid-epoch resume safe. The Trainer must
restore optimizer/scheduler/AMP state, sampler/data-loader order and RNG.
"""

from __future__ import annotations

import math

from fastreid.engine.hooks import HookBase
from fastreid.utils import comm

CURSOR_VERSION = 1


def _strict_int(value, field):
    if type(value) is not int:
        raise TypeError(f"{field} must be a Python int")
    return value


def mid_epoch_state(epoch_index, iters_per_epoch, best_metric):
    epoch = _strict_int(epoch_index, "epoch_index")
    length = _strict_int(iters_per_epoch, "iters_per_epoch")
    if epoch < 0 or length <= 0 or length % 2:
        raise ValueError("Midpoint checkpoint needs a nonnegative epoch and even length")
    metric = float(best_metric)
    if not math.isfinite(metric):
        raise ValueError("Non-finite best_metric")
    completed = length // 2
    return {
        "v3_resume_schema": CURSOR_VERSION,
        "epoch": epoch,
        "iteration_in_epoch": completed,
        "iters_per_epoch": length,
        "global_steps_completed": epoch * length + completed,
        "metric": -1.0,  # Midpoint has NO new evaluation metric.
        "best_metric": metric,
    }


def resolve_v3_resume_cursor(checkpoint, iters_per_epoch, max_epoch):
    """Return (start_epoch, completed_batches_in_epoch, completed_global_steps).

    Compatible with legacy *end-of-epoch* metadata without the V3 schema.
    Must be called only after a trusted checkpoint has been loaded.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dictionary")
    length = _strict_int(iters_per_epoch, "iters_per_epoch")
    limit = _strict_int(max_epoch, "max_epoch")
    if length <= 0 or limit <= 0:
        raise ValueError("Invalid training schedule")

    if "v3_resume_schema" not in checkpoint:
        e = _strict_int(checkpoint.get("epoch", -1), "epoch")
        if not -1 <= e < limit:
            raise ValueError("Invalid completed epoch in legacy-style checkpoint")
        next_epoch = e + 1
        return next_epoch, 0, next_epoch * length

    if checkpoint["v3_resume_schema"] != CURSOR_VERSION:
        raise ValueError("Unsupported V3 checkpoint cursor schema")
    e = _strict_int(checkpoint.get("epoch"), "epoch")
    completed = _strict_int(checkpoint.get("iteration_in_epoch"), "iteration_in_epoch")
    recorded_length = _strict_int(checkpoint.get("iters_per_epoch"), "iters_per_epoch")
    total = _strict_int(checkpoint.get("global_steps_completed"), "global_steps_completed")
    if not (0 <= e < limit and recorded_length == length and length % 2 == 0):
        raise ValueError("Epoch or per-epoch iteration count inconsistent")
    if completed != length // 2 or total != e * length + completed:
        raise ValueError("Invalid V3 midpoint cursor")
    return e, completed, total


class V3MidEpochCheckpointHook(HookBase):
    """One rotating midpoint save per epoch; full-epoch save stays with existing hook."""

    def __init__(self, checkpointer, iters_per_epoch, best_metric_hook):
        self.checkpointer = checkpointer
        self.iters_per_epoch = _strict_int(iters_per_epoch, "iters_per_epoch")
        if self.iters_per_epoch <= 0 or self.iters_per_epoch % 2:
            raise ValueError("Midpoint hook requires a positive even epoch length")
        self.best_metric_hook = best_metric_hook

    def after_step(self):
        epoch = int(self.trainer.epoch)
        iteration = int(self.trainer.iter)
        max_epoch = int(self.trainer.max_epoch)
        local_completed = iteration - epoch * self.iters_per_epoch + 1
        if not (0 <= epoch < max_epoch and 1 <= local_completed <= self.iters_per_epoch):
            raise ValueError("Trainer iteration/epoch contract mismatch")
        if local_completed != self.iters_per_epoch // 2 or not comm.is_main_process():
            return

        best = float(self.best_metric_hook.best_metric)
        state = mid_epoch_state(epoch, self.iters_per_epoch, best)
        self.checkpointer.save("model_mid", **state)

