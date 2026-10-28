"""V3-only trainer midpoint checkpoint/replay integration.

Two-rank midpoint snapshots are gathered on ALL ranks, but only rank 0 saves.
DataLoader cursor replay skips only input batches, never optimizer steps.
This module is not a scientific/runtime clearance: atomic publication, exact
CUDA DataLoaderX continuation, and HF remote recovery need live tests.
"""
from __future__ import annotations

import io
import math

import torch
import torch.distributed as dist
from fastreid.engine.hooks import HookBase
from fastreid.utils import comm

from reidfm.engine.hooks import MeanMetricCheckpointer
from reidfm.engine.v3_multirank_resume import (
    assemble_v3_rank_bundle, capture_v3_rank_record,
    inspect_v3_local_resume,
)
from reidfm.engine.v3_resume_state import restore_rank_rng_state
from reidfm.experiments.protocol import config_fingerprint


def v3_run_identity(cfg):
    if str(cfg.REIDFM.DATA_FORMAT) != 'REIDPACK3':
        raise ValueError('V3 run identity requires REIDPACK3')
    seed = int(cfg.REIDFM.SEED)
    if seed < 0:
        raise ValueError('V3 run identity requires nonnegative seed')
    condition = str(cfg.REIDFM.LOSS_STUDY_CONDITION or 'R0')
    return f'{condition}/seed_{seed}', config_fingerprint(cfg)


def _collect_bundle(local):
    world_size = int(local['world_size'])
    if world_size not in (1, 2):
        raise ValueError('V3 resume currently supports only 1 or 2 ranks')
    if world_size == 1:
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError('Rank/collective world-size mismatch')
        return assemble_v3_rank_bundle([local])
    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise RuntimeError('DDP midpoint snapshot requires live process group on ALL ranks')
    if dist.get_rank() != local['rank']:
        raise ValueError('DDP rank disagrees with loader metadata')

    # Step52: raw Tensor pickling inside object collectives fails on Kaggle.
    encoded = io.BytesIO()
    torch.save(local, encoded)
    packed = [None] * world_size
    dist.all_gather_object(packed, encoded.getvalue())
    records = [
        torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        for blob in packed
    ]
    return assemble_v3_rank_bundle(records)


def _fresh_best_metric(trainer):
    best = -1.0
    if comm.is_main_process():
        for hook in getattr(trainer, '_hooks', []):
            if isinstance(hook, MeanMetricCheckpointer):
                best = float(hook.best_metric)
                break
    if dist.is_initialized() and dist.get_world_size() > 1:
        device = (torch.device('cuda', torch.cuda.current_device())
                  if dist.get_backend() == 'nccl' else torch.device('cpu'))
        value = torch.tensor([best], dtype=torch.float64, device=device)
        dist.broadcast(value, src=0)
        best = float(value.item())
    if not math.isfinite(best):
        raise ValueError('Non-finite V3 best metric')
    return best


class V3TrainerMidpointHook(HookBase):
    """All-rank half-epoch state gather, rank-zero Checkpointer.save."""
    def __init__(self, checkpointer, cfg, iters_per_epoch):
        self.checkpointer = checkpointer
        self.iters_per_epoch = int(iters_per_epoch)
        if self.iters_per_epoch != 2000:
            raise ValueError('V3 midpoint requires 2000 iterations/epoch')
        self.run_id, self.fingerprint = v3_run_identity(cfg)
        self._last_saved_global = -1

    def after_step(self):
        trainer = self.trainer
        epoch = int(trainer.epoch)
        completed = int(trainer.iter) - epoch * self.iters_per_epoch + 1
        if not 1 <= completed <= self.iters_per_epoch:
            raise RuntimeError('Invalid Trainer epoch/iteration position')
        if completed != self.iters_per_epoch // 2:
            return
        current = epoch * self.iters_per_epoch + completed
        if current == self._last_saved_global:
            return

        best = _fresh_best_metric(trainer)
        inner = trainer._trainer
        local = capture_v3_rank_record(
            inner.data_loader, rank=comm.get_rank(), epoch_index=epoch,
            best_metric=best, max_epoch=int(trainer.max_epoch),
            run_id=self.run_id, fingerprint=self.fingerprint,
            amp_scaler=getattr(inner, 'grad_scaler', None),
            cuda_required=torch.cuda.is_initialized(),
        )
        bundle = _collect_bundle(local)
        assert bundle['cursor']['global_steps_completed'] == current

        save_error = None
        if comm.is_main_process():
            try:
                self.checkpointer.save(
                    'model_mid', **bundle['cursor'], v3_multirank_bundle=bundle,
                )
            except Exception as exc:
                save_error = f'{type(exc).__name__}: {exc}'
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Every rank observes success/failure before the next optimizer step.
            reply = [save_error]
            dist.broadcast_object_list(reply, src=0)
            save_error = reply[0]
        if save_error is not None:
            raise RuntimeError(f'V3 midpoint checkpoint failed: {save_error}')
        self._last_saved_global = current


def prepare_v3_midpoint_resume(trainer, checkpoint):
    if 'v3_multirank_bundle' not in checkpoint:
        raise ValueError('Unsafe V3 midpoint: missing all-rank resume bundle')
    inner = trainer._trainer
    run_id, fingerprint = v3_run_identity(trainer.cfg)
    return inspect_v3_local_resume(
        checkpoint['v3_multirank_bundle'], rank=comm.get_rank(),
        loader=inner.data_loader, run_id=run_id, fingerprint=fingerprint,
        amp_enabled=getattr(inner, 'grad_scaler', None) is not None,
        cuda_required=torch.cuda.is_initialized(),
    )


def replay_v3_midpoint_loader(trainer, pending, completed):
    if pending is None or pending['cursor'][1] != completed or completed != 1000:
        raise ValueError('Invalid V3 midpoint replay contract')
    inner = trainer._trainer
    if not hasattr(inner, '_data_loader_iter'):
        raise RuntimeError('Missing real FastReID DataLoader iterator')
    # No optimizer/model forward/backward: only input iterator is advanced.
    for _ in range(completed):
        next(inner._data_loader_iter)
    scaler_state = pending['pending_amp_scaler_state']
    if scaler_state is not None:
        scaler = getattr(inner, 'grad_scaler', None)
        if scaler is None:
            raise ValueError('Saved AMP state requires active AMPTrainer')
        scaler.load_state_dict(scaler_state)
    # DataLoader replay may draw random values; restore parent process RNG last.
    restore_rank_rng_state(pending['pending_rng_after_replay'])
