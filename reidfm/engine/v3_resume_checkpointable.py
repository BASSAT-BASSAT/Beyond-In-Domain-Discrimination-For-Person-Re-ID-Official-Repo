"""V3 checkpointable for a single-rank *prototype* mid-epoch save.

This is compatible with FastReID Checkpointer's state_dict/load_state_dict
interface. It encodes NumPy RNG as primitive values/tensors for safe
``torch.load(weights_only=True)``. Loading does NOT immediately reset RNG;
the caller must fully replay/skips batches BEFORE restoring RNG.

Not automatically installed into ReidfmTrainer. Multi-rank DDP, persistent
worker randomness, 4-worker/1000-batch replay, AMP, and atomic publication are
still unverified. DO NOT enable live midpoint resume based on this module.
"""

from __future__ import annotations

import numpy as np
import torch

from reidfm.engine.v3_mid_epoch_checkpoint import (
    mid_epoch_state, resolve_v3_resume_cursor,
)
from reidfm.engine.v3_resume_state import (
    build_resume_state_payload, loader_seed_signature,
    restore_rank_rng_state, validate_resume_state_payload,
)

SCHEMA = 1


def _encode_numpy_rng(payload):
    """Remove the NumPy ndarray/object from the serialized RNG payload."""
    state = dict(payload['process_rng_state'])
    kind, keys, position, has_gauss, cached_gaussian = state.pop('numpy')
    if kind != 'MT19937' or not isinstance(keys, np.ndarray) or keys.dtype != np.uint32:
        raise ValueError('Unsupported NumPy RNG format')
    state['numpy_mt19937'] = {
        'kind': kind,
        'keys': torch.from_numpy(keys.copy()),
        'position': int(position),
        'has_gauss': int(has_gauss),
        'cached_gaussian': float(cached_gaussian),
    }
    result = dict(payload)
    result['process_rng_state'] = state
    return result


def _decode_numpy_rng(payload):
    result = dict(payload)
    state = dict(result['process_rng_state'])
    encoded = state.pop('numpy_mt19937')
    if not isinstance(encoded, dict) or set(encoded) != {
        'kind', 'keys', 'position', 'has_gauss', 'cached_gaussian'
    }:
        raise ValueError('Invalid encoded NumPy RNG')
    keys = encoded['keys']
    if encoded['kind'] != 'MT19937' or not isinstance(keys, torch.Tensor):
        raise ValueError('Unsupported encoded NumPy RNG')
    if keys.dtype != torch.uint32 or keys.ndim != 1 or keys.numel() != 624:
        raise ValueError('Invalid MT19937 state tensor')
    position = encoded['position']
    has_gauss = encoded['has_gauss']
    cached = encoded['cached_gaussian']
    if type(position) is not int or not 0 <= position <= 624:
        raise ValueError('Invalid MT19937 position')
    if type(has_gauss) is not int or has_gauss not in (0, 1):
        raise ValueError('Invalid MT19937 Gaussian flag')
    if type(cached) is not float or not np.isfinite(cached):
        raise ValueError('Invalid MT19937 Gaussian cache')
    state['numpy'] = (
        'MT19937', keys.cpu().numpy().copy(), position, has_gauss, cached
    )
    result['process_rng_state'] = state
    return result


class V3ResumeCheckpointable:
    """Single-rank checkpoint metadata prototype; inert until explicitly staged."""

    def __init__(self, loader, iters_per_epoch: int, max_epoch: int):
        self.loader = loader
        self.signature = loader_seed_signature(loader)
        if self.signature['world_size'] != 1:
            raise RuntimeError('Prototype is single-rank only; DDP RNG gather is pending')
        if type(iters_per_epoch) is not int or iters_per_epoch <= 0 or iters_per_epoch % 2:
            raise ValueError('Invalid per-epoch iteration count')
        if type(max_epoch) is not int or max_epoch <= 0:
            raise ValueError('Invalid max_epoch')
        self.iters_per_epoch = iters_per_epoch
        self.max_epoch = max_epoch
        self._staged_cursor = None
        self._loaded_cursor = None
        self._pending_rng = None

    def stage_midpoint(self, epoch_index: int, best_metric: float):
        cursor = mid_epoch_state(epoch_index, self.iters_per_epoch, best_metric)
        resolve_v3_resume_cursor(cursor, self.iters_per_epoch, self.max_epoch)
        self._staged_cursor = cursor
        return dict(cursor)

    def state_dict(self):
        if self._staged_cursor is None:
            raise RuntimeError('Midpoint cursor not staged; refusing to save stale metadata')
        if loader_seed_signature(self.loader) != self.signature:
            raise ValueError('Loader drift before checkpoint')
        return {
            'v3_checkpointable_schema': SCHEMA,
            'cursor': dict(self._staged_cursor),
            'resume': _encode_numpy_rng(
                build_resume_state_payload(self.loader, include_cuda=False)
            ),
        }

    def load_state_dict(self, data):
        if not isinstance(data, dict) or data.get('v3_checkpointable_schema') != SCHEMA:
            raise ValueError('Unsupported V3 checkpointable schema')
        cursor = data.get('cursor')
        resolve_v3_resume_cursor(cursor, self.iters_per_epoch, self.max_epoch)
        resume = _decode_numpy_rng(data['resume'])
        state = validate_resume_state_payload(resume, self.loader)
        if loader_seed_signature(self.loader) != self.signature:
            raise ValueError('Loader drift on checkpoint load')
        self._loaded_cursor = dict(cursor)
        self._pending_rng = state
        self._staged_cursor = None  # A loaded midpoint is NOT a fresh save cursor.

    def resume_cursor(self):
        if self._loaded_cursor is None:
            raise RuntimeError('No loaded V3 midpoint cursor')
        return resolve_v3_resume_cursor(
            self._loaded_cursor, self.iters_per_epoch, self.max_epoch
        )

    def restore_rng_after_replay(self):
        if self._pending_rng is None:
            raise RuntimeError('No pending RNG state; resume/replay is incomplete')
        restore_rank_rng_state(self._pending_rng)
        self._pending_rng = None

