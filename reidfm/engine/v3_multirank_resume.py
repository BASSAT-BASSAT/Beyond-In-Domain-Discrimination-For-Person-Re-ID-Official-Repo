"""V3 rank-aware *metadata* for future single-/multi-GPU checkpoint resume.

This is a pure CPU-testable bundle contract. All participating DDP ranks must
produce their own record at the same optimizer-step boundary and collectively
gather them BEFORE rank 0 publishes a checkpoint. The production collective,
AMPTrainer connection, CUDA DataLoaderX replay, atomic checkpoint publishing,
and Hugging Face upload are NOT implemented here.

Saving model/optimizer/scheduler weights remains FastReID Checkpointer's job.
RNG and per-rank GradScaler metadata must be restored on the correct rank,
AFTER recreating and fast-forwarding that rank's DataLoader iterator.
"""

from __future__ import annotations

import copy

import torch

from reidfm.engine.v3_mid_epoch_checkpoint import (
    mid_epoch_state, resolve_v3_resume_cursor,
)
from reidfm.engine.v3_resume_state import (
    build_resume_state_payload, loader_seed_signature,
    validate_resume_state_payload,
)
from reidfm.engine.v3_resume_checkpointable import (
    _decode_numpy_rng, _encode_numpy_rng,
)

SCHEMA = 1
RANK_KEYS = {
    'schema', 'rank', 'run_id', 'fingerprint', 'world_size', 'cursor',
    'resume', 'amp_enabled', 'amp_scaler_state', 'cuda_required',
}


def _number(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be int >= {minimum}')
    return value


def _label(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Invalid {label}')
    return value


def _expected_seeds(study_seed, rank):
    base = (_number(study_seed, 'study_seed') + 1) * 1000003
    return base + 11, base + 97 + 1009 * rank


def _verify_rank_identity(record):
    if not isinstance(record, dict) or set(record) != RANK_KEYS:
        raise ValueError('Invalid V3 rank-record keys')
    if record['schema'] != SCHEMA:
        raise ValueError('Unsupported V3 multirank schema')
    rank = _number(record['rank'], 'rank')
    world_size = _number(record['world_size'], 'world_size', minimum=1)
    if rank >= world_size:
        raise ValueError('Rank outside world_size')
    _label(record['run_id'], 'run_id')
    _label(record['fingerprint'], 'fingerprint')
    signature = record['resume'].get('loader_signature')
    if not isinstance(signature, dict):
        raise ValueError('Missing per-rank loader signature')
    shared, worker = _expected_seeds(signature.get('study_seed'), rank)
    if (signature.get('sampler_seed') != shared
            or signature.get('worker_seed') != worker
            or signature.get('world_size') != world_size):
        raise ValueError('Rank-specific sampler/worker seed contract violated')
    if type(signature.get('num_workers')) is not int or signature['num_workers'] < 0:
        raise ValueError('Invalid num_workers in rank signature')
    if not isinstance(record['amp_enabled'], bool):
        raise ValueError('Invalid AMP mode')
    if record['amp_enabled']:
        if not isinstance(record['amp_scaler_state'], dict) or not record['amp_scaler_state']:
            raise ValueError('AMP rank must provide a non-empty GradScaler state')
    elif record['amp_scaler_state'] is not None:
        raise ValueError('AMP disabled but scaler state was supplied')
    if not isinstance(record['cuda_required'], bool):
        raise ValueError('Invalid CUDA requirement')
    if record['cuda_required'] and not record['resume']['process_rng_state']['torch_cuda_all']:
        raise ValueError('Missing required per-rank CUDA RNG')
    return rank, world_size


def capture_v3_rank_record(
    loader, *, rank, epoch_index, best_metric, max_epoch,
    run_id, fingerprint, amp_scaler=None, cuda_required=False,
):
    rank = _number(rank, 'rank')
    max_epoch = _number(max_epoch, 'max_epoch', minimum=1)
    signature = loader_seed_signature(loader)
    world_size = signature['world_size']
    if rank >= world_size:
        raise ValueError('Rank outside configured DDP world_size')
    expected_shared, expected_worker = _expected_seeds(signature['study_seed'], rank)
    if (signature['sampler_seed'] != expected_shared
            or signature['worker_seed'] != expected_worker):
        raise ValueError('Actual loader seeds are inconsistent with DDP rank')
    if cuda_required and not torch.cuda.is_initialized():
        raise RuntimeError('CUDA RNG required, but CUDA was never initialized')
    amp_enabled = amp_scaler is not None
    scaler_state = None
    if amp_enabled:
        scaler_state = amp_scaler.state_dict()
        if not isinstance(scaler_state, dict) or not scaler_state:
            raise ValueError('GradScaler state_dict is missing or empty')

    cursor = mid_epoch_state(epoch_index, 2000, best_metric)
    resolve_v3_resume_cursor(cursor, 2000, max_epoch)
    payload = _encode_numpy_rng(build_resume_state_payload(
        loader, include_cuda=cuda_required,
    ))
    record = {
        'schema': SCHEMA,
        'rank': rank,
        'run_id': _label(run_id, 'run_id'),
        'fingerprint': _label(fingerprint, 'fingerprint'),
        'world_size': world_size,
        'cursor': cursor,
        'resume': payload,
        'amp_enabled': amp_enabled,
        'amp_scaler_state': copy.deepcopy(scaler_state),
        'cuda_required': bool(cuda_required),
    }
    _verify_rank_identity(record)
    return record


def assemble_v3_rank_bundle(records, *, max_epoch=60, iters_per_epoch=2000):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('At least one rank record required')
    if iters_per_epoch != 2000:
        raise ValueError('Approved V3 schedule requires 2000 iterations/epoch')
    by_rank = {}
    first = records[0]
    _, world_size = _verify_rank_identity(first)
    expected_run = first['run_id']
    expected_fp = first['fingerprint']
    expected_cursor = first['cursor']
    expected_amp = first['amp_enabled']
    expected_cuda = first['cuda_required']
    for record in records:
        rank, size = _verify_rank_identity(record)
        resolve_v3_resume_cursor(record['cursor'], iters_per_epoch, max_epoch)
        if rank in by_rank:
            raise ValueError('Duplicate DDP rank in checkpoint bundle')
        if (size != world_size or record['run_id'] != expected_run
                or record['fingerprint'] != expected_fp
                or record['cursor'] != expected_cursor
                or record['amp_enabled'] != expected_amp
                or record['cuda_required'] != expected_cuda):
            raise ValueError('Rank metadata differs across DDP processes')
        by_rank[rank] = copy.deepcopy(record)
    if set(by_rank) != set(range(world_size)):
        raise ValueError('Missing DDP rank state in checkpoint bundle')
    return {
        'v3_multirank_bundle_schema': SCHEMA,
        'run_id': expected_run,
        'fingerprint': expected_fp,
        'world_size': world_size,
        'iters_per_epoch': iters_per_epoch,
        'max_epoch': max_epoch,
        'cursor': copy.deepcopy(expected_cursor),
        'ranks': by_rank,
    }


def inspect_v3_local_resume(bundle, *, rank, loader, run_id, fingerprint,
                            amp_enabled, cuda_required):
    if not isinstance(bundle, dict) or bundle.get('v3_multirank_bundle_schema') != SCHEMA:
        raise ValueError('Invalid V3 resume bundle schema')
    # Reassemble all ranks to revalidate completeness, seeds and cross-rank agreement.
    revalidated = assemble_v3_rank_bundle(
        list(bundle.get('ranks', {}).values()),
        max_epoch=bundle.get('max_epoch'),
        iters_per_epoch=bundle.get('iters_per_epoch'),
    )
    if any(bundle.get(key) != revalidated[key] for key in (
        'run_id', 'fingerprint', 'world_size', 'max_epoch',
        'iters_per_epoch', 'cursor',
    )):
        raise ValueError('Bundle header differs from rank records')
    rank = _number(rank, 'rank')
    if (bundle['run_id'] != run_id or bundle['fingerprint'] != fingerprint
            or loader_seed_signature(loader)['world_size'] != bundle['world_size']):
        raise ValueError('Checkpoint run identity or world size changed')
    if rank not in revalidated['ranks']:
        raise ValueError('Requested rank missing from checkpoint')
    record = revalidated['ranks'][rank]
    if record['amp_enabled'] is not amp_enabled or record['cuda_required'] is not cuda_required:
        raise ValueError('AMP/CUDA mode differs at resume')
    payload = _decode_numpy_rng(record['resume'])
    pending_rng = validate_resume_state_payload(payload, loader)
    # DO NOT apply RNG here. DataLoader replay must finish first.
    return {
        'cursor': resolve_v3_resume_cursor(
            record['cursor'], bundle['iters_per_epoch'], bundle['max_epoch']
        ),
        'pending_rng_after_replay': pending_rng,
        'pending_amp_scaler_state': copy.deepcopy(record['amp_scaler_state']),
    }

