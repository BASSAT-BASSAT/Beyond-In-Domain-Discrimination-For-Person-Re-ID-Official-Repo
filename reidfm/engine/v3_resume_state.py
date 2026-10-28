"""V3 training-loader identity and RNG metadata for future safe resume.

This module is NOT yet wired to ReidfmTrainer. A production checkpoint must
also restore registered optimizer/scheduler/AMP state, deal with DataLoader
iterator replay, and collect per-rank RNG state in distributed training.
"""

from __future__ import annotations

import random

import numpy as np
import torch

SCHEMA = 1
SEED_FIELDS = (
    "_reidfm_v3_study_seed", "_reidfm_v3_sampler_seed",
    "_reidfm_v3_worker_seed", "_reidfm_v3_world_size",
    "_reidfm_v3_workers",
)


def _strict_nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative Python int")
    return value


def loader_seed_signature(loader):
    values = [_strict_nonnegative(getattr(loader, key, None), key)
              for key in SEED_FIELDS]
    study, sampler_seed, worker_seed, world_size, workers = values
    if world_size < 1:
        raise ValueError("Invalid V3 world_size")
    if int(getattr(loader, "num_workers", -1)) != workers:
        raise ValueError("V3 num_workers differs from recorded configuration")
    generator = getattr(loader, "generator", None)
    if not isinstance(generator, torch.Generator):
        raise ValueError("Missing V3 torch DataLoader generator")
    if int(generator.initial_seed()) != worker_seed:
        raise ValueError("DataLoader generator seed changed")
    batch_sampler = getattr(loader, "batch_sampler", None)
    sampler = getattr(batch_sampler, "sampler", None)
    if int(getattr(sampler, "_seed", -1)) != sampler_seed:
        raise ValueError("Actual sampler seed differs from V3 signature")
    return {
        "study_seed": study,
        "sampler_seed": sampler_seed,
        "worker_seed": worker_seed,
        "world_size": world_size,
        "num_workers": workers,
    }


def capture_cpu_rng_state():
    """CPU/process RNG only; never initializes CUDA."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
    }


def restore_cpu_rng_state(state):
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu"}:
        raise ValueError("Invalid CPU RNG state keys")
    if not isinstance(state["torch_cpu"], torch.Tensor):
        raise TypeError("Invalid torch CPU RNG tensor")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])


def capture_rank_rng_state(include_cuda=True):
    result = capture_cpu_rng_state()
    if include_cuda and torch.cuda.is_initialized():
        result["torch_cuda_all"] = [x.clone() for x in torch.cuda.get_rng_state_all()]
    else:
        result["torch_cuda_all"] = []
    return result


def build_resume_state_payload(loader, include_cuda=True):
    return {
        "v3_resume_state_schema": SCHEMA,
        "loader_signature": loader_seed_signature(loader),
        "process_rng_state": capture_rank_rng_state(include_cuda=include_cuda),
    }


def validate_resume_state_payload(payload, loader):
    if not isinstance(payload, dict) or type(payload.get("v3_resume_state_schema")) is not int:
        raise ValueError("Missing V3 resume-state schema")
    if payload["v3_resume_state_schema"] != SCHEMA:
        raise ValueError("Unsupported V3 resume-state version")
    expected = loader_seed_signature(loader)
    actual = payload.get("loader_signature")
    if not isinstance(actual, dict) or actual != expected:
        raise ValueError("V3 loader signature mismatch on resume")
    state = payload.get("process_rng_state")
    if not isinstance(state, dict) or not {
        "python", "numpy", "torch_cpu", "torch_cuda_all"
    }.issubset(state):
        raise ValueError("Missing per-process RNG fields")
    if not isinstance(state["torch_cuda_all"], list):
        raise ValueError("Invalid CUDA RNG state list")
    return state


def restore_rank_rng_state(state):
    """Restore process RNG after loader replay (not before it).

    In DDP this must be applied separately on every rank using THAT rank's
    saved state; a rank-0-only checkpoint is insufficient for exact DDP resume.
    """
    cuda_states = state.get("torch_cuda_all", [])
    if cuda_states:
        if not torch.cuda.is_initialized():
            raise RuntimeError("Saved CUDA RNG cannot be restored without initialized CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs at resume")
    restore_cpu_rng_state({key: state[key] for key in ("python", "numpy", "torch_cpu")})
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)

