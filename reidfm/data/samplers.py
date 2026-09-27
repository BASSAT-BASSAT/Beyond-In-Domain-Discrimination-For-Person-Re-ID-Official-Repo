"""Domain-balanced P x K identity sampler for multi-source training.

FastReID's ``NaiveIdentitySampler`` draws identities uniformly, so a big source (LaST,
LPW, MOT) would dominate every batch. Here each identity slot of a batch first draws a
domain with probability proportional to its weight, then takes the next identity from
that domain's shuffled queue, then ``K`` of its images.

Domain = FastReID train pid prefix (``"<dataset>_<pid>"`` -> ``<dataset>``).
Weights: ``sqrt_ids`` (default), ``ids`` (= uniform over identities), or ``uniform``
(= equal per domain), times optional per-domain multipliers.

Distributed behaviour matches FastReID: every rank runs the same seeded stream, and
``reorder_index`` + rank ``islice`` hands each rank whole identities.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data.sampler import Sampler


def _comm():
    try:
        from fastreid.utils import comm

        return comm
    except ImportError:
        return None


def domain_of(pid: str) -> str:
    return str(pid).split("_", 1)[0]


def domain_weights(
    ids_per_domain: dict[str, int],
    mode: str = "sqrt_ids",
    multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    multipliers = multipliers or {}
    raw = {}
    for domain, count in ids_per_domain.items():
        base = {"sqrt_ids": math.sqrt(count), "ids": float(count), "uniform": 1.0}.get(mode)
        if base is None:
            raise ValueError(f"unknown weight mode {mode!r}")
        raw[domain] = base * float(multipliers.get(domain, 1.0))
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("all domain weights are zero")
    return {domain: value / total for domain, value in raw.items()}


def reorder_index(batch_indices: list[int], world_size: int) -> list[int]:
    mini = len(batch_indices) // world_size
    return [batch_indices[i + j * mini] for i in range(mini) for j in range(world_size)]


class DomainBalancedIdentitySampler(Sampler):
    def __init__(
        self,
        data_source: Sequence,
        mini_batch_size: int,
        num_instances: int,
        weight_mode: str = "sqrt_ids",
        multipliers: dict[str, float] | None = None,
        seed: int | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        if mini_batch_size % num_instances:
            raise ValueError(f"mini batch {mini_batch_size} must be a multiple of NUM_INSTANCE {num_instances}")
        comm = _comm()
        self._rank = rank if rank is not None else (comm.get_rank() if comm else 0)
        self._world_size = world_size if world_size is not None else (comm.get_world_size() if comm else 1)
        self.num_instances = num_instances
        self.batch_size = mini_batch_size * self._world_size
        self.num_pids_per_batch = self.batch_size // num_instances

        self.pid_index: dict[str, list[int]] = defaultdict(list)
        for index, info in enumerate(data_source):
            self.pid_index[info[1]].append(index)
        self.domain_pids: dict[str, list[str]] = defaultdict(list)
        for pid in sorted(self.pid_index):
            self.domain_pids[domain_of(pid)].append(pid)
        self.domains = sorted(self.domain_pids)
        self.weights = domain_weights({d: len(p) for d, p in self.domain_pids.items()}, weight_mode, multipliers)
        self._probs = np.array([self.weights[d] for d in self.domains], dtype=np.float64)

        if seed is None:
            seed = comm.shared_random_seed() if comm else 0
        self._seed = int(seed)

    def __iter__(self) -> Iterator[int]:
        yield from itertools.islice(self._infinite_indices(), self._rank, None, self._world_size)

    def _infinite_indices(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed)
        queues: dict[str, list[str]] = {d: [] for d in self.domains}
        pools: dict[str, list[int]] = {}

        def next_pid(domain: str, taken: set[str]) -> str:
            # Avoid repeating an identity inside one batch unless the domain is too small.
            pids = self.domain_pids[domain]
            pid = pids[0]
            for _ in range(min(len(pids), 64)):
                if not queues[domain]:
                    queues[domain] = [pids[i] for i in rng.permutation(len(pids))]
                pid = queues[domain].pop()
                if pid not in taken:
                    return pid
            return pid

        def take_instances(pid: str) -> list[int]:
            indices = self.pid_index[pid]
            if len(indices) < self.num_instances:
                return rng.choice(indices, size=self.num_instances, replace=True).tolist()
            pool = pools.get(pid)
            if not pool or len(pool) < self.num_instances:
                pool = [indices[i] for i in rng.permutation(len(indices))]
            chosen, pools[pid] = pool[: self.num_instances], pool[self.num_instances :]
            return chosen

        while True:
            slots = rng.choice(len(self.domains), size=self.num_pids_per_batch, p=self._probs)
            taken: set[str] = set()
            batch: list[int] = []
            for slot in slots:
                pid = next_pid(self.domains[slot], taken)
                taken.add(pid)
                batch.extend(take_instances(pid))
            yield from reorder_index(batch, self._world_size)


def register_samplers() -> None:
    """Expose the sampler under ``fastreid.data.samplers`` (the trainer builds it by name)."""
    from fastreid.data import samplers

    samplers.DomainBalancedIdentitySampler = DomainBalancedIdentitySampler
