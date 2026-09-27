from __future__ import annotations

import itertools
from collections import Counter

import pytest

from reidfm.data.samplers import DomainBalancedIdentitySampler, domain_of, domain_weights, reorder_index


def _items(ids_per_domain: dict[str, int], per_id: int = 4):
    return [(f"{d}/{p}_{i}.jpg", f"{d}_{p}", f"{d}_0") for d, n in ids_per_domain.items() for p in range(n) for i in range(per_id)]


def test_domain_weights_modes():
    counts = {"big": 400, "small": 100}
    assert domain_weights(counts, "sqrt_ids")["big"] == pytest.approx(20 / 30)
    assert domain_weights(counts, "ids")["big"] == pytest.approx(0.8)
    assert domain_weights(counts, "uniform")["big"] == pytest.approx(0.5)
    assert domain_weights(counts, "uniform", {"small": 3})["small"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        domain_weights(counts, "nope")


def test_domain_of():
    assert domain_of("market1501_0002") == "market1501"
    assert domain_of("msmt17_test12") == "msmt17"


def test_reorder_index_interleaves_ranks():
    assert reorder_index([0, 1, 2, 3], 2) == [0, 2, 1, 3]


def test_sampler_batches_follow_weights_and_pk():
    items = _items({"big": 400, "small": 25})
    sampler = DomainBalancedIdentitySampler(items, mini_batch_size=32, num_instances=4, weight_mode="sqrt_ids",
                                            seed=0, rank=0, world_size=1)
    indices = list(itertools.islice(iter(sampler), 32 * 200))
    domains = Counter(domain_of(items[i][1]) for i in indices)
    expected = sampler.weights["small"]  # sqrt(25) / (sqrt(400) + sqrt(25)) = 0.2
    assert domains["small"] / len(indices) == pytest.approx(expected, abs=0.03)
    first = indices[:32]
    pids = Counter(items[i][1] for i in first)
    assert all(count % 4 == 0 for count in pids.values())


def test_sampler_ranks_are_disjoint_and_deterministic():
    items = _items({"a": 50, "b": 50})
    streams = [
        list(itertools.islice(iter(DomainBalancedIdentitySampler(items, 16, 4, seed=3, rank=r, world_size=2)), 64))
        for r in (0, 1)
    ]
    again = list(itertools.islice(iter(DomainBalancedIdentitySampler(items, 16, 4, seed=3, rank=0, world_size=2)), 64))
    assert streams[0] == again
    assert streams[0] != streams[1]


def test_sampler_rejects_bad_batch():
    with pytest.raises(ValueError):
        DomainBalancedIdentitySampler(_items({"a": 5}), mini_batch_size=30, num_instances=4, seed=0, rank=0, world_size=1)
