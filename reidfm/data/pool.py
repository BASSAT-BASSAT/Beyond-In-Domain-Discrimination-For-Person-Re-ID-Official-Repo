"""Pool configs (``configs/datasets/*.yaml``): which datasets train, validate, test, and stay OOD."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from reidfm.data.registry import get_dataset_spec
from reidfm.utils.config import load_yaml
from reidfm.utils.paths import repo_path

WEIGHT_MODES = ("sqrt_ids", "ids", "uniform")


@dataclass
class PoolConfig:
    name: str
    sources: list[str]
    ood: list[str]
    val_sets: list[str]
    test_sets: list[str]
    weight_mode: str = "sqrt_ids"
    domain_weights: dict[str, float] = field(default_factory=dict)
    converters: dict[str, dict] = field(default_factory=dict)
    path: Path | None = None

    @property
    def all_datasets(self) -> list[str]:
        ordered: list[str] = []
        for key in [*self.sources, *self.test_sets, *self.ood]:
            if key not in ordered:
                ordered.append(key)
        return ordered

    def fastreid_names(self, keys: list[str]) -> list[str]:
        return [get_dataset_spec(key).fastreid_name for key in keys]

    def converter_options(self, key: str) -> dict:
        options = dict(get_dataset_spec(key).default_options)
        options.update(self.converters.get(key) or {})
        return options

    def validate(self) -> None:
        for key in self.all_datasets + self.val_sets:
            get_dataset_spec(key)
        leaked = sorted(set(self.sources) & set(self.ood))
        if leaked:
            raise ValueError(f"pool {self.name!r}: OOD datasets must never be training sources: {leaked}")
        no_train = [key for key in self.sources if not get_dataset_spec(key).has_train]
        if no_train:
            raise ValueError(f"pool {self.name!r}: datasets without a train split listed as sources: {no_train}")
        no_test = [key for key in self.val_sets + self.test_sets + self.ood if not get_dataset_spec(key).has_test]
        if no_test:
            raise ValueError(f"pool {self.name!r}: datasets without query/gallery listed for evaluation: {no_test}")
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {self.weight_mode!r}")
        unknown = sorted(set(self.domain_weights) - set(self.sources))
        if unknown:
            raise ValueError(f"domain_weights for non-source datasets: {unknown}")


def load_pool(path: str | Path) -> PoolConfig:
    path = repo_path(path)
    data = load_yaml(path)
    sampler = data.get("sampler") or {}
    pool = PoolConfig(
        name=data.get("name") or path.stem,
        sources=list(data.get("sources") or []),
        ood=list(data.get("ood") or []),
        val_sets=list(data.get("val_sets") or []),
        test_sets=list(data.get("test_sets") or []),
        weight_mode=sampler.get("weight_mode", "sqrt_ids"),
        domain_weights={k: float(v) for k, v in (sampler.get("domain_weights") or {}).items()},
        converters=dict(data.get("converters") or {}),
        path=path,
    )
    pool.validate()
    return pool
