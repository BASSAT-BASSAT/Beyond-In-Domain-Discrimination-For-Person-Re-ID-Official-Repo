"""Datasets: unified manifests, the dataset registry, pool configs, converters, samplers."""

from reidfm.data.manifest import Manifest, ManifestRow, find_manifest, load_manifest, read_manifest, write_manifest
from reidfm.data.pool import PoolConfig, load_pool
from reidfm.data.registry import DATASET_REGISTRY, DatasetSpec, available_datasets, get_dataset_spec

__all__ = [
    "DATASET_REGISTRY",
    "DatasetSpec",
    "Manifest",
    "ManifestRow",
    "PoolConfig",
    "available_datasets",
    "find_manifest",
    "get_dataset_spec",
    "load_manifest",
    "load_pool",
    "read_manifest",
    "write_manifest",
]
