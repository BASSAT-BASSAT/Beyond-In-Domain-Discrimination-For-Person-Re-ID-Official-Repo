"""Expose every manifest dataset to FastReID's ``DATASET_REGISTRY``.

For each :class:`~reidfm.data.registry.DatasetSpec` we register a ``ManifestDataset``
subclass named ``spec.fastreid_name`` (for example ``Market1501M``). FastReID then builds
it with ``DATASET_REGISTRY.get(name)(root=...)`` exactly like its built-in datasets.

Train items follow FastReID's convention: ``(path, "<key>_<pid>", "<key>_<camid>")``.
The ``<key>_`` prefix keeps identities from different datasets apart after relabeling and
lets :class:`~reidfm.data.samplers.DomainBalancedIdentitySampler` recover the domain.
Test pids/camids are mapped to ints for FastReID's own evaluator.
"""

from __future__ import annotations

from reidfm.data.manifest import Manifest, load_manifest
from reidfm.data.registry import DATASET_REGISTRY, DatasetSpec

_REGISTERED = False


def _int_ids(values: list[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}


def _build_class(spec: DatasetSpec):
    from fastreid.data.datasets.bases import ImageDataset

    class ManifestDataset(ImageDataset):
        dataset_name = spec.key
        reidfm_spec = spec

        def __init__(self, root: str = "", **kwargs):
            manifest: Manifest = load_manifest(spec.key)
            self.manifest = manifest
            key = spec.key

            def train():
                return [
                    (str(manifest.resolve(row)), f"{key}_{row.pid}", f"{key}_{row.camid}")
                    for row in manifest.train
                ]

            test_rows = manifest.query + manifest.gallery
            pid_map = _int_ids([row.pid for row in test_rows])
            cam_map = _int_ids([row.camid for row in test_rows])

            def query():
                return [(str(manifest.resolve(r)), pid_map[r.pid], cam_map[r.camid]) for r in manifest.query]

            def gallery():
                return [(str(manifest.resolve(r)), pid_map[r.pid], cam_map[r.camid]) for r in manifest.gallery]

            super().__init__(train, query, gallery, **kwargs)

    ManifestDataset.__name__ = spec.fastreid_name
    ManifestDataset.__qualname__ = spec.fastreid_name
    return ManifestDataset


def register_manifest_datasets() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from fastreid.data.datasets import DATASET_REGISTRY as FASTREID_DATASETS

    for spec in DATASET_REGISTRY.values():
        if spec.fastreid_name not in FASTREID_DATASETS._obj_map:
            FASTREID_DATASETS.register(_build_class(spec))
    _REGISTERED = True
