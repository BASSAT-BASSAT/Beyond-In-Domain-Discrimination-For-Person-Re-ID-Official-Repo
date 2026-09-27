"""Dataset registry: one :class:`DatasetSpec` per supported dataset.

Registry keys double as the manifest folder name, the ``dataset`` column, and the
FastReID train pid prefix, so they must be lowercase and must not contain ``_``.
The FastReID-visible dataset name (``DATASETS.NAMES``) is ``fastreid_name``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

DATASET_REGISTRY: dict[str, "DatasetSpec"] = {}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    fastreid_name: str
    kind: str  # "image" or "video"
    converter: str  # "module:function"
    raw_dirs: tuple[str, ...]  # candidate folder names under REIDFM_RAW_ROOT
    has_train: bool = True
    has_test: bool = True
    restricted: bool = False
    protocol: str = "official"  # "official" or "reidfm" (our own query/gallery split)
    notes: str = ""
    default_options: dict = field(default_factory=dict)

    def load_converter(self) -> Callable:
        module_name, func_name = self.converter.split(":")
        return getattr(importlib.import_module(module_name), func_name)


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    if "_" in spec.key or spec.key != spec.key.lower():
        raise ValueError(f"dataset key {spec.key!r} must be lowercase without '_'")
    if spec.key in DATASET_REGISTRY:
        raise KeyError(f"dataset {spec.key!r} already registered")
    DATASET_REGISTRY[spec.key] = spec
    return spec


def get_dataset_spec(key: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unknown dataset {key!r}; known: {', '.join(sorted(DATASET_REGISTRY))}") from exc


def spec_for_fastreid_name(name: str) -> DatasetSpec:
    for spec in DATASET_REGISTRY.values():
        if spec.fastreid_name == name:
            return spec
    raise KeyError(f"no reidfm dataset registered as FastReID name {name!r}")


def available_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)


_C = "reidfm.data.converters"

for _spec in (
    DatasetSpec("market1501", "Market1501M", "image", f"{_C}.market_style:convert_market1501",
                ("Market-1501-v15.09.15", "Market-1501", "market1501")),
    DatasetSpec("msmt17", "MSMT17M", "image", f"{_C}.msmt17:convert", ("MSMT17_V1", "MSMT17_V2", "MSMT17", "msmt17"),
                restricted=True),
    DatasetSpec("cuhk03np", "CUHK03NPM", "image", f"{_C}.market_style:convert_cuhk03np",
                ("cuhk03-np", "CUHK03-NP", "cuhk03_np", "cuhk03np"), default_options={"variant": "detected"}),
    DatasetSpec("last", "LaSTM", "image", f"{_C}.last:convert", ("LaST", "last"), restricted=True,
                notes="every test image gets its own camid (official LaST protocol)"),
    DatasetSpec("occposetrack", "OccPoseTrackM", "image", f"{_C}.occposetrack:convert",
                ("Occluded-PoseTrack-ReID", "occluded_posetrack", "OccPoseTrack"), protocol="reidfm",
                notes="query/gallery split generated deterministically unless the raw folder has query/ and gallery/"),
    DatasetSpec("deepchange", "DeepChangeM", "image", f"{_C}.pattern_dirs:convert_deepchange", ("DeepChange", "deepchange"),
                restricted=True, notes="layout assumed from the paper; verify folder names/regex on the real copy"),
    DatasetSpec("prw", "PRWM", "image", f"{_C}.prw:convert", ("PRW-v16.04.20", "PRW", "prw"),
                notes="person crops cut from frames using the official boxes"),
    DatasetSpec("mot17", "MOT17M", "video", f"{_C}.mot:convert_mot17", ("MOT17", "mot17"), has_test=False,
                notes="train only: GT crops from MOT17 train sequences"),
    DatasetSpec("mot20", "MOT20M", "video", f"{_C}.mot:convert_mot20", ("MOT20", "mot20"), has_test=False,
                notes="train only: GT crops from MOT20 train sequences"),
    DatasetSpec("mars", "MARSM", "video", f"{_C}.mars:convert", ("MARS", "mars")),
    DatasetSpec("lpw", "LPWM", "video", f"{_C}.lpw:convert", ("LPW", "lpw", "pep_256x128")),
    DatasetSpec("prid2011", "PRID2011M", "video", f"{_C}.prid2011:convert", ("prid_2011", "PRID2011", "prid2011"),
                has_train=False, notes="OOD test set: cam_a query vs cam_b gallery on the 200 shared ids"),
    DatasetSpec("iust", "IUSTM", "image", f"{_C}.market_style:convert_iust", ("IUST_PersonReID", "IUST", "iust"),
                restricted=True),
    DatasetSpec("rpifield", "RPIfieldM", "image", f"{_C}.pattern_dirs:convert_rpifield", ("RPIfield", "rpifield"),
                notes="needs splits/regex in the pool converter options"),
    DatasetSpec("orbenchrgb", "ORBenchRGBM", "image", f"{_C}.pattern_dirs:convert_orbench_rgb", ("ORBench", "orbench"),
                restricted=True, notes="RGB modality only; private sharing"),
):
    register_dataset(_spec)
