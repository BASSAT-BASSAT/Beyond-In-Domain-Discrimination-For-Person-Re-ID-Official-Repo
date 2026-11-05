"""Declarative nine-condition loss study manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reidfm.experiments.contract import LOSS_STUDY_CONDITIONS, LOSS_STUDY_WHITELIST, PRIMARY_INIT_MODEL  # noqa: F401
from reidfm.utils.config import load_yaml
from reidfm.utils.paths import repo_path

_MANIFEST = repo_path("configs/experiments/loss_study_9.yaml")
BACKBONE_MANIFEST = repo_path("configs/experiments/backbone_study.yaml")


def load_backbone_study_manifest(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or BACKBONE_MANIFEST)


BACKBONE_STUDY_CONDITIONS = frozenset(load_backbone_study_manifest()["conditions"].keys())


@dataclass(frozen=True)
class StudyRun:
    condition: str
    seed: int
    config_file: str
    init_model: str
    output_dir: str
    opts: list[str]
    config_hash: str | None = None


def load_loss_study_manifest(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or _MANIFEST)


def _condition_opts(manifest: dict[str, Any], condition: str) -> list[str]:
    if condition not in manifest["conditions"]:
        raise KeyError(f"unknown condition {condition!r}; expected one of {sorted(manifest['conditions'])}")
    entry = manifest["conditions"][condition]
    opts: list[str] = []
    raw = entry.get("opts") or {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, list):
                value = tuple(value)
            elif isinstance(value, bool):
                value = str(value).lower()
            elif not isinstance(value, (tuple, int, float)):
                value = str(value)
            opts.extend([str(key), value])
    else:
        for item in raw:
            if isinstance(item, str):
                opts.extend(item.split())
            elif isinstance(item, (list, tuple)):
                opts.extend(str(x) for x in item)
    opts.extend(["REIDFM.LOSS_STUDY_CONDITION", condition, "REIDFM.STUDY_MODE", True])
    return opts


def flatten_opts_for_cli(opts: list) -> list[str]:
    """Turn study opts (with tuples/bools) into ``train.py`` CLI override pairs."""
    flat: list[str] = []
    i = 0
    while i < len(opts):
        key = str(opts[i])
        value = opts[i + 1] if i + 1 < len(opts) else ""
        if isinstance(value, tuple):
            value = "(" + ", ".join(str(x) for x in value) + ")"
        elif isinstance(value, bool):
            value = str(value).lower()
        else:
            value = str(value)
        flat.extend([key, value])
        i += 2
    return flat


def expand_loss_study_run(
    condition: str,
    seed: int,
    manifest: dict[str, Any] | None = None,
    output_root: str = "outputs/loss_study",
) -> StudyRun:
    manifest = manifest or load_loss_study_manifest()
    base = manifest["base_config"]
    opts = _condition_opts(manifest, condition)
    init_model = manifest["conditions"][condition].get(
        "init_model", manifest.get("init_model", PRIMARY_INIT_MODEL)
    )
    if seed >= 0:
        opts.extend(["REIDFM.SEED", str(seed)])
    out = f"{output_root}/{condition}/seed{seed}"
    return StudyRun(
        condition=condition,
        seed=seed,
        config_file=base,
        init_model=init_model,
        output_dir=out,
        opts=opts,
    )


def validate_condition_diff(base_opts: list[str], condition_opts: list[str]) -> None:
    """Ensure condition-only opts stay within the whitelist (flat KEY VALUE pairs)."""
    def to_dict(pairs: list[str]) -> dict[str, str]:
        data: dict[str, str] = {}
        i = 0
        while i < len(pairs):
            key = pairs[i]
            if i + 1 >= len(pairs):
                break
            data[key] = pairs[i + 1]
            i += 2
        return data

    base = to_dict(base_opts)
    cond = to_dict(condition_opts)
    for key in cond:
        if key in ("REIDFM.STUDY_MODE", "REIDFM.LOSS_STUDY_CONDITION", "REIDFM.SEED"):
            continue
        if key not in LOSS_STUDY_WHITELIST and cond.get(key) != base.get(key):
            raise ValueError(f"condition override {key!r} is not in LOSS_STUDY_WHITELIST")
