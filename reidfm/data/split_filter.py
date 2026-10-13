"""Load train/val split JSON files produced by ``build_train_val_split.py``."""

from __future__ import annotations

import json
from pathlib import Path

from reidfm.utils.paths import repo_path


def load_train_split(path: str | Path | None) -> dict[str, set[str]]:
    if not path:
        return {}
    p = repo_path(path) if not Path(path).is_absolute() else Path(path)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): set(str(x) for x in v) for k, v in data.items()}


def train_pid_allowed(dataset_key: str, pid: str, allowed: dict[str, set[str]]) -> bool:
    if not allowed:
        return True
    if dataset_key not in allowed:
        return False
    return pid in allowed[dataset_key]
