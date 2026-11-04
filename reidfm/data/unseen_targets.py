"""Kaggle unseen-target bundle (``.reidraw`` + ``reid_targets_index.json``).

Paper held-out evaluation uses **five public benchmarks only**. ``TeamRecorded`` (and
``TeamRecorded_manifest.json``) are never loaded or evaluated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator, Mapping

from reidfm.experiments.contract import HELD_OUT_TEST_KEYS

# Kaggle filenames in a7medlabib/reid-unseen-evaluation-targets-v1
DEFAULT_REIDRAW_FILES: dict[str, str] = {
    "market1501": "Market1501.reidraw",
    "msmt17": "MSMT17_V2.reidraw",
    "mars": "MARS.reidraw",
    "dukemtmc": "DukeMTMC.reidraw",
    "occludedduke": "OccludedDukeMTMC.reidraw",
}

DATASET_ROOT_ID_BY_KEY: dict[str, str] = {
    "market1501": "Market1501",
    "msmt17": "MSMT17_V2",
    "mars": "MARS",
    "dukemtmc": "DukeMTMC",
    "occludedduke": "OccludedDukeMTMC",
}

SKIP_TARGET_TOKENS = ("teamrecorded", "team_recorded")


def _normalize_key(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "market1501": "market1501",
        "msmt17": "msmt17",
        "msmt17v2": "msmt17",
        "mars": "mars",
        "dukemtmc": "dukemtmc",
        "occludeddukemtmc": "occludedduke",
        "occludedduke": "occludedduke",
    }
    return aliases.get(key, name.strip().lower())


def _skip_target(name: str) -> bool:
    lowered = name.lower().replace("-", "").replace("_", "")
    return any(token.replace("_", "") in lowered for token in SKIP_TARGET_TOKENS)


def find_target_root(candidates: list[Path | str] | None = None) -> Path:
    if candidates is None:
        candidates = [
            Path("/kaggle/input/reid-unseen-evaluation-targets-v1"),
            Path("/kaggle/input/datasets/a7medlabib/reid-unseen-evaluation-targets-v1"),
            Path("/kaggle/input/a7medlabib/reid-unseen-evaluation-targets-v1"),
        ]
    for raw in candidates:
        root = Path(raw)
        if (root / "reid_targets_index.json").is_file():
            return root
    raise FileNotFoundError(
        "Attach Kaggle dataset a7medlabib/reid-unseen-evaluation-targets-v1 "
        "(needs reid_targets_index.json at the dataset root)."
    )


def _coerce_spec(raw: Mapping, root: Path) -> dict | None:
    name = str(
        raw.get("dataset_key")
        or raw.get("key")
        or raw.get("name")
        or raw.get("dataset")
        or ""
    )
    if not name or _skip_target(name):
        return None
    reidraw = (
        raw.get("reidraw_file")
        or raw.get("reidraw")
        or raw.get("file")
        or raw.get("path")
    )
    if not reidraw:
        norm = _normalize_key(name)
        reidraw = DEFAULT_REIDRAW_FILES.get(norm)
    if not reidraw:
        return None
    reidraw_path = root / str(reidraw)
    manifest = raw.get("manifest") or raw.get("manifest_file") or raw.get("manifest_path")
    manifest_path = None
    if manifest:
        candidate = root / str(manifest)
        if candidate.is_file() and not _skip_target(candidate.stem):
            manifest_path = candidate
    return {
        "key": _normalize_key(name),
        "reidraw_path": reidraw_path,
        "manifest_path": manifest_path,
        "dataset_root_id": str(raw.get("dataset_root_id") or raw.get("root_id") or raw.get("dataset") or name),
        "index_dataset": str(raw.get("dataset") or name),
    }


def load_target_specs(root: Path) -> dict[str, dict]:
    index_path = root / "reid_targets_index.json"
    doc = json.loads(index_path.read_text(encoding="utf-8"))
    specs: dict[str, dict] = {}

    entries: list = []
    if isinstance(doc, dict):
        if "targets" in doc and isinstance(doc["targets"], dict):
            for name, item in doc["targets"].items():
                if isinstance(item, dict):
                    merged = {"name": name, **item}
                    entries.append(merged)
                elif isinstance(item, str):
                    entries.append({"name": name, "reidraw_file": item})
        elif "targets" in doc and isinstance(doc["targets"], list):
            entries.extend(doc["targets"])
        else:
            for name, item in doc.items():
                if name in ("version", "schema", "readme"):
                    continue
                if isinstance(item, dict):
                    entries.append({"name": name, **item})
                elif isinstance(item, str):
                    entries.append({"name": name, "reidraw_file": item})
    elif isinstance(doc, list):
        entries = doc

    for raw in entries:
        if not isinstance(raw, Mapping):
            continue
        spec = _coerce_spec(raw, root)
        if spec is None:
            continue
        specs[spec["key"]] = spec

    for key, filename in DEFAULT_REIDRAW_FILES.items():
        if key in specs:
            continue
        path = root / filename
        if path.is_file():
            specs[key] = {
                "key": key,
                "reidraw_path": path,
                "manifest_path": None,
                "dataset_root_id": DATASET_ROOT_ID_BY_KEY.get(key, key),
                "index_dataset": DATASET_ROOT_ID_BY_KEY.get(key, key),
            }

    missing = [k for k in HELD_OUT_TEST_KEYS if k not in specs]
    if missing:
        raise FileNotFoundError(
            f"Unseen target bundle missing benchmarks: {missing}. "
            f"Found: {sorted(specs)} under {root}"
        )
    return {k: specs[k] for k in HELD_OUT_TEST_KEYS if k in specs}


def _manifest_for_spec(spec: dict, root: Path) -> Path:
    if spec.get("manifest_path") and Path(spec["manifest_path"]).is_file():
        return Path(spec["manifest_path"])
    key = spec["key"]
    reidraw_stem = Path(spec["reidraw_path"]).stem
    for pattern in (
        f"{reidraw_stem}_manifest.json",
        f"{reidraw_stem}_manifest.csv",
        f"{key}_manifest.json",
        f"{key}_manifest.csv",
        f"{key}_eval_manifest.csv",
        f"{key}.csv",
        f"{spec['dataset_root_id']}_manifest.csv",
    ):
        path = root / pattern
        if path.is_file() and not _skip_target(path.stem):
            return path
    raise FileNotFoundError(
        f"No query/gallery manifest for target {key!r} under {root}. "
        "Expected a manifest listed in reid_targets_index.json."
    )


def _rows_from_manifest(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, list):
            return [dict(row) for row in doc]
        if isinstance(doc, dict):
            for key in ("records", "rows", "samples", "manifest"):
                if isinstance(doc.get(key), list):
                    return [dict(row) for row in doc[key]]
        raise ValueError(f"Unsupported manifest JSON layout: {path.name}")
    required = {
        "sample_uid",
        "dataset",
        "dataset_root_id",
        "relative_path",
        "identity_key",
        "validation_protocol_id",
        "retrieval_role",
    }
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        rows.extend(dict(row) for row in reader)
    return rows


def iter_target_records(spec: dict, root: Path) -> Iterator[dict]:
    if _skip_target(spec["key"]) or _skip_target(str(spec.get("dataset_root_id", ""))):
        raise ValueError(
            f"{spec['key']!r} is excluded from paper held-out evaluation (TeamRecorded not used)"
        )
    from reidfm.data.reid_target_reader import iter_eval_rows

    if not spec["reidraw_path"].is_file():
        raise FileNotFoundError(f"Missing reidraw archive: {spec['reidraw_path']}")
    dataset = spec.get("index_dataset") or spec["dataset_root_id"]
    for row in iter_eval_rows(
        spec["reidraw_path"],
        bundle_root=root,
        dataset=dataset,
        dataset_root_id=spec["dataset_root_id"],
    ):
        if row.get("retrieval_role") in ("query", "gallery"):
            yield row


def reidraw_files_by_root_id(specs: dict[str, dict]) -> dict[str, Path]:
    return {spec["dataset_root_id"]: Path(spec["reidraw_path"]) for spec in specs.values()}
