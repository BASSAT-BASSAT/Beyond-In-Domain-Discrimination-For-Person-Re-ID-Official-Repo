"""Export Unified Schema v1 CSV rows into reidfm ``manifest.csv`` format."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reidfm.data.identity import visual_train_pid
from reidfm.data.manifest import META_NAME, MANIFEST_NAME, ManifestRow, manifest_counts, validate_rows, write_manifest
from reidfm.data.registry import get_dataset_spec
from reidfm.utils.paths import raw_root, repo_path

# unified ``dataset`` column -> registry key
UNIFIED_DATASET_TO_KEY: dict[str, str] = {
    "CUHK03_NP": "cuhk03np",
    "LaST": "last",
    "DeepChange": "deepchange",
    "LPW": "lpw",
    "Occluded-PoseTrack-ReID": "occposetrack",
    "ORBench_RGB": "orbenchrgb",
    "PRID2011": "prid2011",
    "RPIfield": "rpifield",
    "IUST_PersonReID": "iust",
}

SPLIT_MAP = {
    "train": "train",
    "query": "query",
    "gallery": "gallery",
}


def registry_key_from_row(row: dict[str, str]) -> str:
    name = (row.get("dataset") or "").strip()
    if name in UNIFIED_DATASET_TO_KEY:
        return UNIFIED_DATASET_TO_KEY[name]
    lowered = name.lower().replace("-", "").replace("_", "")
    for spec_key in ("cuhk03np", "last", "deepchange", "lpw", "occposetrack", "orbenchrgb", "prid2011", "rpifield", "iust"):
        if spec_key in lowered or lowered in spec_key:
            return spec_key
    raise KeyError(f"cannot map unified dataset name {name!r} to a registry key")


def manifest_pid_from_row(row: dict[str, str]) -> str:
    key = (row.get("identity_key") or "").strip()
    if key and "::" in key:
        return key.rsplit("::", 1)[-1]
    raw_pid = (row.get("raw_pid") or "").strip()
    if not raw_pid:
        raise ValueError("row missing raw_pid and identity_key")
    return visual_train_pid(raw_pid, row.get("clothes_id") or "")


def resolve_dataset_root(row: dict[str, str], raw_roots: Path | None = None) -> Path:
    raw_roots = raw_roots or raw_root()
    root_id = (row.get("dataset_root_id") or row.get("dataset") or "").strip()
    for key, unified_name in UNIFIED_DATASET_TO_KEY.items():
        if root_id in (key, unified_name):
            spec = get_dataset_spec(UNIFIED_DATASET_TO_KEY[key])
            for name in spec.raw_dirs:
                candidate = raw_roots / name
                if candidate.is_dir():
                    return candidate
    # fallback: datasets/<root_id> under repo
    repo_ds = repo_path("datasets") / root_id
    if repo_ds.is_dir():
        return repo_ds
    raise FileNotFoundError(f"no raw root for dataset_root_id={root_id!r} under {raw_roots}")


def unified_row_to_manifest(
    row: dict[str, str],
    dataset_key: str,
    dataset_root: Path,
    path_mode: str = "relative",
) -> ManifestRow:
    source_split = (row.get("source_split") or "train").strip().lower()
    split = SPLIT_MAP.get(source_split, source_split)
    if split not in SPLIT_MAP.values():
        raise ValueError(f"unsupported source_split {source_split!r}")

    rel = (row.get("relative_path") or "").strip().replace("\\", "/")
    if not rel:
        raise ValueError("relative_path is empty")
    if path_mode == "absolute":
        path = str((dataset_root / rel).resolve())
    else:
        path = rel

    cam = (row.get("camera_context_id") or row.get("raw_camera_or_view") or "0").strip() or "0"
    tracklet = (row.get("tracklet_id") or "").strip()
    frame = (row.get("frame_id") or "").strip()
    return ManifestRow(
        path=path,
        pid=manifest_pid_from_row(row),
        camid=cam,
        dataset=dataset_key,
        split=split,
        tracklet_id=tracklet,
        frame=frame,
    )


def read_unified_csv(path: Path) -> list[dict[str, str]]:
    from scripts.unified.schema import validate_header

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        validate_header(reader.fieldnames)
        return list(reader)


def export_unified_csv(
    unified_path: Path,
    out_dir: Path,
    dataset_key: str | None = None,
    raw_roots: Path | None = None,
    path_mode: str = "relative",
    kind: str = "image",
) -> dict:
    """Write ``out_dir/manifest.csv`` + ``meta.json`` from one unified v1 CSV."""
    rows_in = read_unified_csv(unified_path)
    if not rows_in:
        raise RuntimeError(f"{unified_path}: no rows")
    key = dataset_key or registry_key_from_row(rows_in[0])
    root = resolve_dataset_root(rows_in[0], raw_roots)
    manifest_rows = [unified_row_to_manifest(r, key, root, path_mode=path_mode) for r in rows_in]
    problems = validate_rows(manifest_rows)
    if problems:
        raise ValueError(f"manifest validation failed: {problems[:5]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / MANIFEST_NAME, manifest_rows)
    meta = {
        "dataset": key,
        "kind": kind,
        "source": str(unified_path),
        "counts": manifest_counts(manifest_rows),
        "dataset_root": str(root),
        "path_mode": path_mode,
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
