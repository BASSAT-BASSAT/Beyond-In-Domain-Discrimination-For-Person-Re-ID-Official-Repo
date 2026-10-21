"""ORBench RGB (ReID5o): official train_annos + test_gallery_and_queries JSON."""

from __future__ import annotations

import json
from pathlib import Path

from reidfm.data.converters.base import SourceRow, find_dir


def _resolve_image(root: Path, entry: dict | str, modalities: tuple[str, ...] = ("vis",)) -> Path | None:
    if isinstance(entry, str):
        candidate = root / entry
        return candidate if candidate.is_file() else None
    path_keys = ("file_path", "filepath", "path", "img_path", "filename", "file_name", "image")
    rel = None
    for key in path_keys:
        if key in entry and entry[key]:
            rel = entry[key]
            break
    if rel is None:
        return None
    rel_path = Path(str(rel))
    if rel_path.is_file():
        return rel_path
    for base in (root, *(root / m for m in modalities)):
        candidate = base / rel_path
        if candidate.is_file():
            return candidate
        if rel_path.name != str(rel_path):
            candidate = base / rel_path.name
            if candidate.is_file():
                return candidate
    return None


def _pid_from_entry(entry: dict | str) -> str | None:
    if isinstance(entry, str):
        return None
    for key in ("pid", "person_id", "identity", "id", "label"):
        if key in entry and entry[key] is not None:
            return str(entry[key]).lstrip("0") or "0"
    return None


def _cam_from_entry(entry: dict | str, default: str = "vis") -> str:
    if isinstance(entry, dict):
        for key in ("camid", "camera", "cam", "modality"):
            if key in entry and entry[key]:
                return str(entry[key])
    return default


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_annotations(blob) -> list:
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for key in ("annotations", "data", "train", "images"):
            if key in blob and isinstance(blob[key], list):
                return blob[key]
        # identity -> list of paths
        rows = []
        for pid, items in blob.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        item = {**item, "pid": item.get("pid", pid)}
                    rows.append(item)
            elif isinstance(items, str):
                rows.append({"file_name": items, "pid": pid})
        return rows
    return []


def convert(raw: Path, options: dict) -> list[SourceRow]:
    root = raw
    if not (root / "train_annos.json").is_file():
        found = find_dir(raw, ["ORBench", "orbench"], max_depth=2)
        if found is not None:
            root = found
    train_json = root / "train_annos.json"
    test_json = root / "test_gallery_and_queries.json"
    if not train_json.is_file():
        raise FileNotFoundError(
            f"ORBench: expected train_annos.json under {raw}; use official ReID5o/ORBench layout"
        )

    rgb_only = bool(options.get("rgb_only", True))
    modalities = ("vis",) if rgb_only else ("vis", "nir", "cp", "sk")
    test_prefix = options.get("test_pid_prefix", "test")

    rows: list[SourceRow] = []
    for entry in _iter_annotations(_load_json(train_json)):
        path = _resolve_image(root, entry, modalities)
        pid = _pid_from_entry(entry)
        if path is None or pid is None:
            continue
        if rgb_only and isinstance(entry, dict):
            mod = str(entry.get("modality", entry.get("cam", "vis"))).lower()
            if mod not in ("vis", "rgb", "0", ""):
                continue
        rows.append(SourceRow(src=path, pid=pid, camid=_cam_from_entry(entry), split="train"))

    if test_json.is_file():
        test_blob = _load_json(test_json)
        queries = test_blob.get("query") or test_blob.get("queries") or []
        gallery = test_blob.get("gallery") or test_blob.get("galleries") or []
        if isinstance(queries, dict):
            queries = [item for sub in queries.values() for item in (sub if isinstance(sub, list) else [sub])]
        if isinstance(gallery, dict):
            gallery = [item for sub in gallery.values() for item in (sub if isinstance(sub, list) else [sub])]
        for split_name, items in (("query", queries), ("gallery", gallery)):
            for entry in items:
                path = _resolve_image(root, entry, modalities)
                pid = _pid_from_entry(entry)
                if path is None or pid is None:
                    continue
                if rgb_only and isinstance(entry, dict):
                    mod = str(entry.get("modality", entry.get("cam", "vis"))).lower()
                    if split_name == "gallery" and mod not in ("vis", "rgb", "0", ""):
                        continue
                ns_pid = f"{test_prefix}{pid}" if test_prefix else pid
                rows.append(
                    SourceRow(src=path, pid=ns_pid, camid=_cam_from_entry(entry), split=split_name)
                )

    if not rows:
        raise FileNotFoundError(f"ORBench: no images resolved under {root} (check JSON paths and vis/ folders)")
    return rows
