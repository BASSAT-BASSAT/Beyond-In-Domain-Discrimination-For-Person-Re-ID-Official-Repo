"""Read official query/gallery rows from Kaggle ``*.reidraw`` evaluation archives.

The unseen-target bundle ships ``reid_target_reader.py`` beside the ``.reidraw`` files.
When present, we delegate to that module (authoritative).

Kaggle targets use ``REID_RAW_ARCHIVE_MIRROR_V1``: renamed official zip/tar files, not
REIDPACK3. We read embedded ``validation_manifest.csv`` (or rebuild query/gallery rows
from the official layout) via :mod:`reidfm.data.raw_archive_mirror`. REIDPACK3 footers are
only used when the file header is ``REIDPACK3``.
"""

from __future__ import annotations

import importlib.util
import inspect
import sqlite3
import struct
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

HEADER = b"REIDPACK3\n"
FOOTER = struct.Struct("<8sQQ32s")

_ROW_ALIASES = {
    "relative_path": ("relative_path", "path", "file_path", "image_path"),
    "identity_key": ("identity_key", "pid", "person_id", "label"),
    "sample_uid": ("sample_uid", "uid", "image_id"),
    "validation_protocol_id": ("validation_protocol_id", "protocol_id", "protocol"),
    "retrieval_role": ("retrieval_role", "role", "split"),
    "raw_camera_or_view": ("raw_camera_or_view", "camid", "camera_id", "camera"),
    "camera_context_id": ("camera_context_id", "camera_context", "context_id"),
    "tracklet_id": ("tracklet_id", "track_id"),
    "dataset": ("dataset",),
    "dataset_root_id": ("dataset_root_id", "root_id", "domain"),
}


def _pick(row: Mapping, field: str, default: str = "") -> str:
    for key in _ROW_ALIASES[field]:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize_eval_row(row: Mapping, *, dataset: str, dataset_root_id: str) -> dict:
    role = _pick(row, "retrieval_role").lower()
    if role in ("q", "query"):
        role = "query"
    elif role in ("g", "gallery"):
        role = "gallery"
    out = {
        "sample_uid": _pick(row, "sample_uid"),
        "dataset": _pick(row, "dataset") or dataset,
        "dataset_root_id": _pick(row, "dataset_root_id") or dataset_root_id,
        "relative_path": _pick(row, "relative_path"),
        "identity_key": _pick(row, "identity_key"),
        "validation_protocol_id": _pick(row, "validation_protocol_id"),
        "retrieval_role": role,
        "raw_camera_or_view": _pick(row, "raw_camera_or_view"),
        "camera_context_id": _pick(row, "camera_context_id"),
        "tracklet_id": _pick(row, "tracklet_id"),
    }
    if not out["sample_uid"]:
        out["sample_uid"] = f"{out['dataset_root_id']}::{out['relative_path']}"
    return out


def _load_bundle_module(bundle_root: Path):
    import sys

    script = bundle_root / "reid_target_reader.py"
    if not script.is_file():
        return None
    bundle_root = Path(bundle_root).resolve()
    path_token = str(bundle_root)
    inserted = False
    if sys.path[:1] != [path_token]:
        sys.path.insert(0, path_token)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location("reid_target_reader", script)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted and sys.path[:1] == [path_token]:
            sys.path.pop(0)


def _call_bundle_reader(
    module,
    reidraw_path: Path,
    dataset: str,
    dataset_root_id: str,
    *,
    bundle_root: Path | None = None,
    index_path: Path | None = None,
) -> list[dict] | None:
    reidraw_path = Path(reidraw_path)
    reidraw_str = str(reidraw_path)
    bundle_root = Path(bundle_root) if bundle_root else None
    common = {
        "reidraw_path": reidraw_path,
        "path": reidraw_path,
        "archive_path": reidraw_path,
        "reidraw": reidraw_path,
        "reidraw_file": reidraw_path,
        "dataset": dataset,
        "dataset_name": dataset,
        "dataset_root_id": dataset_root_id,
        "root_id": dataset_root_id,
        "bundle_root": bundle_root,
        "target_root": bundle_root,
        "targets_root": bundle_root,
        "index_path": index_path,
        "reid_targets_index": index_path,
    }
    attempts = (
        ("iter_eval_manifest_rows", common),
        ("iter_eval_rows", common),
        ("iter_eval_records", common),
        ("iter_target_eval_rows", common),
        ("load_eval_manifest_rows", common),
        ("load_eval_rows", common),
        ("read_eval_manifest_rows", common),
        ("read_eval_rows", common),
        ("iter_manifest_rows", common),
        ("get_eval_rows", common),
    )
    for name, kwargs in attempts:
        func = getattr(module, name, None)
        if not callable(func):
            continue
        params = inspect.signature(func).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
        try:
            result = func(**filtered)
        except TypeError:
            try:
                result = func(reidraw_str, dataset)
            except TypeError:
                try:
                    result = func(reidraw_path)
                except (TypeError, ValueError):
                    continue
            except ValueError:
                continue
        if isinstance(result, Mapping):
            for key in ("records", "rows", "manifest"):
                if isinstance(result.get(key), list):
                    result = result[key]
                    break
        if not hasattr(result, "__iter__") or isinstance(result, (str, bytes)):
            continue
        rows = [normalize_eval_row(dict(row), dataset=dataset, dataset_root_id=dataset_root_id) for row in result]
        if rows:
            return rows
    cls = getattr(module, "ReidTargetReader", None) or getattr(module, "ReIDTargetReader", None)
    if cls is not None:
        for kwargs in (
            {"reidraw_path": reidraw_path, "dataset": dataset},
            {"path": reidraw_path, "dataset": dataset},
            {"reidraw_path": reidraw_path},
        ):
            params = inspect.signature(cls).parameters
            filtered = {k: v for k, v in kwargs.items() if k in params}
            try:
                reader = cls(**filtered)
            except TypeError:
                continue
            for method in ("iter_rows", "iter_records", "manifest_rows"):
                fn = getattr(reader, method, None)
                if callable(fn):
                    rows = [
                        normalize_eval_row(dict(row), dataset=dataset, dataset_root_id=dataset_root_id)
                        for row in fn()
                    ]
                    if rows:
                        return rows
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        if name in {item[0] for item in attempts}:
            continue
        func = getattr(module, name, None)
        if not callable(func):
            continue
        lowered = name.lower()
        if "eval" not in lowered and "manifest" not in lowered and "target" not in lowered:
            continue
        try:
            filtered = {k: v for k, v in common.items() if k in inspect.signature(func).parameters}
            result = func(**filtered)
        except (TypeError, ValueError):
            continue
        if isinstance(result, Mapping):
            for key in ("records", "rows", "manifest"):
                if isinstance(result.get(key), list):
                    result = result[key]
                    break
        if not hasattr(result, "__iter__") or isinstance(result, (str, bytes)):
            continue
        rows = [
            normalize_eval_row(dict(row), dataset=dataset, dataset_root_id=dataset_root_id)
            for row in result
        ]
        if rows:
            return rows
    return None


def _extract_sqlite_index(reidraw_path: Path) -> Path:
    import hashlib
    import os

    with reidraw_path.open("rb") as stream:
        if stream.read(len(HEADER)) != HEADER:
            raise ValueError(f"{reidraw_path.name}: expected REIDPACK3 header")
        stream.seek(0, os.SEEK_END)
        total = stream.tell()
        if total <= len(HEADER) + FOOTER.size:
            raise ValueError(f"{reidraw_path.name}: file too small for REIDPACK3 footer")
        stream.seek(total - FOOTER.size)
        marker, start, size, digest = FOOTER.unpack(stream.read(FOOTER.size))
        if marker != b"RIDX3END":
            raise ValueError(f"{reidraw_path.name}: invalid REIDPACK3 footer")
        stream.seek(start)
        payload = stream.read(size)
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError(f"{reidraw_path.name}: REIDPACK3 index checksum mismatch")
    tmp = tempfile.NamedTemporaryFile(prefix="reid_eval_index_", suffix=".sqlite", delete=False)
    tmp.write(payload)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _manifest_table(conn: sqlite3.Connection) -> tuple[str, list[str]] | None:
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        lower = {c.lower() for c in cols}
        if "retrieval_role" in lower and (
            "relative_path" in lower or "path" in lower or "leaf_path" in lower
        ):
            return table, cols
    return None


def _row_from_sqlite(names: list[str], values: tuple) -> dict:
    return {name: values[index] for index, name in enumerate(names)}


def iter_eval_rows_from_reidraw_index(
    reidraw_path: Path,
    *,
    dataset: str,
    dataset_root_id: str,
) -> Iterator[dict]:
    index_path = _extract_sqlite_index(Path(reidraw_path))
    try:
        conn = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
        try:
            found = _manifest_table(conn)
            if found is None:
                tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                raise ValueError(
                    f"{reidraw_path.name}: no eval manifest table in REIDPACK3 index "
                    f"(tables: {tables})"
                )
            table, columns = found
            query = f"SELECT * FROM {table}"
            for raw in conn.execute(query):
                row = normalize_eval_row(
                    _row_from_sqlite(columns, raw),
                    dataset=dataset,
                    dataset_root_id=dataset_root_id,
                )
                if row["retrieval_role"] in ("query", "gallery") and row["relative_path"]:
                    yield row
        finally:
            conn.close()
    finally:
        index_path.unlink(missing_ok=True)


def iter_eval_rows(
    reidraw_path: Path,
    *,
    bundle_root: Path | None,
    dataset: str,
    dataset_root_id: str,
) -> Iterator[dict]:
    """Yield normalized V3 eval rows for one ``.reidraw`` target."""
    from reidfm.data.raw_archive_mirror import detect_archive_kind, iter_eval_rows_from_raw_mirror

    reidraw_path = Path(reidraw_path)
    index_path = None
    if bundle_root is not None:
        candidate = Path(bundle_root) / "reid_targets_index.json"
        if candidate.is_file():
            index_path = candidate
    if bundle_root is not None:
        module = _load_bundle_module(Path(bundle_root))
        if module is not None:
            rows = _call_bundle_reader(
                module,
                reidraw_path,
                dataset,
                dataset_root_id,
                bundle_root=Path(bundle_root),
                index_path=index_path,
            )
            if rows:
                for row in rows:
                    if row["retrieval_role"] in ("query", "gallery"):
                        yield row
                return
    try:
        kind = detect_archive_kind(reidraw_path)
    except ValueError as error:
        raise ValueError(
            f"{reidraw_path.name}: cannot read eval rows (bundle reader failed and archive is not "
            f"REIDPACK3 / zip / tar). Attach the Kaggle target bundle or update reidfm."
        ) from error
    if kind != "reidpack3":
        yielded = False
        for row in iter_eval_rows_from_raw_mirror(
            reidraw_path,
            dataset=dataset,
            dataset_root_id=dataset_root_id,
        ):
            yielded = True
            yield row
        if yielded:
            return
        raise ValueError(
            f"{reidraw_path.name}: no query/gallery manifest inside mirror archive and "
            f"layout fallback produced no eval rows"
        )
    yield from iter_eval_rows_from_reidraw_index(
        reidraw_path,
        dataset=dataset,
        dataset_root_id=dataset_root_id,
    )
