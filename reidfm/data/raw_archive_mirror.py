"""REID_RAW_ARCHIVE_MIRROR_V1: official benchmark zips/tars exposed as ``*.reidraw`` on Kaggle."""

from __future__ import annotations

import csv
import io
import json
import posixpath
import tarfile
import zipfile
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from reidfm.data.v3_pack_reader import HEADER as REIDPACK3_HEADER

ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

MANIFEST_BASENAMES = (
    "validation_manifest.csv",
    "eval_manifest.csv",
    "query_gallery_manifest.csv",
    "held_out_eval_manifest.csv",
    "held_out_manifest.csv",
)

ROOT_ID_TO_REGISTRY_KEY: dict[str, str] = {
    "Market1501": "market1501",
    "MSMT17_V2": "msmt17",
    "MARS": "mars",
    "DukeMTMC": "dukemtmc",
    "OccludedDukeMTMC": "occludedduke",
}

_RESOLVED_CACHE_SIZE = 8192


def detect_archive_kind(path: Path) -> str:
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(len(REIDPACK3_HEADER) + 4)
    if head.startswith(REIDPACK3_HEADER):
        return "reidpack3"
    if any(head.startswith(magic) for magic in ZIP_MAGICS):
        return "zip"
    if tarfile.is_tarfile(path):
        return "tar"
    raise ValueError(
        f"{path.name}: not REIDPACK3, zip, or tar (REID_RAW_ARCHIVE_MIRROR expected on Kaggle targets)"
    )


def _member_basename(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name


def _find_manifest_member(members: list[str]) -> str | None:
    lowered = {m: _member_basename(m).lower() for m in members}
    for target in MANIFEST_BASENAMES:
        target_lower = target.lower()
        for member, base in lowered.items():
            if base == target_lower:
                return member
    for member, base in lowered.items():
        if base.endswith("_manifest.csv") and "train" not in base:
            return member
    for member, base in lowered.items():
        if base.endswith("_eval_manifest.json") or base.endswith("_manifest.json"):
            if "teamrecorded" in base.replace("-", "").replace("_", ""):
                continue
            return member
    return None


def _parse_manifest_bytes(payload: bytes) -> list[dict]:
    text = payload.decode("utf-8-sig")
    if text.lstrip().startswith(("{", "[")):
        doc = json.loads(text)
        if isinstance(doc, list):
            return [dict(row) for row in doc]
        if isinstance(doc, dict):
            for key in ("records", "rows", "samples", "manifest"):
                if isinstance(doc.get(key), list):
                    return [dict(row) for row in doc[key]]
        raise ValueError("Unsupported JSON manifest layout")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Empty CSV manifest")
    return [dict(row) for row in reader]


def _read_zip_member(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member)


def _read_tar_member(path: Path, member: str) -> bytes:
    with tarfile.open(path, "r:*") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        return extracted.read()


def _archive_members(path: Path, kind: str) -> list[str]:
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            return list(archive.namelist())
    with tarfile.open(path, "r:*") as archive:
        return [member.name for member in archive.getmembers() if member.isfile()]
    return []


def _read_member(path: Path, kind: str, member: str) -> bytes:
    if kind == "zip":
        return _read_zip_member(path, member)
    return _read_tar_member(path, member)


def iter_manifest_rows_from_archive(reidraw_path: Path) -> list[dict]:
    """Return raw manifest dict rows embedded in a mirror ``.reidraw`` archive."""
    path = Path(reidraw_path)
    kind = detect_archive_kind(path)
    if kind == "reidpack3":
        return []
    members = _archive_members(path, kind)
    manifest_member = _find_manifest_member(members)
    if manifest_member is None:
        return []
    payload = _read_member(path, kind, manifest_member)
    return _parse_manifest_bytes(payload)


def _default_protocol_id(dataset_root_id: str) -> str:
    key = ROOT_ID_TO_REGISTRY_KEY.get(dataset_root_id, dataset_root_id).lower()
    return f"{key}_official_v1"


def _source_rows_from_archive(reidraw_path: Path, dataset_root_id: str) -> list[dict]:
    """Build eval rows from official folder layout when no embedded manifest exists."""
    registry_key = ROOT_ID_TO_REGISTRY_KEY.get(dataset_root_id)
    if registry_key is None:
        return []
    import tempfile

    from reidfm.data.registry import get_dataset_spec

    kind = detect_archive_kind(reidraw_path)
    if kind == "reidpack3":
        return []
    protocol = _default_protocol_id(dataset_root_id)
    spec = get_dataset_spec(registry_key)
    with tempfile.TemporaryDirectory(prefix="reid_mirror_extract_") as tmp_name:
        tmp = Path(tmp_name)
        if kind == "zip":
            with zipfile.ZipFile(reidraw_path) as archive:
                archive.extractall(tmp)
        else:
            with tarfile.open(reidraw_path, "r:*") as archive:
                archive.extractall(tmp)
        raw_root = tmp
        for name in spec.raw_dirs:
            candidate = tmp / name
            if candidate.is_dir():
                raw_root = candidate
                break
        else:
            # Single top-level folder inside the zip (common for Market zip).
            children = [p for p in tmp.iterdir() if p.is_dir()]
            if len(children) == 1:
                raw_root = children[0]
        rows = spec.load_converter()(raw_root, dict(spec.default_options))
        out: list[dict] = []
        for row in rows:
            if row.split not in ("query", "gallery"):
                continue
            try:
                rel = row.src.resolve().relative_to(raw_root.resolve()).as_posix()
            except ValueError:
                rel = row.src.name
            out.append(
                {
                    "sample_uid": f"{dataset_root_id}::{rel}",
                    "dataset": dataset_root_id,
                    "dataset_root_id": dataset_root_id,
                    "relative_path": rel,
                    "identity_key": row.pid,
                    "validation_protocol_id": protocol,
                    "retrieval_role": row.split,
                    "raw_camera_or_view": row.camid,
                    "camera_context_id": "",
                    "tracklet_id": row.tracklet_id or "",
                }
            )
        return out


class RawArchiveImageReader:
    """Random-access image bytes from a zip/tar ``.reidraw`` mirror (no full extract)."""

    def __init__(self, archive_path: str | Path):
        self.path = Path(archive_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.kind = detect_archive_kind(self.path)
        if self.kind == "reidpack3":
            raise ValueError(f"{self.path.name}: use V3PackReader for REIDPACK3")
        self._zip: zipfile.ZipFile | None = None
        self._tar: tarfile.TarFile | None = None
        self._members: list[str] = []
        self._by_basename: dict[str, list[str]] = {}
        self._resolved: OrderedDict[str, str] = OrderedDict()
        self._open()

    def _open(self) -> None:
        if self.kind == "zip":
            self._zip = zipfile.ZipFile(self.path)
            self._members = self._zip.namelist()
        else:
            self._tar = tarfile.open(self.path, "r:*")
            self._members = [m.name for m in self._tar.getmembers() if m.isfile()]
        for member in self._members:
            base = _member_basename(member).casefold()
            self._by_basename.setdefault(base, []).append(member)

    def _resolve_member(self, relative_path: str) -> str:
        requested = posixpath.normpath(relative_path.replace("\\", "/"))
        if requested in ("", "."):
            raise ValueError("Empty image path")
        cached = self._resolved.get(requested)
        if cached is not None:
            self._resolved.move_to_end(requested)
            return cached
        if requested in self._members:
            self._store_resolved(requested, requested)
            return requested
        suffix = PurePosixPath(requested).name.casefold()
        candidates = self._by_basename.get(suffix, [])
        if len(candidates) == 1:
            self._store_resolved(requested, candidates[0])
            return candidates[0]
        req_parts = tuple(PurePosixPath(requested).parts)
        best_score = 0
        best: list[str] = []
        for member in candidates:
            score = 0
            mem_parts = tuple(PurePosixPath(member).parts)
            for left, right in zip(reversed(req_parts), reversed(mem_parts)):
                if left.casefold() != right.casefold():
                    break
                score += 1
            if score > best_score:
                best_score = score
                best = [member]
            elif score == best_score and score > 0:
                best.append(member)
        if len(best) != 1 or best_score < 1:
            raise FileNotFoundError(relative_path)
        self._store_resolved(requested, best[0])
        return best[0]

    def _store_resolved(self, requested: str, member: str) -> None:
        self._resolved[requested] = member
        if len(self._resolved) > _RESOLVED_CACHE_SIZE:
            self._resolved.popitem(last=False)

    def read_bytes(self, relative_path: str) -> bytes:
        member = self._resolve_member(relative_path)
        if self._zip is not None:
            return self._zip.read(member)
        assert self._tar is not None
        extracted = self._tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(relative_path)
        return extracted.read()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        self._resolved.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def iter_eval_rows_from_raw_mirror(
    reidraw_path: Path,
    *,
    dataset: str,
    dataset_root_id: str,
) -> Iterator[dict]:
    from reidfm.data.reid_target_reader import normalize_eval_row

    path = Path(reidraw_path)
    raw_rows = iter_manifest_rows_from_archive(path)
    if not raw_rows:
        raw_rows = _source_rows_from_archive(path, dataset_root_id)
    if not raw_rows:
        return
    for row in raw_rows:
        normalized = normalize_eval_row(row, dataset=dataset, dataset_root_id=dataset_root_id)
        if normalized["retrieval_role"] in ("query", "gallery") and normalized["relative_path"]:
            yield normalized
