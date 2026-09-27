"""Unified manifest format shared by every converted dataset.

One folder per dataset::

    <data_root>/<dataset>/manifest.csv
    <data_root>/<dataset>/meta.json
    <data_root>/<dataset>/images/...      (when materialized)

``manifest.csv`` columns: ``path,pid,camid,dataset,split,tracklet_id,frame``.

* ``path`` is relative to the manifest folder, or absolute (``--materialize none``).
* ``pid`` / ``camid`` are strings, unique within the dataset. Test pids that could collide
  with train pids must already be namespaced by the converter (for example ``test_12``).
* ``split`` is ``train``, ``query``, or ``gallery``.
* ``tracklet_id`` groups frames of one track (video datasets); empty for image datasets.
* ``frame`` is the frame index inside the tracklet, when known.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from reidfm.utils.paths import data_roots

FIELDS = ("path", "pid", "camid", "dataset", "split", "tracklet_id", "frame")
SPLITS = ("train", "query", "gallery")
MANIFEST_NAME = "manifest.csv"
META_NAME = "meta.json"


@dataclass(frozen=True)
class ManifestRow:
    path: str
    pid: str
    camid: str
    dataset: str
    split: str
    tracklet_id: str = ""
    frame: str = ""


@dataclass
class Manifest:
    name: str
    root: Path
    rows: list[ManifestRow]
    meta: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.meta.get("kind") or ("video" if any(r.tracklet_id for r in self.rows if r.split != "train") else "image")

    def split(self, split: str) -> list[ManifestRow]:
        return [row for row in self.rows if row.split == split]

    @property
    def train(self) -> list[ManifestRow]:
        return self.split("train")

    @property
    def query(self) -> list[ManifestRow]:
        return self.split("query")

    @property
    def gallery(self) -> list[ManifestRow]:
        return self.split("gallery")

    def resolve(self, row: ManifestRow) -> Path:
        return resolve_path(row, self.root)

    @property
    def has_train(self) -> bool:
        return any(row.split == "train" for row in self.rows)

    @property
    def has_test(self) -> bool:
        return any(row.split == "query" for row in self.rows) and any(row.split == "gallery" for row in self.rows)


def resolve_path(row: ManifestRow, manifest_dir: Path) -> Path:
    path = Path(row.path)
    return path if path.is_absolute() else manifest_dir / path


def validate_rows(rows: Sequence[ManifestRow]) -> list[str]:
    """Return human-readable problems; empty list means the manifest is usable."""
    errors: list[str] = []
    datasets = {row.dataset for row in rows}
    if len(datasets) > 1:
        errors.append(f"multiple dataset names in one manifest: {sorted(datasets)}")
    for name in datasets:
        if "_" in name:
            errors.append(f"dataset name {name!r} must not contain '_' (FastReID pid prefix separator)")
    for index, row in enumerate(rows):
        if row.split not in SPLITS:
            errors.append(f"row {index}: bad split {row.split!r}")
        if not row.path or not row.pid or row.camid == "":
            errors.append(f"row {index}: empty path/pid/camid")
        if len(errors) > 50:
            errors.append("... (truncated)")
            break
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.split, row.path)
        if key in seen:
            errors.append(f"duplicate path in split {row.split}: {row.path}")
            break
        seen.add(key)
    return errors


def write_manifest(path: str | Path, rows: Iterable[ManifestRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return [ManifestRow(**{key: (record.get(key) or "") for key in FIELDS}) for record in reader]


def find_manifest(name: str, roots: Sequence[Path] | None = None) -> Path:
    """Find ``<name>/manifest.csv`` directly under a data root or one level deeper.

    One level deeper covers Kaggle, where an attached dataset is mounted as
    ``/kaggle/input/<kaggle-slug>/<dataset>/manifest.csv``.
    """
    roots = list(roots) if roots is not None else data_roots()
    for root in roots:
        direct = root / name / MANIFEST_NAME
        if direct.is_file():
            return direct
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            candidate = child / name / MANIFEST_NAME
            if child.is_dir() and candidate.is_file():
                return candidate
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"No {name}/{MANIFEST_NAME} under [{searched}]. Set REIDFM_DATA_ROOT or run scripts/prepare_datasets.py.")


def load_manifest(name_or_path: str | Path, roots: Sequence[Path] | None = None) -> Manifest:
    candidate = Path(name_or_path)
    if candidate.suffix == ".csv" and candidate.is_file():
        path = candidate
    elif (candidate / MANIFEST_NAME).is_file():
        path = candidate / MANIFEST_NAME
    else:
        path = find_manifest(str(name_or_path), roots)
    meta_path = path.parent / META_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    rows = read_manifest(path)
    name = meta.get("dataset") or (rows[0].dataset if rows else path.parent.name)
    return Manifest(name=name, root=path.parent, rows=rows, meta=meta)


def manifest_counts(rows: Sequence[ManifestRow]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        subset = [row for row in rows if row.split == split]
        counts[split] = {
            "images": len(subset),
            "ids": len({row.pid for row in subset}),
            "cams": len({row.camid for row in subset}),
            "tracklets": len({row.tracklet_id for row in subset if row.tracklet_id}),
        }
    return counts
