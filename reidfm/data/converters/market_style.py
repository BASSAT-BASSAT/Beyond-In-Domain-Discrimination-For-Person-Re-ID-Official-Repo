"""Market-1501-style folders: ``bounding_box_train/``, ``query/``, ``bounding_box_test/``.

File names start with ``<pid>_c<cam>`` (for example ``0002_c1s1_000451_03.jpg``).
``pid == -1`` is junk and is always dropped; ``pid == 0`` is a gallery distractor and is
kept only in the gallery (same as the benchmark and FastReID).

Used for Market-1501, CUHK03-NP (``detected``/``labeled``), and IUST_PersonReID.
"""

from __future__ import annotations

import re
from pathlib import Path

from reidfm.data.converters.base import SourceRow, find_dir, list_images

DEFAULT_DIRS = {"train": "bounding_box_train", "query": "query", "gallery": "bounding_box_test"}
DEFAULT_PATTERN = r"([-\d]+)_c(\d+)"


def convert_market_style(
    root: Path,
    options: dict,
    *,
    dirs: dict[str, str] | None = None,
    pattern: str = DEFAULT_PATTERN,
) -> list[SourceRow]:
    dirs = {**DEFAULT_DIRS, **(dirs or {}), **(options.get("dirs") or {})}
    regex = re.compile(options.get("pattern", pattern))
    rows: list[SourceRow] = []
    for split, folder_name in dirs.items():
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"missing {split} folder {folder}")
        for path in list_images(folder):
            match = regex.search(path.name)
            if not match:
                continue
            pid, cam = int(match.group(1)), int(match.group(2))
            if pid == -1 or (pid == 0 and split != "gallery"):
                continue
            rows.append(SourceRow(src=path, pid=str(pid), camid=str(cam), split=split))
    return rows


def _root_with(raw: Path, folder: str) -> Path:
    if (raw / folder).is_dir():
        return raw
    found = find_dir(raw, [folder], max_depth=3)
    if found is None:
        raise FileNotFoundError(f"no {folder}/ under {raw}")
    return found.parent


def convert_market1501(raw: Path, options: dict) -> list[SourceRow]:
    return convert_market_style(_root_with(raw, "bounding_box_train"), options)


def convert_cuhk03np(raw: Path, options: dict) -> list[SourceRow]:
    variant = options.get("variant", "detected")
    variant_dir = raw / variant if (raw / variant).is_dir() else find_dir(raw, [variant], max_depth=2)
    if variant_dir is None:
        raise FileNotFoundError(f"CUHK03-NP: no '{variant}' folder under {raw}")
    return convert_market_style(_root_with(variant_dir, "bounding_box_train"), options)


def convert_iust(raw: Path, options: dict) -> list[SourceRow]:
    return convert_market_style(_root_with(raw, (options.get("dirs") or {}).get("train", "bounding_box_train")), options)


def convert_dukemtmc(raw: Path, options: dict) -> list[SourceRow]:
    return convert_market_style(_root_with(raw, "bounding_box_train"), options)
