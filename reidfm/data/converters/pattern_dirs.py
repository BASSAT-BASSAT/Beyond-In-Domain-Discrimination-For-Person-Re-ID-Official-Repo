"""Generic "folders + filename regex" converter for datasets without a dedicated loader.

Options (all overridable from the pool's ``converters.<key>`` block)::

    splits:                       # split -> list of folders (relative to the raw dir)
      train: [train-set, val-set]
      query: [test-set-query]
      gallery: [test-set-gallery]
    recursive: true               # search sub-folders
    pid_regex: '^P?(\\d+)'        # group 1 = identity
    cam_regex: '_C(\\d+)'         # group 1 = camera; null -> use cam_from
    cam_from: parent              # when cam_regex is null: parent | grandparent | none
    pid_from: regex               # regex | parent (identity = parent folder name)
    test_pid_prefix: test         # namespace test ids unless ids are globally unique
"""

from __future__ import annotations

import re
from pathlib import Path

from reidfm.data.converters.base import SourceRow, list_images


def convert_pattern_dirs(raw: Path, options: dict, defaults: dict | None = None) -> list[SourceRow]:
    opts = {**(defaults or {}), **options}
    splits = opts.get("splits")
    if not splits:
        raise ValueError(f"{raw}: pattern_dirs converter needs a 'splits' option (split -> folders)")
    pid_regex = re.compile(opts["pid_regex"]) if opts.get("pid_regex") else None
    cam_regex = re.compile(opts["cam_regex"]) if opts.get("cam_regex") else None
    pid_from = opts.get("pid_from", "regex")
    cam_from = opts.get("cam_from", "parent")
    test_prefix = opts.get("test_pid_prefix", "test")
    recursive = bool(opts.get("recursive", True))

    rows: list[SourceRow] = []
    for split, folders in splits.items():
        if split not in ("train", "query", "gallery"):
            raise ValueError(f"unknown split {split!r}")
        skip_missing_eval = bool(opts.get("skip_missing_eval_folders")) and split in ("query", "gallery")
        for folder_name in [folders] if isinstance(folders, str) else folders:
            folder = raw / folder_name
            if not folder.is_dir():
                if skip_missing_eval:
                    continue
                raise FileNotFoundError(f"missing {split} folder {folder}")
            for path in list_images(folder, recursive=recursive):
                if pid_from == "parent":
                    pid = path.parent.name
                else:
                    match = pid_regex.search(path.name) if pid_regex else None
                    if not match:
                        continue
                    pid = match.group(1).lstrip("0") or "0"
                if cam_regex:
                    match = cam_regex.search(path.name)
                    camid = match.group(1) if match else "0"
                elif cam_from == "parent":
                    camid = path.parent.name
                elif cam_from == "grandparent":
                    camid = path.parent.parent.name
                else:
                    camid = "0"
                if split != "train" and test_prefix:
                    pid = f"{test_prefix}{pid}"
                rows.append(SourceRow(src=path, pid=pid, camid=camid, split=split))
    return rows


DEEPCHANGE_DEFAULTS = {
    "splits": {
        "train": ["train-set"],
        "query": ["val-set-query", "test-set-query"],
        "gallery": ["val-set-gallery", "test-set-gallery"],
    },
    "skip_missing_eval_folders": True,
    "pid_regex": r"^P?(\d+)",
    "cam_regex": r"_C(\d+)",
    "test_pid_prefix": "",  # DeepChange ids are globally unique across splits
}


def convert_deepchange(raw: Path, options: dict) -> list[SourceRow]:
    return convert_pattern_dirs(raw, options, DEEPCHANGE_DEFAULTS)


def convert_rpifield(raw: Path, options: dict) -> list[SourceRow]:
    return convert_pattern_dirs(raw, options, {"pid_from": "parent", "cam_from": "grandparent"})


def convert_orbench_rgb(raw: Path, options: dict) -> list[SourceRow]:
    return convert_pattern_dirs(raw, options, {"pid_regex": r"^(\d+)", "cam_from": "none"})
