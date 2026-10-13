#!/usr/bin/env python3
"""MOT Challenge download helper (license gate + layout checks).

MOT data must be obtained from https://motchallenge.net/ under their terms.
Place downloaded zips in ``$REIDFM_RAW_ROOT/downloads/`` and run with ``--extract``.

Expected layout after extract (matches reidfm MOT converters):
    MOT17/train/<SEQ>/img1/*.jpg
    MOT17/train/<SEQ>/gt/gt.txt
    MOT20/train/<SEQ>/img1/*.jpg
    MOT20/train/<SEQ>/gt/gt.txt
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from reidfm.utils.paths import raw_root

MOT17_TRAIN_SEQS = (
    "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13",
)
MOT20_TRAIN_SEQS = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


def _check_seq(root: Path, seq: str) -> list[str]:
    issues = []
    base = root / "train" / seq
    if not base.is_dir():
        return [f"missing {base}"]
    if not (base / "img1").is_dir():
        issues.append(f"{seq}: no img1/")
    gt = base / "gt" / "gt.txt"
    if not gt.is_file():
        issues.append(f"{seq}: no gt/gt.txt")
    return issues


def audit_layout(mot_root: Path, sequences: tuple[str, ...]) -> list[str]:
    return [msg for seq in sequences for msg in _check_seq(mot_root, seq)]


def extract_zips(download_dir: Path, dest: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    for zip_path in sorted(download_dir.glob("*.zip")):
        print(f"extracting {zip_path.name} -> {dest}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--extract", action="store_true", help="extract downloads/*.zip into MOT17/ or MOT20/")
    parser.add_argument("--dataset", choices=("mot17", "mot20", "both"), default="both")
    args = parser.parse_args()

    raw = args.raw_root or raw_root()
    print(
        "MOT Challenge data is not redistributed by this repo.\n"
        "1. Register at https://motchallenge.net/\n"
        "2. Download MOT17 / MOT20 training packs\n"
        f"3. Place zips in: {raw / 'downloads'}\n"
        f"4. Run: python scripts/download_mot.py --extract\n"
        f"Raw root: {raw}\n"
    )

    if args.extract:
        dl = raw / "downloads"
        if args.dataset in ("mot17", "both"):
            extract_zips(dl, raw / "MOT17")
        if args.dataset in ("mot20", "both"):
            extract_zips(dl, raw / "MOT20")

    issues = []
    if args.dataset in ("mot17", "both"):
        issues += audit_layout(raw / "MOT17", MOT17_TRAIN_SEQS)
    if args.dataset in ("mot20", "both"):
        issues += audit_layout(raw / "MOT20", MOT20_TRAIN_SEQS)

    if issues:
        print("Layout issues:")
        for line in issues:
            print(" ", line)
        return 1
    print("MOT layout OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
