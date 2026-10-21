"""Prepare Occluded-DukeMTMC layout from a local DukeMTMC-reID zip (license-gated)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("duke_zip", type=Path, help="path to DukeMTMC-reID.zip")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder under REIDFM_RAW_ROOT (default: Occluded_Duke)",
    )
    parser.add_argument(
        "--converter-repo",
        type=Path,
        default=None,
        help="clone of https://github.com/lightas/Occluded-DukeMTMC-Dataset",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.duke_zip.is_file():
        print(f"missing Duke zip: {args.duke_zip}", file=sys.stderr)
        return 1
    repo = args.converter_repo
    if repo is None:
        repo = REPO_ROOT / "third_party" / "Occluded-DukeMTMC-Dataset"
    script = repo / "convert_duke_to_occduke.py"
    if not script.is_file():
        print(
            f"converter not found at {script}; clone Occluded-DukeMTMC-Dataset into third_party/",
            file=sys.stderr,
        )
        return 1
    out = args.out
    if out is None:
        from reidfm.utils.paths import raw_root

        out = raw_root() / "Occluded_Duke"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, str(script), str(args.duke_zip)], cwd=repo)
    built = repo / "Occluded_Duke"
    if built.is_dir():
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(built, out)
    print(f"Occluded-Duke ready at {out}")
    print("Then: python scripts/prepare_datasets.py --datasets occludedduke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
