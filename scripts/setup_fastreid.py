"""Clone FastReID at the pinned commit into ./fast-reid (idempotent).

Usage:
    python scripts/setup_fastreid.py            # clone + checkout pinned commit
    python scripts/setup_fastreid.py --cython   # also build the fast rank cython ext
    python scripts/setup_fastreid.py --from /path/to/existing/fast-reid   # copy a local clone
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reidfm.utils.paths import FASTREID_COMMIT, FASTREID_REPO, FASTREID_ROOT  # noqa: E402


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    print("+", " ".join(cmd))
    out = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _current_commit(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        return _run(["git", "rev-parse", "HEAD"], cwd=root)
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=FASTREID_ROOT)
    parser.add_argument("--from", dest="src", type=Path, default=None, help="copy an existing local clone instead of cloning")
    parser.add_argument("--cython", action="store_true", help="build fastreid/evaluation/rank_cylib (optional speedup)")
    args = parser.parse_args()
    dest: Path = args.dest

    if (dest / "fastreid").is_dir():
        commit = _current_commit(dest)
        if commit and commit != FASTREID_COMMIT:
            print(f"WARNING: {dest} is at {commit}, expected {FASTREID_COMMIT}")
        print(f"FastReID already present at {dest}")
    elif args.src is not None:
        print(f"Copying {args.src} -> {dest}")
        src_root = Path(args.src).resolve()

        def ignore(directory: str, names: list[str]) -> set[str]:
            skipped = {n for n in names if n.endswith(".pth")}
            if Path(directory).resolve() == src_root:  # only the top-level data/log dirs, not fastreid/data/datasets
                skipped |= {n for n in names if n in {"logs", "datasets"}}
            return skipped

        shutil.copytree(src_root, dest, ignore=ignore)
    else:
        _run(["git", "clone", "--quiet", FASTREID_REPO, str(dest)])
        _run(["git", "checkout", "--quiet", FASTREID_COMMIT], cwd=dest)

    commit = _current_commit(dest)
    print(f"FastReID commit: {commit or 'unknown (not a git checkout)'}")

    if args.cython:
        cylib = dest / "fastreid" / "evaluation" / "rank_cylib"
        try:
            _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=cylib)
            print("Built rank_cylib")
        except subprocess.CalledProcessError as exc:
            print(f"rank_cylib build failed (the python fallback still works):\n{exc.stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
