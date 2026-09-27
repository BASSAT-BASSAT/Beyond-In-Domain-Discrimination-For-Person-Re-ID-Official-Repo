"""List registered models (zoo starting points and models trained with this repo).

Examples:
    python scripts/list_models.py
    python scripts/list_models.py --family SBS --trained-on MSMT
    python scripts/list_models.py --init trained --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tabulate import tabulate  # noqa: E402

from reidfm.models.registry import available_models, get_model_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family")
    parser.add_argument("--trained-on")
    parser.add_argument("--init", choices=("reid_pretrained", "imagenet", "trained"))
    parser.add_argument("--backend", choices=("fastreid", "osnet"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    keys = available_models(family=args.family, trained_on=args.trained_on, init=args.init, backend=args.backend)
    specs = [get_model_spec(key) for key in keys]
    if args.json:
        print(json.dumps([spec.to_dict() for spec in specs], indent=2))
        return 0
    rows = []
    for spec in specs:
        local = spec.weights_path.is_file() if spec.weights_path else None
        rows.append([
            spec.key,
            spec.family,
            spec.backbone,
            spec.trained_on,
            spec.init,
            "x".join(map(str, spec.input_size)),
            "-" if local is None else ("yes" if local else "no"),
            "yes" if spec.sha256 else "",
        ])
    print(tabulate(rows, headers=["key", "family", "backbone", "trained on", "init", "input", "local", "sha256"]))
    print(f"\n{len(rows)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
