#!/usr/bin/env bash
# Clone FastReID at the pinned commit (see reidfm/utils/paths.py) and optionally build
# the cython rank extension. Thin wrapper around setup_fastreid.py for shell/Kaggle use.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/setup_fastreid.py "$@"
