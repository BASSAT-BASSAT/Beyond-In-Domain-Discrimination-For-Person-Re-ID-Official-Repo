from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def fastreid_available() -> bool:
    try:
        from reidfm.fastreid_compat import ensure_fastreid

        ensure_fastreid()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if fastreid_available():
        return
    skip = pytest.mark.skip(reason="FastReID not set up (python scripts/setup_fastreid.py)")
    for item in items:
        if "fastreid" in item.keywords:
            item.add_marker(skip)


def write_jpg(path: Path, seed: int, size=(64, 32)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def market_like(tmp_path: Path) -> Path:
    """Tiny Market-1501-style raw folder: 3 train ids, 2 test ids over 2 cameras."""
    root = tmp_path / "raw" / "Market-1501-v15.09.15"
    seed = 0
    for split, pids in (("bounding_box_train", (1, 2, 3)), ("bounding_box_test", (10, 11)), ("query", (10, 11))):
        for pid in pids:
            for cam in (1, 2):
                if split == "query" and cam == 2:
                    continue
                for frame in range(2):
                    write_jpg(root / split / f"{pid:04d}_c{cam}s1_{frame:06d}_00.jpg", seed)
                    seed += 1
    # Market junk/distractor ids that converters must drop.
    write_jpg(root / "bounding_box_test" / "-1_c1s1_000000_00.jpg", 999)
    write_jpg(root / "bounding_box_test" / "0000_c1s1_000000_00.jpg", 998)
    return tmp_path / "raw"
