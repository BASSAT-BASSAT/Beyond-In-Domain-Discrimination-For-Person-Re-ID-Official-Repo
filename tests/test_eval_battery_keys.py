"""Protocol result keys -> paper battery keys (held-out eval script)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_unseen_reidraw import _result_key_to_battery  # noqa: E402


def test_msmt17_v2_dataset_name_maps_to_msmt17():
    assert _result_key_to_battery("MSMT17_V2 | msmt17_official_v1") == "msmt17"


def test_market1501_protocol_key():
    assert _result_key_to_battery("Market1501 | market1501_official_v1") == "market1501"
