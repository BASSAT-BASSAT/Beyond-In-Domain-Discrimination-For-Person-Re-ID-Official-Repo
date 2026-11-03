"""Paper report aggregation (macro R1/R5/mAP/CMC)."""

from __future__ import annotations

import pytest

from reidfm.evaluation.paper_report import build_paper_payload, format_paper_markdown, macro_over
from reidfm.experiments.contract import HELD_OUT_TEST_KEYS


def test_macro_and_cmc_from_results():
    fake = {
        "market1501": {
            "Rank-1": 80.0,
            "Rank-5": 90.0,
            "mAP": 70.0,
            "cmc": [80.0, 85.0, 88.0, 90.0] + [90.0] * 16,
        },
        "msmt17": {
            "Rank-1": 60.0,
            "Rank-5": 70.0,
            "mAP": 50.0,
            "cmc": [60.0] + [65.0] * 19,
        },
    }
    payload = build_paper_payload(fake, held_out_keys=("market1501", "msmt17"))
    macro = payload["macro_public"]
    assert macro["Rank-1"] == pytest.approx(70.0)
    assert macro["Rank-5"] == pytest.approx(80.0)
    assert macro["mAP"] == pytest.approx(60.0)
    assert len(macro["cmc"]) == 20
    md = format_paper_markdown(payload, title="test")
    assert "Macro (public)" in md
    assert "market1501" in md


def test_macro_over_contract_keys():
    results = {
        k: {"Rank-1": 10.0, "Rank-5": 20.0, "mAP": 15.0, "cmc": [10.0] * 20}
        for k in HELD_OUT_TEST_KEYS[:2]
    }
    row = macro_over(list(results), results)
    assert row["Rank-1"] == pytest.approx(10.0)
