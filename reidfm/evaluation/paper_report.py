"""Paper-facing held-out evaluation tables (R1, R5, mAP, CMC curve, macro)."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from reidfm.experiments.contract import HELD_OUT_TEST_KEYS

PAPER_RANKS = (1, 5, 10, 20)
CMC_LEN = 20


def _rank_get(metrics: Mapping, rank: int) -> float:
    key = f"Rank-{rank}"
    if key in metrics:
        return float(metrics[key])
    cmc = metrics.get("cmc")
    if isinstance(cmc, list) and len(cmc) >= rank:
        return float(cmc[rank - 1])
    raise KeyError(f"Rank-{rank} missing from metrics")


def summarize_dataset(metrics: Mapping) -> dict:
    cmc = metrics.get("cmc")
    if not isinstance(cmc, list):
        cmc = [_rank_get(metrics, r) for r in PAPER_RANKS if r <= 20]
        if len(cmc) >= CMC_LEN:
            cmc = cmc[:CMC_LEN]
    elif len(cmc) < CMC_LEN:
        last = float(cmc[-1]) if cmc else 0.0
        cmc = list(cmc) + [last] * (CMC_LEN - len(cmc))
    else:
        cmc = list(cmc[:CMC_LEN])
    return {
        "Rank-1": _rank_get(metrics, 1),
        "Rank-5": _rank_get(metrics, 5),
        "mAP": float(metrics["mAP"]),
        "cmc": cmc,
    }


def macro_over(keys: Sequence[str], results: Mapping[str, Mapping]) -> dict:
    rows = [summarize_dataset(results[k]) for k in keys if k in results]
    if not rows:
        return {}
    out = {
        "Rank-1": float(np.mean([r["Rank-1"] for r in rows])),
        "Rank-5": float(np.mean([r["Rank-5"] for r in rows])),
        "mAP": float(np.mean([r["mAP"] for r in rows])),
        "cmc": [float(np.mean([r["cmc"][i] for r in rows])) for i in range(CMC_LEN)],
    }
    return out


def build_paper_payload(
    results: Mapping[str, Mapping],
    *,
    held_out_keys: Sequence[str] = HELD_OUT_TEST_KEYS,
    extra_keys: Sequence[str] = (),
) -> dict:
    per_dataset = {}
    for key in (*held_out_keys, *extra_keys):
        if key not in results or key == "mean":
            continue
        per_dataset[key] = summarize_dataset(results[key])
    public = [k for k in held_out_keys if k in per_dataset]
    payload = {
        "per_dataset": per_dataset,
        "macro_public": macro_over(public, results),
    }
    extras = [k for k in extra_keys if k in per_dataset]
    if extras:
        payload["macro_extra"] = macro_over(extras, results)
    return payload


def format_paper_markdown(payload: dict, title: str = "Held-out evaluation") -> str:
    lines = [f"## {title}", "", "| Dataset | R1 | R5 | mAP |", "|---|---:|---:|---:|"]
    for name, row in payload.get("per_dataset", {}).items():
        lines.append(
            f"| {name} | {row['Rank-1']:.2f} | {row['Rank-5']:.2f} | {row['mAP']:.2f} |"
        )
    macro = payload.get("macro_public") or {}
    if macro:
        lines.append(
            f"| **Macro (public)** | **{macro['Rank-1']:.2f}** | **{macro['Rank-5']:.2f}** | **{macro['mAP']:.2f}** |"
        )
    lines.extend(["", "### CMC (%), ranks 1–20", ""])
    header = "| Dataset | " + " | ".join(str(i) for i in range(1, CMC_LEN + 1)) + " |"
    sep = "|---|" + "|".join("---:" for _ in range(CMC_LEN)) + "|"
    lines.extend([header, sep])
    for name, row in payload.get("per_dataset", {}).items():
        cells = " | ".join(f"{v:.1f}" for v in row["cmc"])
        lines.append(f"| {name} | {cells} |")
    if macro:
        cells = " | ".join(f"{v:.1f}" for v in macro["cmc"])
        lines.append(f"| **Macro (public)** | {cells} |")
    return "\n".join(lines) + "\n"
