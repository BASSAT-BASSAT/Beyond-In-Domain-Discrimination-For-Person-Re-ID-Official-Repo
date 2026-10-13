#!/usr/bin/env python3
"""Summarize MOT17 train GT for the ReID-FM pool audit."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_mot import MOT17_TRAIN_SEQS, audit_layout  # noqa: E402
from reidfm.utils.paths import raw_root, repo_path  # noqa: E402


def parse_gt(gt_path: Path) -> list[tuple[int, int, float]]:
    """frame, track_id, visibility."""
    rows = []
    with gt_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame, tid, vis = int(parts[0]), int(parts[1]), float(parts[7])
            if tid <= 0:
                continue
            rows.append((frame, tid, vis))
    return rows


def main() -> int:
    mot = raw_root() / "MOT17"
    issues = audit_layout(mot, MOT17_TRAIN_SEQS)
    report_dir = repo_path("reports/MOT17")
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# MOT17 audit (train GT)\n", f"Root: `{mot}`\n\n"]
    if issues:
        lines.append("## Layout\n\n")
        for i in issues:
            lines.append(f"- {i}\n")
        (report_dir / "MOT17_Audit_Report.md").write_text("".join(lines), encoding="utf-8")
        print("layout incomplete; wrote partial report")
        return 1

    vis_hist = defaultdict(int)
    per_seq_ids: dict[str, int] = {}
    for seq in MOT17_TRAIN_SEQS:
        gt = mot / "train" / seq / "gt" / "gt.txt"
        parsed = parse_gt(gt)
        ids = {t for _, t, _ in parsed}
        per_seq_ids[seq] = len(ids)
        for _, _, v in parsed:
            vis_hist[int(v)] += 1

    lines.append("## Sequences\n\n| Sequence | GT track ids |\n|----------|-------------|\n")
    for seq, n in per_seq_ids.items():
        lines.append(f"| {seq} | {n} |\n")
    lines.append("\n## Visibility histogram (GT rows)\n\n")
    for v in sorted(vis_hist):
        lines.append(f"- vis={v}: {vis_hist[v]} rows\n")
    lines.append("\nConverter uses detector FRCNN sequences only; verify zip contains FRCNN train packs.\n")
    out = report_dir / "MOT17_Audit_Report.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
