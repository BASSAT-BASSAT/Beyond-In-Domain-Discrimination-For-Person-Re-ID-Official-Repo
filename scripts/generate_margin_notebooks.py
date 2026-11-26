"""Clone the production ArcFace notebooks into ArcFace margin-study notebooks (camera-ready request).

Reviewer 1 asked whether the backbone-objective interaction survives per-backbone margin
tuning. ArcFace (B2) used the base margin m=0.35 on every backbone; this adds m=0.25 and
m=0.5 on the Strong Baseline, OSNet and ConvNeXt-S, three seeds each (18 runs).

Each generated notebook is its source ArcFace notebook with only these changes:
  * CONDITION -> the margin condition (e.g. OSNet_B2_m025);
  * RESEARCH_COMMIT and the HF bucket prefix -> the commit that defines the new conditions,
    so checkpoints go to a fresh prefix (v3_50epoch_<commit7>/...) and can never overwrite
    the paper's runs;
  * an assert that cfg.MODEL.HEADS.MARGIN equals the intended margin.

The commit must be pushed before the notebooks run (Kaggle clones the repository).

    python scripts/generate_margin_notebooks.py --research-commit <full sha>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "notebooks/reid_loss_study_50epoch_production"
OUT_DIR = ROOT / "notebooks/review_response/margin_study"

# source folder, source stem, source condition, source commit, output stem prefix
SOURCES = (
    ("sbs-arcface", "SBS_ArcFace", "B2", "28c68a1ea068cbe0ca1ea95273a6f80a5094b6f3", "SBS_ArcFace"),
    ("osnet-arcface", "OSNet_ArcFace", "OSNet_B2", "3caf58654fa84227b390b1d3b2aeab3efaab23c0", "OSNet_ArcFace"),
    ("convnext-s-arcface", "ConvNeXtS_ArcFace", "ConvNeXtS_B2", "3caf58654fa84227b390b1d3b2aeab3efaab23c0", "ConvNeXtS_ArcFace"),
)
MARGINS = (("m025", 0.25), ("m050", 0.5))
SEEDS = (0, 1, 2)
ANCHOR = "fingerprint = config_fingerprint(cfg)"


def transform(text: str, *, src_cond: str, dst_cond: str, src_commit: str, dst_commit: str, margin: float) -> str:
    out = text.replace(src_commit, dst_commit).replace(src_commit[:7], dst_commit[:7])
    out = re.sub(rf"(?<![\w]){re.escape(src_cond)}(?![\w])", dst_cond, out)
    if ANCHOR in out:
        out = out.replace(
            ANCHOR,
            f"assert abs(float(cfg.MODEL.HEADS.MARGIN) - {margin}) < 1e-9, cfg.MODEL.HEADS.MARGIN\n"
            f"assert cfg.MODEL.HEADS.CLS_LAYER == 'ArcSoftmax', cfg.MODEL.HEADS.CLS_LAYER\n"
            f"print(f\"[PASS] ArcFace margin m={{cfg.MODEL.HEADS.MARGIN}} (camera-ready margin study)\")\n"
            + ANCHOR,
            1,
        )
    return out


def set_source(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def generate(commit: str) -> list[Path]:
    written = []
    for folder, stem, src_cond, src_commit, out_stem in SOURCES:
        for tag, margin in MARGINS:
            dst_cond = f"{src_cond}_{tag}"
            dst_dir = OUT_DIR / f"{folder}-{tag}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                nb = json.loads((PROD / folder / f"{stem}_Seed{seed}.ipynb").read_text(encoding="utf-8"))
                anchors = 0
                for i, cell in enumerate(nb["cells"]):
                    if cell["cell_type"] == "code":
                        cell["outputs"], cell["execution_count"] = [], None
                    text = "".join(cell.get("source", []))
                    if not text:
                        continue
                    anchors += text.count(ANCHOR)
                    new = transform(text, src_cond=src_cond, dst_cond=dst_cond, src_commit=src_commit,
                                    dst_commit=commit, margin=margin)
                    if i == 0 and cell["cell_type"] == "markdown":
                        new = (f"> **Camera-ready margin study:** ArcFace with m = {margin} "
                               f"(paper runs use m = 0.35). Condition `{dst_cond}`, seed {seed}.\n\n" + new)
                    set_source(cell, new)
                assert anchors == 1, f"{folder} seed {seed}: expected one fingerprint anchor, found {anchors}"
                body = json.dumps(nb, ensure_ascii=False)
                assert src_commit not in body and src_commit[:7] not in body, "old commit left in notebook"
                assert f"CONDITION = '{dst_cond}'" in body, "CONDITION not rewritten"
                path = dst_dir / f"{out_stem}_{tag}_Seed{seed}.ipynb"
                path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                written.append(path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--research-commit", default="", help="full SHA containing the margin conditions (default: HEAD)")
    args = p.parse_args()
    commit = args.research_commit.strip() or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert re.fullmatch(r"[0-9a-f]{40}", commit), commit
    for path in generate(commit):
        print("Wrote", path.relative_to(ROOT))
    print(f"Research commit {commit} | HF prefix v3_50epoch_{commit[:7]}/<CONDITION>/seed<SEED>/")


if __name__ == "__main__":
    main()
