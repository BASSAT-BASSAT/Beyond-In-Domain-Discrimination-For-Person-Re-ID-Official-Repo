"""Clone the OSNet and ConvNeXt-S Circle-softmax notebooks into ArcFace variants.

Requires the research commit that contains the ArcFace backbone conditions in
``configs/experiments/backbone_study.yaml``. Pass ``--research-commit`` after
pushing, or omit to use the current ``HEAD``. Checkpoint keys inside the
notebooks stay as stored in the bucket.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "notebooks/reid_loss_study_50epoch_production"

# folder, notebook stem, checkpoint key
PAIRS = (
    (
        "osnet-circle-softmax-triplet",
        "osnet-arcface",
        "OSNet_CircleSoftmax_Triplet",
        "OSNet_ArcFace",
        "OSNet_R0",
        "OSNet_B2",
    ),
    (
        "convnext-s-circle-softmax-triplet",
        "convnext-s-arcface",
        "ConvNeXtS_CircleSoftmax_Triplet",
        "ConvNeXtS_ArcFace",
        "ConvNeXtS_R0",
        "ConvNeXtS_B2",
    ),
)


def _cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_cell_text(cell: dict, text: str) -> None:
    if text.endswith("\n"):
        body = text[:-1]
        cell["source"] = [line + "\n" for line in body.split("\n")]
    else:
        cell["source"] = [line + "\n" for line in text.split("\n")]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")


def _transform(
    text: str,
    *,
    src_cond: str,
    dst_cond: str,
    src_title: str,
    dst_title: str,
    commit: str,
    bucket: str,
) -> str:
    out = text
    out = out.replace(src_cond, dst_cond)
    out = out.replace(src_title, dst_title)
    out = out.replace("Circle-softmax + triplet", "ArcFace")
    out = out.replace("Circle-softmax + Triplet", "ArcFace")
    out = out.replace("v3_50epoch_b69c47a", bucket)
    out = out.replace(
        "reid_loss_study_50epoch_b69c47a",
        f"reid_loss_study_50epoch_{bucket.removeprefix('v3_50epoch_')}",
    )
    out = re.sub(
        r'RESEARCH_COMMIT = "[0-9a-f]{40}"',
        f'RESEARCH_COMMIT = "{commit}"',
        out,
        count=1,
    )
    # Objective asserts (R0 notebooks only checked backbone / ImageNet).
    needle = 'assert cfg.MODEL.BACKBONE.PRETRAIN, "ImageNet initialisation must be enabled"'
    if needle in out and "CLS_LAYER == 'ArcSoftmax'" not in out:
        out = out.replace(
            needle,
            needle
            + "\n"
            + "assert cfg.MODEL.HEADS.CLS_LAYER == 'ArcSoftmax', cfg.MODEL.HEADS.CLS_LAYER\n"
            + "assert list(cfg.MODEL.LOSSES.NAME) == ['CrossEntropyLoss'], list(cfg.MODEL.LOSSES.NAME)\n"
            + "assert 'TripletLoss' not in cfg.MODEL.LOSSES.NAME",
            1,
        )
    return out


def generate(research_commit: str) -> list[Path]:
    bucket = f"v3_50epoch_{research_commit[:7]}"
    written: list[Path] = []
    for src_dir, dst_dir_name, src_stem, dst_stem, src_cond, dst_cond in PAIRS:
        dst_dir = PROD / dst_dir_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for seed in (0, 1, 2):
            src_path = PROD / src_dir / f"{src_stem}_Seed{seed}.ipynb"
            dst_path = dst_dir / f"{dst_stem}_Seed{seed}.ipynb"
            nb = json.loads(src_path.read_text(encoding="utf-8"))
            for cell in nb["cells"]:
                text = _cell_text(cell)
                if not text:
                    continue
                text = _transform(
                    text,
                    src_cond=src_cond,
                    dst_cond=dst_cond,
                    src_title="Circle-softmax + triplet",
                    dst_title="ArcFace",
                    commit=research_commit,
                    bucket=bucket,
                )
                if "CONDITION =" in text and "SEED =" in text:
                    text = re.sub(r"CONDITION = '[^']+'", f"CONDITION = '{dst_cond}'", text, count=1)
                    text = re.sub(r"SEED = \d+", f"SEED = {seed}", text, count=1)
                _set_cell_text(cell, text)
            dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(dst_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-commit",
        default="",
        help="Full git SHA that contains OSNet_B2/ConvNeXtS_B2 (default: HEAD)",
    )
    args = parser.parse_args()
    commit = args.research_commit.strip() or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = generate(commit)
    print(f"Research commit: {commit}")
    print(f"HF bucket prefix: v3_50epoch_{commit[:7]}/<CONDITION>/seed<SEED>/")
    for path in paths:
        print("Wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
