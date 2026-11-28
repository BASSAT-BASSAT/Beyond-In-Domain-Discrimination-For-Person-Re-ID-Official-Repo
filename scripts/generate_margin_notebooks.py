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
CHECKOUT = 'checked([\n    "git", "-C", str(ROOT),\n    "checkout", "--quiet", "--detach", RESEARCH_COMMIT\n])'
FETCH_IF_MISSING = (
    "# Camera-ready: a clone made before the commit was pushed (same session, cell re-run) lacks it.\n"
    "if subprocess.run([\"git\", \"-C\", str(ROOT), \"cat-file\", \"-e\", RESEARCH_COMMIT + \"^{commit}\"],\n"
    "                  capture_output=True).returncode != 0:\n"
    "    checked([\"git\", \"-C\", str(ROOT), \"fetch\", \"--quiet\", \"origin\",\n"
    "             \"+refs/heads/*:refs/remotes/origin/*\"])\n"
    "assert subprocess.run([\"git\", \"-C\", str(ROOT), \"cat-file\", \"-e\", RESEARCH_COMMIT + \"^{commit}\"],\n"
    "                      capture_output=True).returncode == 0, (\n"
    "    f\"Commit {RESEARCH_COMMIT} is not on GitHub yet: push branch camera-ready-margin-study first\")\n\n"
)


TQDM_IMPORT = "from tqdm.std import tqdm\n"
SAFE_WRITE = (
    "from tqdm.std import tqdm\n"
    "# Camera-ready fix: the hf-safe-backup thread uploads through huggingface_hub, which opens tqdm bars;\n"
    "# tqdm's write iterating tqdm._instances at the same moment raised 'Set changed size during iteration'\n"
    "# and killed this monitor. Disable hub progress bars and never let a console write crash the run.\n"
    "os.environ[\"HF_HUB_DISABLE_PROGRESS_BARS\"] = \"1\"\n"
    "try:\n"
    "    from huggingface_hub.utils import disable_progress_bars as _hf_disable_bars\n"
    "    _hf_disable_bars()\n"
    "except Exception:\n"
    "    pass\n\n"
    "def _safe_write(message):\n"
    "    for _ in range(3):\n"
    "        try:\n"
    "            tqdm.write(message)\n"
    "            return\n"
    "        except RuntimeError:\n"
    "            time.sleep(0.05)\n"
    "    print(message, flush=True)\n"
)


BUCKET_LINE = 'HF_BUCKET_ID = "reid-research-hub/reid-loss-study-live"\n'
TOP_ANCHOR = "ENABLE_MID_EPOCH_CHECKPOINT = True\n"
TOP_BUCKET_CONFIG = (
    "ENABLE_MID_EPOCH_CHECKPOINT = True\n"
    "# Checkpoint bucket. Keep the SAME bucket for every session of one CONDITION/SEED, otherwise\n"
    "# the next session cannot find the previous checkpoint and restarts from step 0.\n"
    "HF_BUCKET_ID = \"reid-research-hub/reid-loss-study-live\"\n"
    "# \"public\": a public bucket is accepted (a new bucket is created public); an existing private one\n"
    "#           also works and is left private. \"private\": refuse anything that is not private.\n"
    "# Uploaded: model/optimizer checkpoints, JSON step pointers, training-loss metrics. No images,\n"
    "# no manifests, no dataset paths, no tokens.\n"
    "BUCKET_VISIBILITY = \"public\"\n"
    "assert BUCKET_VISIBILITY in (\"public\", \"private\"), BUCKET_VISIBILITY\n"
)
OLD_ENSURE = (
    '    def ensure_private_repository(self):\n'
    '        """Retain the old call interface but ensure a private Bucket instead."""\n'
    '        create_bucket(\n'
    '            bucket_id=self.bucket_id, private=True,\n'
    '            exist_ok=True, token=self.token,\n'
    '        )\n'
    '        info = bucket_info(self.bucket_id, token=self.token)\n'
    '        if not info.private:\n'
    '            raise RuntimeError("Refusing to save private ReID checkpoints to a public bucket")\n'
    '        print("[PASS] Private HF Storage Bucket available (no Git history)")\n'
)
NEW_ENSURE = (
    '    def ensure_private_repository(self):\n'
    '        """Ensure the Bucket exists with the visibility chosen by BUCKET_VISIBILITY (top cell)."""\n'
    '        want_private = BUCKET_VISIBILITY == "private"\n'
    '        create_bucket(\n'
    '            bucket_id=self.bucket_id, private=want_private,\n'
    '            exist_ok=True, token=self.token,\n'
    '        )\n'
    '        info = bucket_info(self.bucket_id, token=self.token)\n'
    '        is_private = bool(getattr(info, "private", False))\n'
    '        if want_private and not is_private:\n'
    '            raise RuntimeError(\n'
    '                f"BUCKET_VISIBILITY=\'private\' but {self.bucket_id} is public; "\n'
    '                "change BUCKET_VISIBILITY to \'public\' in the top cell to allow it"\n'
    '            )\n'
    '        kind = "private" if is_private else "PUBLIC"\n'
    '        print(f"[PASS] {kind} HF Storage Bucket {self.bucket_id} available (no Git history)")\n'
    '        if not is_private:\n'
    '            print("[NOTE] Checkpoints and metrics in this bucket are publicly downloadable")\n'
)


def transform(text: str, *, src_cond: str, dst_cond: str, src_commit: str, dst_commit: str, margin: float) -> str:
    out = text.replace(src_commit, dst_commit).replace(src_commit[:7], dst_commit[:7])
    # backbone notebooks name their clone folder after an older commit; keep it consistent
    out = out.replace("Person-Re-ID-Foundational-model-b69c47a", f"Person-Re-ID-Foundational-model-{dst_commit[:7]}")
    if CHECKOUT in out:
        out = out.replace(CHECKOUT, FETCH_IF_MISSING + CHECKOUT, 1)
    if TOP_ANCHOR in out and BUCKET_LINE in out:              # top config cell
        out = out.replace(BUCKET_LINE, "", 1).replace(TOP_ANCHOR, TOP_BUCKET_CONFIG, 1)
        out = out.replace("WRITE permission for the private HF organization Bucket",
                          "WRITE permission for the HF Bucket in HF_BUCKET_ID")
    if OLD_ENSURE in out:
        out = out.replace(OLD_ENSURE, NEW_ENSURE, 1)
    out = out.replace("**PRIVATE HF Storage Bucket** `reid-research-hub/reid-loss-study-live`",
                      "**HF Storage Bucket** set by `HF_BUCKET_ID` / `BUCKET_VISIBILITY` in the top cell")
    out = out.replace("model weights remain on the private HF Bucket", "model weights remain on the HF Bucket")
    if TQDM_IMPORT in out and "tqdm.write(" in out:
        out = out.replace("tqdm.write(", "_safe_write(")
        out = out.replace(TQDM_IMPORT, SAFE_WRITE, 1)
    out = re.sub(rf"(?<![\w]){re.escape(src_cond)}(?![\w])", dst_cond, out)
    if ANCHOR in out:
        out = out.replace(
            ANCHOR,
            "_EXPECTED_MARGIN = {'m025': 0.25, 'm050': 0.5}[CONDITION.rsplit('_', 1)[-1]]  # from the top-cell CONDITION\n"
            "assert abs(float(cfg.MODEL.HEADS.MARGIN) - _EXPECTED_MARGIN) < 1e-9, (CONDITION, cfg.MODEL.HEADS.MARGIN)\n"
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
                assert body.count("cat-file") == 2, "fetch-if-missing guard not inserted"
                assert "_safe_write(" in body and body.count("tqdm.write(") == 1, "tqdm race fix not applied"
                assert "Refusing to save private ReID checkpoints" not in body, "old private-only check left"
                assert body.count("BUCKET_VISIBILITY = ") == 1 and body.count("HF_BUCKET_ID = ") == 1, "bucket config"
                assert "PRIVATE HF Storage Bucket" not in body and "private HF Bucket" not in body, "stale private wording"
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
