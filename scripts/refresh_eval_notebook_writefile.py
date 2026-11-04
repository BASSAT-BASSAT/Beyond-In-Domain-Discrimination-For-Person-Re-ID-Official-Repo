"""Refresh the held-out eval notebook %%writefile hotfix cell from scripts/evaluate_unseen_reidraw.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks/reid_loss_study_50epoch_production/ReID_Paper_HeldOut_Evaluation.ipynb"
eval_path = ROOT / "scripts/evaluate_unseen_reidraw.py"

writefile_source = [
    "%%writefile /kaggle/working/Person-Re-ID-Foundational-model-eval/scripts/evaluate_unseen_reidraw.py\n",
]
writefile_source.extend(eval_path.read_text(encoding="utf-8").splitlines(keepends=True))

hotfix_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Hotfix (only if `git pull` is still on an old commit)\n",
        "\n",
        "Run **once** before §4 Evaluate. Overwrites `scripts/evaluate_unseen_reidraw.py` so eval never loads `pool_v2.yaml`. "
        "Skip after `main` @ `afcf896+`.\n",
    ],
}
hotfix_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": writefile_source,
}

nb = json.loads(nb_path.read_text(encoding="utf-8"))
out: list = []
i = 0
cells = nb["cells"]
while i < len(cells):
    src = "".join(cells[i].get("source", []))
    if "### Hotfix" in src and "pool_v2" in src:
        out.append(hotfix_md)
        out.append(hotfix_cell)
        i += 1
        while i < len(cells):
            nxt = "".join(cells[i].get("source", []))
            if nxt.lstrip().startswith("## 1."):
                break
            if (
                "### Hotfix" in nxt
                or "_SETUP_EVAL_CFG" in nxt
                or ("%%writefile" in nxt and "evaluate_unseen_reidraw" in nxt)
            ):
                i += 1
                continue
            break
        continue
    out.append(cells[i])
    i += 1

nb["cells"] = out
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Updated", nb_path, "hotfix lines", len(writefile_source))
