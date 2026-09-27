"""Export a trained run as an inference checkpoint + standalone FastReID yaml.

The output loads with plain FastReID (``get_cfg().merge_from_file`` + ``build_model``), so it
drops straight into the benchmark's ``FastReIDEncoder`` or any FastReID code:

* ``<weights_root>/<key>.pth``: ``{"model": state_dict}`` without the classifier / optimizer.
* ``configs/models/exported/<key>.yml``: the full resolved cfg, minus the REIDFM node and
  ``_BASE_`` chains, with ``MODEL.WEIGHTS`` pointing nowhere.

Examples:
    python scripts/export_weights.py --run outputs/sbs_r50ibn_msmt_v1 --key reidfm_sbs_r50_ibn_v1 --register
    python scripts/export_weights.py --checkpoint outputs/.../model_0029.pth --key tmp_export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from reidfm.fastreid_compat import ensure_fastreid  # noqa: E402
from reidfm.utils.paths import repo_path, weights_root  # noqa: E402

EXPORT_CFG_DIR = REPO_ROOT / "configs" / "models" / "exported"
TRAINED_REGISTRY = REPO_ROOT / "configs" / "models" / "trained.yaml"


def pick_checkpoint(run_dir: Path, which: str) -> Path:
    if which == "best" and (run_dir / "model_best.pth").is_file():
        return run_dir / "model_best.pth"
    if which in ("best", "final") and (run_dir / "model_final.pth").is_file():
        return run_dir / "model_final.pth"
    last = run_dir / "last_checkpoint"
    if last.is_file():
        return run_dir / last.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"no checkpoint found in {run_dir}")


def export_cfg(config_file: Path) -> tuple[str, dict]:
    """Plain FastReID yaml text + a few fields for the registry entry."""
    from reidfm.config import load_cfg

    cfg = load_cfg(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""
    data = yaml.safe_load(cfg.dump())
    data.pop("REIDFM", None)
    data.pop("_BASE_", None)
    # Pool dataset names are only registered by reidfm; keep the file loadable without it.
    data.setdefault("DATASETS", {})["NAMES"] = ["Market1501"]
    data["DATASETS"]["TESTS"] = ["Market1501"]
    info = {
        "backbone": cfg.MODEL.BACKBONE.NAME,
        "depth": cfg.MODEL.BACKBONE.DEPTH,
        "ibn": bool(cfg.MODEL.BACKBONE.WITH_IBN),
        "input_size": tuple(int(v) for v in cfg.INPUT.SIZE_TEST),
        "feat_dim": int(cfg.MODEL.BACKBONE.FEAT_DIM),
        "init_model": cfg.REIDFM.INIT_MODEL,
        "pool": cfg.REIDFM.POOL,
    }
    header = "# Exported by reidfm scripts/export_weights.py; standalone FastReID config.\n"
    return header + yaml.safe_dump(data, sort_keys=False), info


def strip_state(checkpoint: Path) -> tuple[dict, dict]:
    import torch

    from reidfm.models.init import is_classifier_key

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    kept = {k: v for k, v in state.items() if not is_classifier_key(k)}
    meta = {k: raw[k] for k in ("epoch", "iteration", "best_metric") if isinstance(raw, dict) and k in raw}
    return kept, meta


def backbone_label(info: dict) -> str:
    name = info["backbone"]
    if "resnet" in name:
        label = f"R{info['depth'].rstrip('x')}"
    elif "resnest" in name:
        label = f"S{info['depth'].rstrip('x')}"
    elif "osnet" in name:
        label = "OSNet-AIN"
    else:
        label = name
    return label + ("-IBN" if info["ibn"] else "")


def register(entry: dict) -> None:
    data = yaml.safe_load(TRAINED_REGISTRY.read_text(encoding="utf-8")) if TRAINED_REGISTRY.is_file() else None
    models = list((data or {}).get("models") or [])
    models = [m for m in models if m.get("key") != entry["key"]] + [entry]
    text = TRAINED_REGISTRY.read_text(encoding="utf-8") if TRAINED_REGISTRY.is_file() else ""
    header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    TRAINED_REGISTRY.write_text(header + yaml.safe_dump({"models": models}, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="training output dir")
    source.add_argument("--checkpoint", help="a specific checkpoint (.pth)")
    parser.add_argument("--which", choices=("best", "final", "last"), default="best", help="checkpoint to take from --run")
    parser.add_argument("--config-file", default=None, help="default: config.yaml in the run dir")
    parser.add_argument("--key", required=True, help="registry key / file stem, e.g. reidfm_sbs_r50_ibn_v1")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out-dir", default=None, help="weights folder (default: $REIDFM_WEIGHTS_ROOT)")
    parser.add_argument("--url", default=None, help="public download URL for the registry entry")
    parser.add_argument("--register", action="store_true", help="add/replace the entry in configs/models/trained.yaml")
    args = parser.parse_args()

    ensure_fastreid()
    import torch

    from reidfm.utils.provenance import run_provenance, sha256_file, write_json

    checkpoint = repo_path(args.checkpoint) if args.checkpoint else pick_checkpoint(repo_path(args.run), args.which)
    config_file = repo_path(args.config_file) if args.config_file else checkpoint.parent / "config.yaml"
    cfg_text, info = export_cfg(config_file)
    state, meta = strip_state(checkpoint)

    out_dir = Path(args.out_dir) if args.out_dir else weights_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / f"{args.key}.pth"
    torch.save({"model": state, "reidfm_export": {"source": str(checkpoint), **meta, **info}}, weights)
    EXPORT_CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_out = EXPORT_CFG_DIR / f"{args.key}.yml"
    cfg_out.write_text(cfg_text, encoding="utf-8")
    digest = sha256_file(weights)

    bb = backbone_label(info)
    entry = {
        "key": args.key,
        "label": args.label or f"reidfm SBS-{bb} ({Path(str(info['pool'])).stem})",
        "family": "SBS",
        "backbone": bb,
        "trained_on": f"reidfm {Path(str(info['pool'])).stem}",
        "init": "trained",
        "config_file": cfg_out.relative_to(REPO_ROOT).as_posix(),
        "weights_file": weights.name,
        "weights_url": args.url,
        "sha256": digest,
        "input_size": list(info["input_size"]),
        "feat_dim": info["feat_dim"],
        "source": "reidfm",
        "notes": f"init {info['init_model']}; exported from {checkpoint.name} (epoch {meta.get('epoch', '?')})",
    }
    write_json(weights.with_suffix(".json"), {"entry": entry, "export": meta, "provenance": run_provenance()})
    if args.register:
        register(entry)
    print(f"weights : {weights}\nconfig  : {cfg_out}\nsha256  : {digest}")
    print("registered in configs/models/trained.yaml" if args.register else "(not registered; pass --register)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
