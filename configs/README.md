# Configs

## Kaggle / ablations (pick one file)

All paper and ablation runs for teammates: **[`kaggle/README.md`](kaggle/README.md)** — set `CONFIG_YAML` in `notebooks/01_kaggle_train.ipynb`.

## Base recipes (used via `_BASE_CONFIG` overlays)

These full yamls are the source of truth for hyperparameters. Do not delete; kaggle presets layer on top.

| File | Role |
|------|------|
| `scratch_sbs_r50ibn_imagenet.yml` | ImageNet SBS R50-IBN, pool v2 (primary paper backbone) |
| `scratch_bot_r50ibn_imagenet.yml` / `scratch_agw_r50ibn_imagenet.yml` | Fair appendix CNN baselines |
| `finetune_sbs_r50ibn_msmt.yml` | Optional MSMT zoo init (secondary / not held-out primary) |
| `smoke_sbs_r50ibn.yml` | Local CI smoke on `pool_smoke` |

CLI example: `python scripts/train.py --config-file configs/kaggle/primary_sbs_r50_poolv2.yml`

## Pools (`datasets/`)

| File | Use |
|------|-----|
| `pool_v2.yaml` | **Paper training pool** (11 sources; held-out test battery in yaml) |
| `pool_smoke.yaml` | Tiny Market + MSMT for local smoke tests |

## Experiments manifest (`experiments/`)

- `loss_study_9.yaml` — nine-condition matrix for `scripts/run_loss_study.py` (local/CI)
- `paper_fair_baselines.yaml` — BoT/AGW runner manifest

## Models (`models/`)

- `osnet_ain_x1_0.yml` — OSNet-AIN MSDC eval recipe
- `trained.yaml` — registry entries from `export_weights.py --register`

## `REIDFM` node (common keys)

| Key | Meaning |
|-----|---------|
| `POOL` | `configs/datasets/pool_v2.yaml` → `DATASETS.NAMES` |
| `INIT_MODEL` | Registry key → weights or ImageNet pretrain |
| `TRAIN_SPLIT` | `splits/pool_v2_train_ids.json` after `build_train_val_split.py` |
| `STUDY_MODE` | Fail-fast splits + protocol checks |
| `TIME_BUDGET_H` | Kaggle session guard (also `$REIDFM_TIME_BUDGET_H`) |

Relative `OUTPUT_DIR` → `$REIDFM_OUTPUT_ROOT`.
