# Configs

## Kaggle / ablations (pick one file)

Paper runs are the Kaggle notebooks in [`notebooks/reid_loss_study_50epoch_production/`](../notebooks/reid_loss_study_50epoch_production/README.md). The presets below are what those notebooks load.

## Base recipes (used via `_BASE_CONFIG` overlays)

These full yamls are the source of truth for hyperparameters. Do not delete; kaggle presets layer on top.

| File | Role |
|------|------|
| `scratch_sbs_r50ibn_imagenet.yml` | ImageNet Strong Baseline, the paper backbone |
| `smoke_sbs_r50ibn.yml` | Local smoke test on `pool_smoke` |

CLI example: `python scripts/train.py --config-file configs/kaggle/primary_sbs_r50_poolv2.yml`

## Pools (`datasets/`)

| File | Use |
|------|-----|
| `pool_v2.yaml` | The nine training sources and the four held-out benchmarks |
| `pool_smoke.yaml` | Tiny Market + MSMT pool for local smoke tests |

## Experiments manifest (`experiments/`)

- `loss_study_9.yaml` — the eight Strong Baseline objectives
- `backbone_study.yaml` — Circle-softmax + triplet and ArcFace on OSNet and ConvNeXt-S

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
