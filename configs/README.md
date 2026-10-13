# Configs

## Training (FastReID yaml)

Each training config is **self-contained** (no `_BASE_` chain): model, losses, augmentation,
sampler, solver/schedule, evaluation, checkpointing and the `REIDFM` run settings are all written
out and commented in the one file. Edit the file, or override from the CLI.

| File | Use |
|---|---|
| `finetune_sbs_r50ibn_msmt.yml` | **E2 (main):** SBS-R50-IBN from the FastReID MSMT17 checkpoint, LR 2.5e-4, 30 epochs (~1 Kaggle session). |
| `finetune_sbs_r101ibn_msmt.yml` | E3: SBS-R101-IBN from MSMT17, ~2 sessions. |
| `scratch_sbs_r50ibn_imagenet.yml` | E1 (control): ImageNet R50-IBN, LR 3.5e-4, 60 epochs. |
| `smoke_sbs_r50ibn.yml` | tiny local smoke test on `pool_smoke` (batch 32, 3 x 10 iters). |
| `models/osnet_ain_x1_0.yml` | OSNet-AIN (MSDC checkpoint), 256x128, same recipe. |

To start a new experiment, copy the closest file and change what you need. Keys not listed fall
back to `fast-reid/fastreid/config/defaults.py` (mostly transformer/distillation options that
the ResNet/OSNet SBS recipe does not use).

The `REIDFM:` node:

| Key | Default | Meaning |
|---|---|---|
| `POOL` | `configs/datasets/pool_v2.yaml` | fills `DATASETS.NAMES` (sources); val from `REIDFM.VAL_MANIFEST` when `val_sets` is empty |
| `SOURCES`, `VAL_SETS` | `[]` | non-empty lists override the pool's (OOD sets are still rejected as sources) |
| `SAMPLER_WEIGHT_MODE` | `""` | `sqrt_ids` / `ids` / `uniform`; empty = the pool's |
| `DOMAIN_WEIGHTS` | `[]` | `"key=multiplier"` entries applied on top of the pool's domain weights |
| `SEED` | `-1` | fixed RNG seed when >= 0 (resumed sessions then replay the same sampling order) |
| `INIT_MODEL` | `""` | model registry key; sets `MODEL.WEIGHTS` (converted) or ImageNet `PRETRAIN` |
| `ITERS_PER_EPOCH` | `0` | virtual epoch length (0 = one pass over the pool) |
| `TIME_BUDGET_H` | `11.0` | stop cleanly before this many hours (0 = off) |
| `TIME_GUARD_SAFETY` | `1.25` | stop when remaining time < slowest epoch x this |
| `KEEP_CHECKPOINTS` | `2` | epoch checkpoints kept on disk |
| `EVAL_NUM_FRAMES` | `8` | frames per tracklet at validation |
| `VAL_MAX_QUERIES` | `0` | subsample validation queries (0 = all) |
| `EVAL_AMP`, `EVAL_WORKERS` | `True`, `4` | validation feature extraction |

Precedence, lowest to highest: yaml, `$REIDFM_TIME_BUDGET_H` (budget only), trailing CLI pairs
(`python scripts/train.py --config-file ... SOLVER.BASE_LR 0.0002`), then the dedicated flags
(`--pool`, `--model`, `--output-dir`, `--time-budget-h`). Relative `OUTPUT_DIR`s are placed
under `$REIDFM_OUTPUT_ROOT`.

## Pools (`datasets/`)

- `pool_v2.yaml`: paper protocol (11 train domains, 4 test domains, no OOD bucket)
- `pool_v1.yaml`: deprecated legacy pool
  options (caps, strides, visibility filters).
- `pool_smoke.yaml`: Market + capped MSMT for local testing.

## Models (`models/`)

- `trained.yaml`: registry entries for models trained here (`export_weights.py --register`).
- `exported/<key>.yml`: standalone FastReID configs written by `export_weights.py`.
