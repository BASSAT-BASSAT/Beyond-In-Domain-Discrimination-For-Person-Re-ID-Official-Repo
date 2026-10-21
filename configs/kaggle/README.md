# Kaggle presets (one yaml per run)

Set **`CONFIG_YAML`** in `notebooks/01_kaggle_train.ipynb`. Each file uses `_BASE_CONFIG` to extend a full base recipe under `configs/`.

## Primary (recommended)

| File | Description |
|------|-------------|
| **`primary_sbs_r50_poolv2.yml`** | ImageNet → **SBS R50-IBN**, pool v2, CircleSoftmax + triplet (**loss study R0**) |

## Loss ablations (B0–B7)

Same pool/budget as primary; only head/loss changes.

| File | Condition |
|------|-----------|
| `loss_study_B0.yml` … `loss_study_B7.yml` | Linear / ArcFace / Sub-center / SphereFace2 variants |

Regenerate from `configs/experiments/loss_study_9.yaml` if the matrix changes.

## Other

| File | Description |
|------|-------------|
| `finetune_sbs_r50_msmt_zoo.yml` | MSMT zoo init (secondary; not primary held-out claims) |
| `fair_bot_r50_poolv2.yml` / `fair_agw_r50_poolv2.yml` | Appendix CNN baselines |
| `smoke_sbs_r50.yml` | Short debug run |

`OUTPUT_DIR` is the folder name under `/kaggle/working/output/`.
