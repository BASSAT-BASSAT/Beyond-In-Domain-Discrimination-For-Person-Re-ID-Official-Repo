# Kaggle presets (one yaml per run)

Each file uses `_BASE_CONFIG` to extend a full recipe under `configs/`. The paper notebooks load these presets directly; see [`notebooks/reid_loss_study_50epoch_production/README.md`](../../notebooks/reid_loss_study_50epoch_production/README.md).

## Primary (recommended)

| File | Description |
|------|-------------|
| **`primary_sbs_r50_poolv2.yml`** | ImageNet → Strong Baseline, Circle-softmax + triplet |

## Loss ablations

Same pool and schedule as the primary. Only the head and the triplet term change.

| File | Objective |
|------|-----------|
| `loss_study_B0.yml` | Linear softmax |
| `loss_study_B1.yml` | Linear softmax + triplet |
| `loss_study_B2.yml` | ArcFace |
| `loss_study_B3.yml` | ArcFace + triplet |
| `loss_study_B4.yml` | Sub-center ArcFace |
| `loss_study_B5.yml` | Sub-center ArcFace + triplet |
| `loss_study_B6.yml` | SphereFace2 |
| `loss_study_B7.yml` | SphereFace2 + triplet |

The filename suffix is the checkpoint key stored with the trained weights. Regenerate from `configs/experiments/loss_study_9.yaml` if the matrix changes.
