# ReID Loss Study — 27 Kaggle Production Notebooks (50 epochs)

This folder contains **27 independent Kaggle notebooks**, organized by **9 loss-study conditions** (`R0`, `B0`–`B7`) and **3 seeds** (`0`, `1`, `2`) per condition.

```text
notebooks/reid_loss_study_50epoch_production/
├── R0/  [Seed0, Seed1, Seed2]
├── B0/  [Seed0, Seed1, Seed2]
├── B1/  [Seed0, Seed1, Seed2]
├── B2/  [Seed0, Seed1, Seed2]
├── B3/  [Seed0, Seed1, Seed2]
├── B4/  [Seed0, Seed1, Seed2]
├── B5/  [Seed0, Seed1, Seed2]
├── B6/  [Seed0, Seed1, Seed2]
└── B7/  [Seed0, Seed1, Seed2]
```

Each folder contains:

- `ReID_<CONDITION>_Seed0_50Epoch_Production.ipynb`
- `ReID_<CONDITION>_Seed1_50Epoch_Production.ipynb`
- `ReID_<CONDITION>_Seed2_50Epoch_Production.ipynb`

## Experiment contract

- Research commit: `56923ec3af05abf0c56b9dfe608b9c4b628bdd62`
- FastReID commit: `c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d`
- **50 epochs × 2,000 training steps = 100,000 optimizer steps** per notebook
- **2× T4 GPUs**, DDP, global batch 128
- Evaluate every 5 epochs with the full V3 evaluation protocol
- Requires Kaggle input dataset `a7medlabib/reid-fine-tunning-datasets-v3`
- Kaggle Secrets: `GITHUB_TOKEN` (private source repo access) and `HF_TOKEN` (private Bucket write access)

## Where artifacts belong

**GitHub** stores these lightweight `.ipynb` source notebooks and documentation.

**Hugging Face Storage Bucket**, not GitHub, stores live model checkpoints:

`reid-research-hub/reid-loss-study-live/v3_50epoch_56923ec/<CONDITION>/seed<SEED>/`

Each experiment overwrites its rolling `slot_a.pth`, `slot_b.pth`, and `model_best.pth` there, without creating a Git commit per backup. Do not commit `.pth`, `.pt`, `.ckpt`, datasets, tokens, or generated run outputs into this GitHub folder.

Different condition/seed notebooks can be run concurrently, but do not start two active sessions for **the same condition and seed** at the same time. Every session verifies its resume metadata and checkpoint checksum.

## Repository integration

Copy the **`reid_loss_study_50epoch_production`** folder under the existing repository's `notebooks/` directory. Existing notebooks and project files should remain unchanged. Commit the new folder in a dedicated Git commit or pull request for team review.

> Packaging only: these notebooks have not been executed again here on Kaggle. The source notebooks previously passed offline source/AST/static-contract tests; run an online checkpoint upload/restore smoke test before trusting a new production run.
