# Notebooks

| Notebook | Where | What |
|---|---|---|
| `00_prepare_pool.ipynb` | machine with the raw datasets | convert + audit the pool, package it into Kaggle Dataset folders |
| `01_kaggle_train.ipynb` | Kaggle, GPU T4 x2 | clone repo + FastReID, check attached datasets, DDP training with time guard, resume across sessions |
| `02_kaggle_evaluate.ipynb` | Kaggle, 1 GPU | evaluate the trained run and zero-shot baselines on all test + OOD sets, export weights |

## Multi-session training on Kaggle

1. Session 1: attach the pool datasets, set `CONFIG` / `RUN_NAME`, **Save Version (Run All)**.
   Training stops by itself before `TIME_BUDGET_H` with a checkpoint in
   `/kaggle/working/output/<RUN_NAME>`.
2. Session 2+: add the previous version's output as an input, set `RESTORE_FROM` to it
   (the run folder is found automatically up to 3 levels deep), save a new version.
3. When `model_final.pth` exists, run `02_kaggle_evaluate.ipynb` with `RUN_DIR` pointing at it.

Kaggle's `/kaggle/working` is limited to 20 GB; the run keeps only the newest
`REIDFM.KEEP_CHECKPOINTS` epoch checkpoints plus `model_best.pth`.
