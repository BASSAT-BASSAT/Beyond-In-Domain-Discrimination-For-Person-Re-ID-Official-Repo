# `reidfm.engine`

## `ReidfmTrainer` (`trainer.py`)

FastReID's `DefaultTrainer` with the SBS recipe untouched (losses, augmentation, schedule,
freeze iterations, AMP, DDP, checkpoint format). Changes:

- **Train loader:** all pool sources in one `CommDataset`. `REIDFM.ITERS_PER_EPOCH` fixes a
  virtual epoch length (the sampler is infinite), so schedules and checkpoints don't depend on
  the pool size. Sampler: `DomainBalancedIdentitySampler` (or any FastReID sampler by name).
- **`test`:** the benchmark-protocol evaluator on `DATASETS.TESTS` (the pool's `val_sets`),
  main process only. Logs per-set R1 / mAP / mINP and `metric = (R1 + mAP) / 2`, plus `mean`.
- **Hooks (`hooks.py`):**
  - `MeanMetricCheckpointer`: `model_best.pth` tracks `mean/metric`; the best value is stored in
    every checkpoint so it survives resume.
  - `TimeGuardHook`: after each epoch, if another epoch (1.25x the slowest so far) would not fit
    in `REIDFM.TIME_BUDGET_H` since `$REIDFM_SESSION_START`, save and stop. Rank 0 decides and
    broadcasts, so DDP ranks exit together.
  - `CheckpointPruneHook`: keep only the newest `REIDFM.KEEP_CHECKPOINTS` `model_XXXX.pth` (Kaggle's
    `/kaggle/working` is 20 GB); never deletes the file named in `last_checkpoint`.
- **Resume:** epoch-granular via FastReID's `last_checkpoint` (`--resume`).

## Kaggle sessions (`resume.py`)

`restore_run_dir(src, dst)` copies a previous session's run folder (read-only input) into
the writable `OUTPUT_DIR`: `last_checkpoint`, the checkpoint it names, `model_best.pth`, and
small files (config, logs, metrics). `scripts/train.py --restore-from <dir>` calls it and then
resumes.
