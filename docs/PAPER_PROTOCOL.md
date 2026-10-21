# ReID-FM paper evaluation protocol

## Training pool (`pool_v2`)

- **Sources (11 domains):** CUHK03-NP, LaST, Occluded-PoseTrack, DeepChange, LPW, RPIfield, ORBench RGB, IUST, PRID2011, MOT17, MOT20.
- **Excluded from training:** PRW (Market campus overlap), Market-1501, MSMT17, MARS, DukeMTMC.
- **No separate OOD bucket:** generalization is measured only on held-out test domains.

## Splits

1. Prepare manifests: `python scripts/prepare_datasets.py --pool configs/datasets/pool_v2.yaml`
2. Build leakage-aware train/val: `python scripts/build_train_val_split.py --pool configs/datasets/pool_v2.yaml`
3. Fine-tune with `REIDFM.TRAIN_SPLIT` and in-training val on `pool/poolv2val` (cluster holdout, not official Market/MSMT test).

Per-source split rules (ORBench / PRID / RPIfield vs dHash default): see [DATASET_SPLITS.md](DATASET_SPLITS.md) and `splits/split_policies.json` after step 2.

## Final benchmarks (`--sets test`)

| Dataset | Role |
|---------|------|
| market1501 | Held-out test |
| msmt17 | Held-out test |
| mars | Held-out test (disclose Market campus overlap) |
| dukemtmc | Held-out test (optional if licensed copy missing) |
| occludedduke | Held-out test (Duke-derived; PGFA split) |

```bash
python scripts/evaluate.py --checkpoint outputs/<run>/model_best.pth --pool configs/datasets/pool_v2.yaml --sets test
```

Occluded-Duke: `python scripts/prepare_occluded_duke.py /path/to/DukeMTMC-reID.zip` then `prepare_datasets.py --datasets occludedduke`.

## Angular-loss study (main table)

- **Nine conditions** `R0` … `B7` on **SBS R50-IBN**, **ImageNet init only**, `pool_v2` + leakage split.
- Manifest: `configs/experiments/loss_study_9.yaml`
- Run: `python scripts/run_loss_study.py --screening` or `--all --seed 0`
- Aggregate: `python scripts/aggregate_loss_study.py --eval`

Literature ViT SOTA (e.g. AATransID) stays in related work; do not mix with the main nine-condition table.

## Fair CNN baselines (appendix)

- **BoT R50-IBN** and **AGW R50-IBN**, same pool/splits/budget as SBS scratch (`configs/experiments/paper_fair_baselines.yaml`).
- Run: `python scripts/run_fair_baselines.py`
- Aggregate: `python scripts/aggregate_paper_baselines.py --eval`

## Experiments (suggested)

| ID | Description |
|----|-------------|
| E0 | ImageNet-init SBS R50-IBN, zero-shot on held-out test battery |
| E1 | Fine-tune on `pool_v2` train split, eval on five test domains |
| Loss | Nine-condition matrix (`run_loss_study.py`) |
| Abl | Remove MOT sources / remove clothes-aware LaST ids / no leakage split |

## Efficiency (Pareto)

Report mAP vs **FPS**, latency (ms/image), and parameter count:

```bash
python scripts/benchmark_inference.py --model sbs_r50_ibn_msmt --checkpoint outputs/<run>/model_best.pth
```

Heavy embedders (DINOv2, etc.) are **offline only** for split/dedup audits, not deployed at inference.

## Backbone

FastReID **SBS R50-IBN** at 384×128 — same recipe for training and deployment.
