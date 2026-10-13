# ReID-FM paper evaluation protocol

## Training pool (`pool_v2`)

- **Sources (11 domains):** CUHK03-NP, LaST, Occluded-PoseTrack, DeepChange, LPW, RPIfield, ORBench RGB, IUST, PRID2011, MOT17, MOT20.
- **Excluded from training:** PRW (Market campus overlap), Market-1501, MSMT17, MARS, DukeMTMC.
- **No separate OOD bucket:** generalization is measured only on held-out test domains.

## Splits

1. Prepare manifests: `python scripts/prepare_datasets.py --pool configs/datasets/pool_v2.yaml`
2. Build leakage-aware train/val: `python scripts/build_train_val_split.py --pool configs/datasets/pool_v2.yaml`
3. Fine-tune with `REIDFM.TRAIN_SPLIT` and in-training val on `pool/poolv2val` (cluster holdout, not official Market/MSMT test).

## Final benchmarks (`--sets test`)

| Dataset | Role |
|---------|------|
| market1501 | Unseen domain |
| msmt17 | Unseen domain |
| mars | Unseen domain |
| dukemtmc | Unseen domain (optional if licensed copy missing) |

```bash
python scripts/evaluate.py --checkpoint outputs/<run>/model_best.pth --pool configs/datasets/pool_v2.yaml --sets test
```

## Experiments (suggested)

| ID | Description |
|----|-------------|
| E0 | ImageNet-init SBS R50-IBN, zero-shot on four test domains |
| E1 | Fine-tune on `pool_v2` train split, eval on four test domains |
| Abl | Remove MOT sources / remove clothes-aware LaST ids / no leakage split |

## Efficiency (Pareto)

Report mAP vs **FPS**, latency (ms/image), and parameter count:

```bash
python scripts/benchmark_inference.py --model sbs_r50_ibn_msmt --checkpoint outputs/<run>/model_best.pth
```

Heavy embedders (DINOv2, etc.) are **offline only** for split/dedup audits, not deployed at inference.

## Backbone

FastReID **SBS R50-IBN** at 384×128 — same recipe for training and deployment.
