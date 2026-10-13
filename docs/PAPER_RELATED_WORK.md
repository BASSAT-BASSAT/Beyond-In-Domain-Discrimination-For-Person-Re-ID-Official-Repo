# Related work (ReID-FM positioning)

## A. Unified / foundation-style encoders

- **Instruct-ReID / OmniReID** (CVPR 2024) — multi-task ViT on many datasets including Market/MSMT in pretrain.
- **UniHCP** (CVPR 2023) — human-centric multi-task ViT; dedup vs eval sets.
- **AIO** — frozen large encoder + tokenizer for multimodal ReID.

**Gap:** We hold out Market/MSMT/MARS/Duke entirely and train a **real-time CNN** (SBS R50-IBN) on an audit-curated wild corpus.

## B. Multi-source domain generalization (closest protocol)

- **ADP**, **D2FEL**, mix-norm MS DG, LDL, DEX, DCAC, CLIP-based DG, **ReMix** (WACV 2025; cite for MOT-style unlabeled video).

## C. Practical baselines

- **SBS / AGW** (FastReID).
- **OSNet-AIN MSDC** — multi-dataset CNN checkpoint in our model zoo.

## D. Data / clothes / leakage

- **CCUP** (synthetic cloth-changing pretrain) vs LaST clothes-aware supervised ids.
- UniHCP dedup; embedding/dHash leakage checks for split design.

## Novelty summary

1. Audit-first **ReID-FM-Pool** with PRW excluded and MOT17/20 in the train mix.
2. **Leakage-aware cluster val split** (offline embedders + dHash).
3. **Held-out-domain** test matrix with deployable SBS encoder and explicit **accuracy–latency** Pareto.
