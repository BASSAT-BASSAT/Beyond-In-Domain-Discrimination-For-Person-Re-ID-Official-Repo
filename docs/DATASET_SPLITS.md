# Train / validation split policies (pool v2)

Held-out **test** benchmarks (Market, MSMT17, Duke, Occluded-Duke, MARS) are never used for
train/val. This document covers **in-pool** sources only.

## Summary

| Dataset | Official train | Official val | Official test / eval | ReID-FM train/val policy |
|---------|----------------|--------------|----------------------|---------------------------|
| ORBench RGB | `train_annos.json` (600 IDs) | None | `test_gallery_and_queries.json` | Train IDs = official train only; **custom** identity holdout val (`reidfm_identity_disjoint_val`) |
| PRID2011 | None (we add multi-shot `tr*` train rows) | None | cam_a query vs cam_b gallery | **Custom** identity holdout on `tr*` only; official test rows never in train/val |
| RPIfield | Not standardized | Not standardized | Temporal / growing gallery (paper) | **Custom** identity holdout on train-folder IDs; adjust `converters.rpifield.splits` to your copy |
| CUHK03-NP, LaST, Occ-PoseTrack, DeepChange, LPW, IUST, MOT17, MOT20 | Varies | None standardized for pool | Official eval where applicable | Optional `splits/datasets/<key>.json` from teammates; else **dHash fallback** |
| Other pool sources | Dataset-specific | — | Official query/gallery where applicable | `reidfm_dhash_cluster_val` (default) |

### Teammate split files

See [splits/datasets/README.md](../splits/datasets/README.md). Set `REIDFM_SPLITS_ROOT` on Kaggle if splits live in a separate dataset input.

Artifacts written by `scripts/build_train_val_split.py`:

- `splits/pool_v2_train_ids.json` / `pool_v2_val_ids.json`
- `splits/split_policies.json` — human-readable policy per source
- `splits/split_meta.json` — seed, fractions, per-dataset stats

## ORBench RGB

Converter: `reidfm.data.converters.orbench_rgb` (expects `train_annos.json` + optional test JSON under `ORBench/`).

- **Do not** place official test identities into the training manifest (converter namespaces test pids with `test` prefix).
- Validation identities are a **labeled custom** subset of official training IDs only.

## PRID2011

- Enable pool training with `include_train: true` (multi-shot, `tr{pid}`).
- Official benchmark remains query/gallery numeric IDs for optional OOD-style eval, not for pool train/val.

## RPIfield

- No official three-way split; verify folder names on your download and set `pool_v2.yaml` → `converters.rpifield.splits`.
- Splits are **by identity**, not by random images, to avoid reappearance leakage.
