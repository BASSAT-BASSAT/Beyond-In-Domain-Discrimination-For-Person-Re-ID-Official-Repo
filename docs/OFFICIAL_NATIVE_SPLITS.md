# Official native train / official-test-as-val (pool v2)

Use pool **`configs/datasets/pool_v2_official_val.yaml`** (`split_mode: official_native`).

**Protocol provenance (official sites vs our converters):** see **[DATASET_PROTOCOL_SOURCES.md](DATASET_PROTOCOL_SOURCES.md)** — read this before claiming “official” splits in the thesis.

| Step | Command |
|------|---------|
| Prepare manifests | `python scripts/prepare_datasets.py --pool configs/datasets/pool_v2_official_val.yaml` |
| Build train-id filter + ratios | `python scripts/build_official_native_splits.py` |
| Train | `REIDFM.POOL` = same yaml, `STUDY_MODE: false`, `TRAIN_SPLIT: splits/pool_v2_train_ids.json` |

**Training:** 100% of each dataset’s official **train** identities (`manifest.train`).

**In-training validation:** official **query/gallery** for the eight datasets in `val_sets` (FastReID `DATASETS.TESTS`).

**Paper held-out test** (Market, MSMT, MARS, Duke, Occ-Duke) is unchanged.

## Per-dataset summary (reference ratios)

Counts are **identity / image** shares for **that dataset only** (not pooled across domains). After manifests exist, re-run `build_official_native_splits.py` for exact numbers in `splits/official_native_ratios.json`.

| Dataset | Official train | Official in-training val | Train ID share (typical) | Notes |
|---------|----------------|---------------------------|--------------------------|--------|
| **CUHK03-NP** | `bounding_box_train` | query + `bounding_box_test` | ~767 / (767+700) ≈ **52%** IDs | 140 query imgs; 700 gallery IDs |
| **LaST** | `train/` | **`val` + `test`** query/gallery | train ≫ val+test | `eval_splits: [val, test]` in official pool |
| **Occ-PoseTrack** | `train/` | **`val` + `test`** q/g when present | dataset-specific | Synthetic q/g only if subfolders missing |
| **DeepChange** | `train-set` | **`val-set` + `test-set`** q/g | 450 / 150 / 521 IDs | Val folders skipped if absent on disk |
| **LPW** | scen2 + scen3 | scen1 (view2 query, other views gallery) | ~2/3 scenes train | Video tracklets |
| **ORBench RGB** | 600 IDs `train_annos` | `test_gallery_and_queries.json` | 600 train vs test IDs | Test pids prefixed `test` in manifest |
| **IUST** | Market-style train | query + gallery | similar to Market-style sets | Restricted campus |
| **PRID2011** | 200 IDs `tr*` multi-shot | cam_a query vs cam_b gallery (200 IDs) | **50% / 50%** ID count | `tr1` vs `1` namespaces disjoint |
| **MOT17** | MOT train sequences (FRCNN) | — | **100%** train | No public ReID test GT |
| **MOT20** | MOT train sequences | — | **100%** train | Same as MOT17 |
| **RPIfield** | `images/` (default layout) | — | **100%** train | No default q/g; add converter splits to enable val |

## MOT, RPIfield (no public test)

Cascade (~**80% / 20%** train identities, `train_fraction=0.8`):

1. **Camera / scene groups** — MOT: by sequence (`camid`); RPIfield: by folder camera.
2. **Identity holdout** — if a single camera/scene, shuffle whole person IDs (~80/20).
3. **dHash clusters** — perceptual-hash grouping if still stuck (e.g. one identity).

Held-out val identities are exported to `pool/poolv2val` and evaluated as **`PoolV2ValM`** alongside official `val_sets` (see `apply_pool`).

MOTChallenge **test** GT remains unavailable; this val is only for in-training monitoring.

## vs paper cluster mode

`pool_v2.yaml` (`split_mode: cluster`) keeps ~**85% / 15%** identity holdout + `pool/poolv2val` for the nine-condition loss study (`STUDY_MODE: true`).
