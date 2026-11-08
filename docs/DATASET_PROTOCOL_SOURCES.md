# Dataset protocol audit (official sources vs reidfm)

This document records **what each benchmark’s authors define**, what **`reidfm` converters implement**, and whether they align. Citations were checked against publisher pages / papers / official repos (March 2026). **Always re-verify folder names on your downloaded zip** before paper numbers.

Legend:

| Status | Meaning |
|--------|---------|
| **Aligned** | Converter layout matches a fixed public train/query/gallery (or train/test JSON) from the source. |
| **Partial** | Core eval protocol matches; pool-specific extensions or modality subset differ. |
| **Mismatch** | Repo assumption differs from the paper/release (action required). |
| **N/A** | No static ReID train/test split (tracking / temporal protocol only). |

---

## Training sources

### CUHK03-NP — **Aligned**

| | |
|--|--|
| **Official** | CUHK03 **new protocol** (Zhong et al., CVPR 2017): 767 train IDs / 700 test IDs; Market-1501-style folders under `detected/` or `labeled/`: `bounding_box_train`, `query`, `bounding_box_test`. |
| **Sources** | [CUHK03-NP README](https://github.com/zhunzhong07/person-re-ranking/blob/master/CUHK03-NP/README.md), [CVPR 2017 paper](https://github.com/zhunzhong07/person-re-ranking) |
| **reidfm** | `convert_cuhk03np` → `market_style` on `detected/` (pool: `variant: detected`). |
| **In-training val** | Official test q/g is valid if you accept tuning on CUHK03 test (held-out paper tests remain Market/MSMT/…). |

---

### LaST — **Partial**

| | |
|--|--|
| **Official** | Train **5,000** IDs (71,248 imgs); **Val** 56 IDs (100 query + 21,279 gallery); **Test** 5,806 IDs (10,176 query + 125,353 gallery). Layout: `train/`, `val/query`, `val/gallery`, `test/query`, `test/gallery`. |
| **Sources** | [GitHub shuxjweb/last](https://github.com/shuxjweb/last), [arXiv:2105.15076](https://arxiv.org/abs/2105.15076) Table II |
| **reidfm** | `convert_last`: `train/` → train only; official val and test query/gallery stay in the manifest. |

---

### Occluded-PoseTrack-ReID — **Partial**

| | |
|--|--|
| **Official** | Occ-PTrack (Somers et al., ECCV 2024): 1,000 train IDs; test split with fixed query/gallery (20% most occluded queries per ID). Built from PoseTrack21 + released JSON. |
| **Sources** | [ECCV 2024 paper supp.](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10180-supp.pdf), [KPR repo](https://github.com/VlSomers/keypoint_promptable_reidentification) |
| **reidfm** | `convert_occposetrack`: uses `train/` + `val` or `test/`; if `query/` & `gallery/` exist, uses them; else **reidfm synthetic** q/g (`protocol: reidfm` in registry). |
| **Action** | Ship crops + official Occ-PTrack JSON layout; confirm manifest counts vs Table 4 in supp. (1000 / 1379 query / 1411 gallery IDs). |

---

### DeepChange — **Partial** (split choice)

| | |
|--|--|
| **Official** | Xu et al., ICCV 2023: **450** train / **150** val / **521** test IDs (disjoint). Val and test each have probe + gallery (~5 query boxes per tracklet). |
| **Sources** | [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_DeepChange_A_Long-Term_Person_Re-Identification_Benchmark_with_Clothes_Change_ICCV_2023_paper.pdf), [GitHub PengBoXiangShang/deepchange](https://github.com/PengBoXiangShang/deepchange) |
| **reidfm** | `pattern_dirs`: `train-set` → train; `val-set-*` + `test-set-*` → query/gallery (skips missing val folders if only test is present). |

---

### LPW — **Aligned**

| | |
|--|--|
| **Official** | Train: Scene 2 + Scene 3 (1,975 IDs). Test: Scene 1 — **view2 = probe**, view1 + view3 = gallery (756 / 1,072 IDs). |
| **Sources** | [LPW dataset page](https://liuyu.us/old/dataset/lpw/index.html), [AAAI 2018 LPW paper](https://doi.org/10.1609/aaai.v32i1.12305) |
| **reidfm** | `convert_lpw` defaults match (`train_scenes`, `test_scene`, `query_view`). |

---

### ORBench RGB — **Partial** (modality)

| | |
|--|--|
| **Official** | ReID5o / ORBench (NeurIPS 2025): 1,000 IDs; **600 train** (`train_annos.json`), **400 test** (`test_gallery_and_queries.json`); RGB gallery, multi-modal queries. |
| **Sources** | [GitHub Zplusdragon/ReID5o_ORBench](https://github.com/Zplusdragon/ReID5o_ORBench), [arXiv:2506.09385](https://arxiv.org/abs/2506.09385) §2.4 |
| **reidfm** | `orbench_rgb`: official JSON + files; pool `rgb_only: true` → **RGB-only** subset of full omni-modal protocol. |

---

### IUST PersonReID — **Mismatch** (likely)

| | |
|--|--|
| **Official** | Temporal split: first **75%** of annotated video time → train (1,193 IDs); last **25%** → test (654 IDs); separate **query** from test IDs (1,428 query imgs). Not defined as three static Market folders on the paper alone. |
| **Sources** | [arXiv:2412.18874](https://arxiv.org/abs/2412.18874) §4.1 Table 2, [GitHub ComputerVisionIUST/IUST_PersonReId](https://github.com/ComputerVisionIUST/IUST_PersonReId) (naming like Market-1501) |
| **reidfm** | `convert_iust` assumes **Market-style** `bounding_box_train` / `query` / `bounding_box_test` at dataset root — correct **only if** your release is repackaged that way (e.g. some mirrors/Kaggle). |
| **Action** | Validate manifests against paper Table 2; if using team CSVs (`iust_clean_train_candidates_v0`), train may be **train-only** until test packaging is wired. |

---

### PRID2011 — **Partial**

| | |
|--|--|
| **Official** | Multi-shot: persons **1–200** in both cams; eval **cam_a → cam_b** (query vs gallery); IDs 201+ are gallery distractors. |
| **Sources** | [TU Graz PRID11](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/prid11) |
| **reidfm** | Official q/g on numeric IDs; pool `include_train: true` adds **`tr{pid}`** multi-shot train rows — **project extension**, not in original benchmark. |
| **In-training val** | Official test q/g is valid; `tr*` rows are excluded from test pids by design. |

---

### RPIfield — **Mismatch** (protocol)

| | |
|--|--|
| **Official** | **Temporal** dynamic-gallery benchmark (CVPR 2018W); 112 known actors; evaluation uses timestamps — not a single static train/test folder split. |
| **Sources** | [CVPRW 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w36/Zheng_RPIfield_A_New_CVPR_2018_paper.pdf), [GitHub MZhengRPI/RPIfield](https://github.com/MZhengRPI/RPIfield) |
| **reidfm** | Default pool `rpifield.splits: train: [images]` is a **custom** crop layout, not the paper’s temporal eval. |
| **Action** | Use `structured_or_dhash` ~80/20 for pool training; do not claim official RPIfield benchmark numbers without implementing temporal protocol. |

---

## Held-out benchmarks

| Dataset | Official split | reidfm | Status |
|---------|----------------|--------|--------|
| **Market-1501** | 751 train / 750 test; DPM boxes | `convert_market1501` | **Aligned** — [Zheng et al. 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Zheng_Scalable_Person_Re-Identification_2015_CVPR_paper.pdf) |
| **MSMT17** | `list_train.txt` / `list_val.txt` / `list_query.txt` / `list_gallery.txt` | `convert_msmt17` | **Aligned** — [Wei et al. 2018](https://arxiv.org/abs/1711.08565) |
| **DukeMTMC-reID** | Market-style train/query/gallery | `convert_dukemtmc` | **Aligned** (dataset withdrawn; license/local copy only) |
| **Occluded-Duke** | PGFA query/gallery (no train in pool) | `convert_occluded_duke` | **Aligned** — [Miao et al. 2019](https://arxiv.org/abs/1908.06913) |

---

## Summary

| Dataset | Official static eval for in-training val? | reidfm today |
|---------|------------------------------------------|--------------|
| CUHK03-NP | Yes | Aligned |
| LaST | Yes (val **or** test) | Uses **test** by default |
| Occ-PoseTrack | Yes, if official crops+JSON | Partial if synthetic q/g |
| DeepChange | Yes (val **and** test) | **Test** only in converter; val omitted |
| LPW | Yes | Aligned |
| ORBench | Yes (RGB slice) | Partial (rgb_only) |
| IUST | Yes (temporal packaging) | **Verify** Market folders vs paper |
| PRID2011 | Yes (+ custom `tr*` train) | Partial |
| RPIfield | No static split | Custom / structured val |

---

## How to verify on your machine

```bash
python scripts/prepare_datasets.py --pool configs/datasets/pool_v2.yaml
```

Compare the manifests with the counts in the papers above. For DeepChange and IUST, cross-check the audit notes in `reports/`. The manuscript uses an identity holdout inside each source, and it does not train on Market-1501, MSMT17, DukeMTMC-reID, or Occluded-Duke.
