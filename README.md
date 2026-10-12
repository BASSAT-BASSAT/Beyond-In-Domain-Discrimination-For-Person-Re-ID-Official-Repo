# Domain-Generalizable Person Re-Identification (DG-ReID)

**Multi-source ReID research | Dataset auditing | Unified data pipeline**

This project investigates whether **supervised fine-tuning across diverse source datasets** can improve Person Re-Identification (ReID) performance on **unseen domains**, compared with pretrained and single-source baselines.

> **Status — 12 October 2026:** Dataset auditing, local candidate cleaning, and Unified Manifest v1 are complete. Training protocols, cross-dataset leakage clearance, fine-tuning, and final evaluation are still pending.

## Progress at a Glance

| Milestone | Status |
| --- | --- |
| Datasets audited | **10** |
| Locally cleaned candidate sources | **9** |
| Candidate images | **940,809** |
| Source-scoped identities | **11,297** |
| Unified schema | **19 columns** |
| Dataset adapters | **9 validated** |
| Master Unified Manifest v1 | **Validated** |
| Multi-source fine-tuning | Not started |

The identity count is **source-scoped**; it does not establish that every person is unique across datasets. These are **candidate pools**, not an approved final training set.

## Audited Datasets

| Candidate source | Images | IDs |
| --- | ---: | ---: |
| CUHK03-NP (detected) | 7,365 | 767 |
| IUST-PersonReID | 72,393 | 1,193 |
| LaST | 71,248 | 5,000 |
| DeepChange | 75,009 | 450 |
| LPW | 418,447 | 1,975 |
| Occluded-PoseTrack-ReID | 17,898 | 1,000 |
| ORBench RGB | 26,932 | 600 |
| PRID2011 | 40,033 | 200 |
| RPIfield | 211,484 | 112 |
| **Total** | **940,809** | **11,297** |

**PRW** was audited but excluded from the proposed sources: **3,886 of 5,663 PRW source video frames (68.62%)** were found to overlap Market1501 raw frames. PRID2011 and RPIfield still require defined supervised train/validation splits.

## Data Pipeline

```text
Dataset audits & candidate manifests
              ↓
   9 dataset-specific adapters
              ↓
     Unified Schema v1 (19 fields)
              ↓
     Per-dataset Unified CSVs
              ↓
        Master Unified CSV
              ↓
     Independent validation
```

The schema preserves dataset identity, relative image path, source-scoped person ID, camera/view semantics, video/tracklet/frame metadata where verified, and source provenance. Missing or ambiguous metadata is **not invented**.

**Validated master:** `unified/v1/all_sources_unified_v1.csv`

**SHA-256:** `4fba9e5578b19679275f061dc97740518d300830a0306b725c39e0dfc15e34ce`

## Repository Structure

```text
DG_ReID_Audit/
├── README.md
├── .gitignore
└── scripts/
    ├── CUHK03_NP/           # Dataset audit scripts
    ├── DeepChange/
    ├── LPW/
    ├── ...                 # Other dataset audits
    └── unified/
        ├── adapters/       # 9 dataset adapters
        ├── schema.py
        ├── build_unified.py
        ├── merge_unified.py
        └── validate_master.py
```

`datasets/`, `manifests/`, `reports/`, and `unified/` are **local-only** and excluded from Git. Reproducing the data pipeline requires separately authorized access to these files. Raw data and official evaluation splits are not modified.

## Run Validation

From the project root in the prepared Lightning Studio workspace:

```bash
python -m scripts.unified.validate_master
```

The last verified run reported **940,809 rows checked — MASTER VALIDATION: PASS**. The validator expects the frozen local v1 files and checksum; this is not a standalone download-and-run package.

## Next Steps

1. **P3:** Define identity-disjoint training/validation protocols.
2. **P4:** Check source–source and source–target leakage (exact, pixel, near-duplicate, and acquisition overlap).
3. **P5–P6:** Freeze eligible sources, class labels, and balanced sampling.
4. **P7:** Compare pretrained, single-source, and multi-source fine-tuning.
5. **P8:** Evaluate on unseen **Market1501** (image-based) and **MARS** (tracklet-based), reporting mAP and Rank-1.

**Research note:** Successful manifest validation does not establish cross-dataset independence or model performance. Unseen target test sets must remain outside supervised training and undocumented model selection.
