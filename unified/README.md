# Unified ReID Data Pipeline — v1

## Overview

This pipeline standardizes nine audited and locally cleaned ReID source candidate pools into a consistent 19-column metadata schema. It provides reproducible, source-aware inputs for supervised multi-source Domain Generalization (DG) experiments.

**Important:** The source candidate inventory is not the same as the final training split. P3 source validation protocols and a provisional P5 global handoff have now been generated, but **P4 cross-dataset leakage clearance is still pending**. No fine-tuning has started.

## P2 — Validated Candidate Metadata

| Metric | Result |
|---|---:|
| Source candidate datasets | 9 |
| Candidate images (before P3) | 940,809 |
| Source-scoped identity labels (before P3) | 11,297 |
| Unified metadata columns | 19 |
| Dataset adapters | 9 |
| Independent master metadata validation | PASS |

The 11,297 labels are summed within their respective source namespaces; they are **not** a verified count of distinct physical people across datasets.

### Unified metadata pipeline

1. Read dataset-specific, cleaned candidate manifests.
2. Convert each source with its dataset-specific adapter.
3. Validate the common 19-column schema and portable image-path metadata.
4. Generate nine per-dataset Unified CSVs.
5. Merge the candidate pools into a single master CSV.
6. Independently validate the merged metadata.

Implementation:

- `scripts/unified/schema.py` — shared schema definition
- `scripts/unified/adapters/` — dataset-specific adapters
- `scripts/unified/build_unified.py` — per-source exports
- `scripts/unified/merge_unified.py` — candidate master export
- `scripts/unified/validate_master.py` — independent master validation

The local master candidate manifest is `unified/all_sources_unified_v1.csv`.

**Frozen P2 master SHA-256:**

```text
4fba9e5578b19679275f061dc97740518d300830a0306b725c39e0dfc15e34ce
```

## P3 — Source Training and Validation Protocols

The candidate pools were divided using identity-disjoint validation where appropriate, without using source official test sets. Validation roles and camera/view semantics are dataset-specific; they should not be interpreted as one universal evaluation protocol.

| Contributor | Source datasets | P3 strategy |
|---|---|---|
| Member 1 | CUHK03-NP, DeepChange, Occluded-PoseTrack-ReID | Identity-disjoint internal validation; dataset-specific retrieval protocols |
| Member 2 | ORBench RGB, PRID2011, RPIfield | Identity-disjoint internal validation with cross-context query/gallery checks; RPIfield is **not** certified camera-disjoint |
| Member 3 | IUST-PersonReID, LaST, LPW | IUST identity-disjoint cross-camera validation; LaST official validation; LPW identity-disjoint cross-view validation |

For LaST, the official validation contains **100 query images** and **21,279 gallery images**, including **20,000 unlabeled distractors**. Those distractors are retained in the gallery, not assigned training class IDs.

All validation protocols remain subject to their documented limitations; for example, LaST official validation is dominated by same-video positive matches.

## P5 — Provisional Global Handoff

The three member handoffs were merged into `unified/final_dataset/Global/` without changing their recorded training class assignments.

| Metric | Result |
|---|---:|
| Global training images | 782,064 |
| Global training classes | 10,036 |
| Global class ID range | 0–10035 |
| Global validation manifest rows | 181,597 |
| Unique validation image records | 180,124 |
| Total distinct manifest image records across training and validation | 962,188 |
| Validation protocols checked during merge | 10 |
| P4 clearance | **NOT_CLEARED** |
| Fine-tuning readiness | **False** |

The 962,188 records consist of **940,809 source candidate images** plus **21,379 LaST official validation images**. The validation CSV has **1,473 extra rows** because CUHK03-NP reuses the same validation images in two legitimate, opposite-direction retrieval protocols. These are metadata-record counts, **not** proof that there are no byte- or visually duplicated images between datasets.

### Handoff layout

```text
unified/final_dataset/
├── Member 1/
├── Member 2/
├── Member 3/
└── Global/
    ├── train_manifest.csv               # Full, local only
    ├── validation_manifest.csv          # Full, local only
    ├── identity_class_mapping.json
    ├── source_registry.json
    ├── dataset_protocol.json
    ├── checksums.sha256
    └── preview/
        ├── train_manifest_preview.csv
        └── validation_manifest_preview.csv
```

Each member handoff uses the same six-file schema, and each has its own preview folder. The full CSV manifests are retained locally; **only their first 10 data rows plus the header** are committed as preview CSVs. `checksums.sha256` records the hashes of the **full output files**, not the preview substitutes.

Relevant generation scripts:

- `scripts/dg_reid_data_pipeline/build_final_dataset.py` — Member 1 handoff
- `scripts/dg_reid_data_pipeline/build_member3_p3_p5.py` — Member 3 handoff
- `scripts/dg_reid_data_pipeline/build_global_p3_p5.py` — global merge and validation
- `scripts/dg_reid_data_pipeline/create_global_previews.py` — global 10-row previews

Member 2's handoff was repaired and checked locally; its corrected full manifests are **not reproducible from the committed preview-only builder alone**. Preserve the verified full files and restore a reproducible builder before the final research release.

## Git and Data Availability

- `unified/previews/` contains the original P2 candidate-metadata previews (ten rows per source).
- `unified/final_dataset/*/preview/` contains P3/P5 training and validation manifest previews.
- Full candidate CSVs and full training/validation CSVs are kept outside Git because of size and data-sharing constraints.
- The original images are **not** bundled with this repository.
- Local metadata validation does **not** prove that the image files exist at runtime; verify the roots and every referenced path in the Kaggle/training environment.

## Remaining Gates

1. **P4 — Cross-Dataset Leakage Clearance:** check acquisition provenance, exact image hashes, and risk-targeted near duplicates between sources and unseen targets; document keep/exclude decisions.
2. **P5 Final Approval:** apply any evidence-based P4 exclusions and regenerate/freeze global manifests, mappings, and checksums as needed.
3. **P6 — Multi-Source Sampling:** evaluate source balance, identity diversity, and camera/temporal repetition.
4. **P7 — Model and Training Preparation:** verify checkpoint provenance, runtime paths, dataloader behavior, and training smoke tests.
5. **Unseen-Domain Evaluation:** evaluate on Market1501 and MARS using the specified target protocols, without using target test results for model selection.

**Status (22 October 2026):** P2 candidate metadata validated; P3 member splits and provisional global P5 handoff produced and metadata-checked. **P4 and runtime image-path verification pending. Training has not started.**
