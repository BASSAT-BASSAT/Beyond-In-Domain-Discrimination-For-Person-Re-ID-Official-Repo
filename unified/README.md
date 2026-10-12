# Unified ReID Data Pipeline — v1

## Overview

This pipeline converts nine audited ReID candidate datasets into a consistent 19-column metadata schema.

The goal is to prepare reproducible, source-aware metadata for future multi-source Domain Generalization (DG) experiments.

## Validated Results

| Metric | Result |
|---|---:|
| Source datasets | 9 |
| Candidate images | 940,809 |
| Source-scoped identities | 11,297 |
| Metadata columns | 19 |
| Dataset adapters | 9 |
| Independent master validation | PASS |

These are candidate samples, not a finalized leakage-cleared training set.

## Pipeline

1. Read dataset-specific candidate manifests.
2. Convert rows using nine dataset adapters.
3. Validate the unified 19-column schema.
4. Generate per-dataset Unified CSVs.
5. Merge all sources into one master manifest.
6. Independently validate every master row.

## Implementation

- `scripts/unified/schema.py` — Schema definition
- `scripts/unified/adapters/` — Dataset-specific adapters
- `scripts/unified/build_unified.py` — Individual manifest generation
- `scripts/unified/merge_unified.py` — Master manifest generation
- `scripts/unified/validate_master.py` — Independent validation

## Previews

The `previews/` directory contains ten example rows per dataset, preserving the original 19-column structure.

Full manifests are kept locally and excluded from Git because of their size and data-sharing considerations.

## Reproducibility

Master manifest: `unified/v1/all_sources_unified_v1.csv`

SHA-256:

`4fba9e5578b19679275f061dc97740518d300830a0306b725c39e0dfc15e34ce`

## Next Steps

- Identity-disjoint training/validation protocols
- Cross-dataset leakage detection
- Final training source selection
- Balanced multi-source sampling
- Fine-tuning and unseen-domain evaluation

**Status:** Unified Candidate Metadata v1 validated. Training has not started.
