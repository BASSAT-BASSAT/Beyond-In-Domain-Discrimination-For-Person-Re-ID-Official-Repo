# Domain-Generalizable Person Re-Identification (DG-ReID)
**Multi-source ReID research | Dataset auditing | Reproducible data engineering | Cross-domain evaluation**
> **Research snapshot — 12 October 2026.** Ten datasets have been audited, nine local candidate pools have been prepared, and Unified Candidate Metadata v1 has passed independent validation. Final training/validation protocols, cross-dataset leakage clearance, multi-source fine-tuning, and unseen-domain evaluation are **not complete**.

---

## Overview
Person Re-Identification (ReID) aims to retrieve observations of the same person across cameras, locations, or time.
A model trained on one dataset may perform poorly on another because acquisition conditions are different.
Examples of domain shifts include:
- Camera resolution and image quality.
- Viewpoint, background, and illumination.
- Indoor versus outdoor scenes.
- Occlusion and crowd density.
- Clothing and appearance changes over time.
- Detection, cropping, and annotation procedures.
- Differences in training identities and sampling distributions.
Our research question is:
> Can supervised fine-tuning with multiple diverse source domains improve ReID generalization to genuinely unseen domains compared with pretrained-only and controlled single-source baselines?
The project has two complementary parts:
1. **Data engineering:** audit, clean, normalize, and validate candidate source data.
2. **Model experimentation:** design safe protocols, train controlled baselines, and evaluate generalization.
At this stage, the first part has reached a validated **candidate-metadata milestone**; the second part remains planned.

---

## Current Progress

| Item | Verified status |
| --- | --- |
| Dataset audits | **10 complete** |
| Proposed locally cleaned candidate sources | **9** |
| Excluded source | **PRW**, due to Market1501 acquisition overlap |
| Candidate image rows | **940,809** |
| Source-scoped identity keys | **11,297** |
| Unified schema fields | **19** |
| Dataset-specific adapters | **9 validated** |
| Individual Unified v1 manifests | **9 generated** |
| Merged Unified Candidate Manifest v1 | **Validated** |
| Final source train/validation protocol | **Pending** |
| Cross-source and source-target leakage clearance | **Pending** |
| Multi-source model fine-tuning | **Not started** |
| Final unseen-target benchmark | **Not completed** |
**Interpretation matters:** 11,297 counts distinct identities *within defined source namespaces*.
It does not prove that 11,297 different real-world people appear across all acquisitions.
Likewise, 940,809 is the size of the **candidate pool**, not a frozen training set.
The nine candidate sources may not all be selected for every future experiment.

---

## Research Workflow
```text
10 source-dataset investigations
                |
                v
Dataset protocols, statistics, quality and duplicate audits
                |
                v
Evidence-based cleaning rules and 9 candidate pools
                |
                v
9 dataset-specific adapters -> Unified Schema v1
                |
                v
9 per-source Unified Candidate Manifests
                |
                v
Merged master + independent row-by-row validation
                |
                v
P3: Source train/validation protocol design
                |
                v
P4: Source-source and source-target leakage analysis
                |
                v
P5: Freeze eligible data and training class mapping
                |
                v
P6: Balanced, reproducible multi-source sampler
                |
                v
P7: Baselines, fine-tuning and ablations
                |
                v
P8: Official unseen-domain evaluation and analysis
```
P3 and P4 must both be resolved before approving the final training manifest.

---

## Dataset Inventory
The following numbers describe **locally prepared candidate pools** after their respective audit decisions.

| Source | Candidate images | Source-scoped IDs | Current candidate scope |
| --- | ---: | ---: | --- |
| CUHK03-NP | 7,365 | 767 | Detected, official train |
| IUST-PersonReID | 72,393 | 1,193 | Official train |
| LaST | 71,248 | 5,000 | Official train |
| DeepChange | 75,009 | 450 | Clean official train |
| LPW | 418,447 | 1,975 | Clean official train |
| Occluded-PoseTrack-ReID | 17,898 | 1,000 | Official train |
| ORBench RGB | 26,932 | 600 | Clean RGB/VIS train |
| PRID2011 | 40,033 | 200 | Matched multi-shot; split unassigned |
| RPIfield | 211,484 | 112 | Known actors; split unassigned |
| **Total** | **940,809** | **11,297** | **Candidates, not final training data** |
### CUHK03-NP
- Uses the **detected** image variant for the current candidate pool.
- Contains 7,365 candidate training images and 767 training identities.
- Original collection has five physical camera pairs.
- Filename `c1` and `c2` indicate **pair-local views**, not global camera IDs.
- The corresponding pair must not be guessed from a view label alone.
- Source image indices are not automatically temporal frame identifiers.
### IUST-PersonReID
- Contains 72,393 candidate training images and 1,193 identities.
- Records 17 physical camera codes within the source dataset.
- Preserves sequence/frame information where the source supports it.
- The audit identified substantial temporal near-neighbor redundancy.
- Future sampling must avoid overrepresenting almost identical moments.
- The official test/query protocol remains outside the source candidates.
### LaST
- Contains 71,248 candidate training images and 5,000 identities.
- Provides video, frame, clothing, image, and bounding-box identifiers.
- There are 3,180 training identities with multiple clothing labels.
- Video labels do **not** automatically establish physical camera IDs.
- Clothing changes offer a useful future sampler and ablation dimension.
- Observed difficult crops are not discarded solely for being unusual.
### DeepChange
- Original official training subset: 75,083 images and 450 identities.
- Local hard cleaning excluded 74 documented conflicting rows.
- Current candidate pool: **75,009 images**, retaining 450 identities.
- Audited metadata includes camera, tracklet, frame, and temporal context.
- Candidate rows do not directly expose a complete image path.
- The adapter resolves paths through verified image-hash evidence.
- The path reconciliation is based on a composite source key.
### LPW
- Original official training subset: 418,739 images and 1,975 identities.
- Two conflicting tracklets account for 292 excluded rows.
- Current pool: **418,447 images** and 1,975 identities.
- Retained candidates cover 5,864 tracklets across two training scenes.
- Labels such as `scen2-10` are scene-scoped identity labels.
- View codes are **scene-local**, not guaranteed global physical cameras.
- An image sequence number is not a verified timestamp by default.
### Occluded-PoseTrack-ReID
- Contains 17,898 candidate train crops and 1,000 training identities.
- Training images originate from 474 logical videos.
- `logical_camid` is a **video proxy**, not a physical camera measurement.
- PoseTrack image IDs must not be treated as temporal frame numbers without proof.
- Numeric person IDs can be reused across source splits for different people.
- Identity namespaces therefore include the relevant split context.
### ORBench RGB
- Current work covers the RGB/VIS modality of ORBench.
- Original RGB training rows: 27,063.
- Audited exclusions total 131 rows: 111 geometry outliers and 20 exact copies.
- Current candidate pool: **26,932 images** and 600 identities.
- LLCM contributes 9,643 rows; SYSU contributes 17,289 rows.
- Camera contexts are scoped to the underlying LLCM or SYSU source.
- The existing cleaning exclusions must not be applied twice.
### PRID2011
- Uses 40,033 matched multi-shot images belonging to 200 shared identities.
- Camera `cam_a` contributes 23,330 images; `cam_b` contributes 16,703.
- The candidate identities have 400 camera-specific trajectories.
- Single-shot views overlap multi-shot data and are not added as new samples.
- Other camera-only identities were not selected for this initial pool.
- An official fixed supervised source train/validation partition was not adopted.
- Unified rows currently retain `source_split = unassigned`.
### RPIfield
- The original dataset has 601,581 images and 4,108 source identities.
- Hard audit flags include 1,262 cross-person pixel-conflict rows.
- One additional landscape anomaly was flagged for exclusion.
- Our conservative candidate pool uses **known actors only**.
- That pool contains 211,484 images and 112 identities after exclusions.
- Distractor identities are not assumed valid supervised training classes.
- Appearance labels are not automatically clothing labels.
- Sequence IDs are not automatically verified tracklet IDs.
- Its supervised train/validation split remains `unassigned`.

---

## Why PRW Is Excluded
PRW was investigated as the **tenth audited dataset**.
However, it is not one of the nine proposed source-training pools.
An acquisition-level audit detected exact overlap between PRW source video frames and Market1501 raw frames.
- Matching raw PRW frames: **3,886**.
- Total PRW source video frames considered: **5,663**.
- Fraction affected: **68.62%**.
- The evidence concerns *source frames*, not a claim that 68.62% of all person crops are pixel-identical.
This overlap conflicts with the plan to treat Market1501 as an unseen evaluation domain.
Simply removing identical crops is not sufficient to prove acquisition independence.
Therefore, PRW is excluded at the source-eligibility level for this proposed experiment.
The PRW scripts and audit report remain included for scientific traceability.

---

## Planned Evaluation Targets

| Dataset | Intended role | Protocol note |
| --- | --- | --- |
| Market1501 | Unseen image-based evaluation | Respect official query/gallery conventions |
| MARS | Unseen tracklet-based evaluation | Aggregate and compare at the tracklet level |
| MSMT17 V2 | Potential additional target | Deferred following download/CRC issues |
**Unseen** is an experimental constraint, not a dataset label alone.
Target images and target test labels must not be used as supervised source training samples.
Target-test metrics must not silently become the hyperparameter-selection procedure.
The project needs a documented source-side validation strategy before final evaluation.

---

## Data Engineering Principles
1. **Preserve raw images:** do not rename, overwrite, or mutate original datasets.
2. **Preserve official splits:** do not rewrite canonical query/gallery definitions.
3. **Separate auditing from cleaning:** warnings are not automatic deletions.
4. **Document hard exclusions:** every excluded row needs supporting evidence.
5. **Retain meaningful hard examples:** occluded or unusual crops can still be valid.
6. **Scope identities:** equal numeric PIDs across datasets do not imply one person.
7. **Specify camera semantics:** physical cameras, scene views, and video proxies differ.
8. **Do not invent absent metadata:** unknown values remain missing.
9. **Prefer portable paths:** dataset root keys plus paths relative to that root.
10. **Validate published artifacts:** counts, structure, provenance, and checksums.
11. **Preserve versions:** a revised cleaning policy should not silently replace v1.
12. **Keep source and target roles explicit:** no undocumented crossover.
### Duplicate terminology
- **Filename collision:** identical basenames are not enough to prove identical content.
- **Byte duplicate:** complete encoded files have identical bytes.
- **Pixel duplicate:** decoded images contain exactly matching pixel arrays.
- **Near duplicate:** crops may be visually similar, but not pixel-identical.
- **Acquisition overlap:** observations originate from shared recording material.
Each category needs an appropriate verification method.

---

## Unified Metadata Schema v1
Every individual Unified v1 CSV and the merged candidate master use the same **19 ordered fields**.

| # | Field | Purpose |
| ---: | --- | --- |
| 1 | `sample_uid` | Stable key based on dataset and relative path |
| 2 | `dataset` | Canonical source dataset name |
| 3 | `candidate_pool` | Candidate selection provenance |
| 4 | `dataset_root_id` | Dataset root lookup key |
| 5 | `relative_path` | Portable image location below that root |
| 6 | `source_split` | Source role or `unassigned` |
| 7 | `raw_pid` | Original/source person label |
| 8 | `identity_namespace` | Dataset and split scope for identity labels |
| 9 | `identity_key` | Namespaced person identity |
| 10 | `raw_camera_or_view` | Original view/camera/video-proxy value |
| 11 | `camera_semantics` | Meaning of the camera/view field |
| 12 | `camera_context_id` | Validated source-scoped context, where available |
| 13 | `video_id` | Source video identifier, if supported |
| 14 | `tracklet_id` | Source tracklet identifier, if supported |
| 15 | `frame_id` | Verified frame identifier, if supported |
| 16 | `clothes_id` | Source clothing category, if supported |
| 17 | `source_row_number` | Original candidate CSV row index |
| 18 | `source_image_index` | Dataset-specific sequence/image index |
| 19 | `source_metadata_json` | Additional source metadata retained as JSON |
The presence of a column does not mean every dataset supplies that information.
Empty values indicate absent or unverified attributes, not a default value of zero.
`sample_uid` is derived from dataset name and relative path using SHA-256.
It is **not** a file-byte or decoded-pixel hash and cannot establish duplicate freedom.
### Identity examples
```text
CUHK03_NP::detected::train::1
IUST_PersonReID::train::1
LaST::train::2
LPW::train::scen2-10
Occluded-PoseTrack-ReID::train::2125
PRID2011::matched_multishot::shared-0001
RPIfield::known_actor::1
```
These are source labels, not the final contiguous classifier indices.
Final numeric training label mapping is planned for P5.

---

## Camera and Temporal Semantics
Different datasets use similar field names for **different real-world concepts**.

| Dataset | Camera/view semantics | Limitation |
| --- | --- | --- |
| CUHK03-NP | `pair_local_view` | `c1` is not a global camera ID |
| IUST | `physical_camera_code` | Source-scoped physical camera labels |
| LaST | `not_available` | Video ID is not a verified physical camera |
| DeepChange | `physical_camera_code` | Camera code is dataset-scoped |
| LPW | `scene_local_view` | Scene/view is not automatically global |
| Occluded-PoseTrack | `logical_video_proxy` | Logical video is not a physical camera |
| ORBench RGB | `source_scoped_camera_code` | LLCM and SYSU contexts differ |
| PRID2011 | `physical_camera_code` | `cam_a` and `cam_b` |
| RPIfield | `physical_camera_code` | Twelve source camera codes |
A future DataLoader or sampler must check `camera_semantics` before using camera-aware strategies.
Temporal IDs require the same care: filename order and annotation IDs are not always physical timestamps.
This distinction prevents invalid cross-camera or temporal sampling assumptions.

---

## Repository Organization
The Git branch contains the research code and documentation, while full image data remains in the authorized workspace.
**Important:** selected reports, evidence CSVs, and small previews may be committed individually even though their parent directories are ignored by default.
```text
DG_ReID_Audit/
├── README.md
├── .gitignore
├── scripts/
│   ├── CUHK03_NP/
│   ├── DeepChange/
│   ├── IUST_PersonReID/
│   ├── LaST/
│   ├── LPW/
│   ├── MARS/
│   ├── Occluded-PoseTrack-ReID/
│   ├── ORBench/
│   ├── PRID2011/
│   ├── PRW/
│   ├── RPIfield/
│   └── unified/
│       ├── adapters/
│       ├── schema.py
│       ├── build_unified.py
│       ├── merge_unified.py
│       └── validate_master.py
├── reports/                    # Selected audit Markdown reports / evidence
├── manifests/                  # Selected cleaning-evidence CSVs
├── unified/
│   ├── README.md               # Pipeline documentation
│   ├── previews/               # Small per-dataset schema examples
│   ├── v0/                     # Local-only historical output
│   └── v1/                     # Local-only full Unified CSVs
├── datasets/                   # Local-only original datasets
└── samples/                    # Local visual-audit examples
```
This is a **combined workspace and publication layout**, not a promise that all shown folders are tracked by Git.
The `.gitignore` excludes `/datasets/`, `/manifests/`, `/reports/`, and `/unified/` by default.
`git add -f` can stage **specific reviewed files** from ignored directories.
No blanket `git add -f` of whole dataset directories should be used.

---

## Audit Reports and Supporting Evidence
The audit record is organized by dataset to keep results traceable.
The ten Markdown audit documents cover:
- Dataset and split structure.
- Person identity conventions.
- Physical camera versus view semantics.
- Image geometry and quality conditions.
- Missing, corrupted, and suspicious data.
- Exact-byte and decoded-pixel duplicate investigations.
- Temporal or tracklet-related considerations, when supported.
- Proposed local cleaning decisions and their limitations.
Selected small evidence files currently prepared for Git include:
```text
manifests/DeepChange/deepchange_hard_exclusions_v0.csv
manifests/LPW/lpw_hard_exclusions_v0.csv
manifests/RPIfield/rpifield_hard_exclusions_v0.csv
reports/PRW/prw_market1501_overlap_summary.csv
```
These files document **why** specific candidate rows were excluded or why PRW was rejected as a source.
They are not substitutes for the full local audit evidence collection.
Before sharing new CSVs or visual examples, review licensing, personal data, source paths, and other sensitive content.

---

## Unified Pipeline Components
### `scripts/unified/schema.py`
- Defines the 19-column v1 schema.
- Checks column names and ordering.
- Includes a reference-header validation mechanism.
### `scripts/unified/adapters/`
- Contains nine source-specific adapters.
- Reads the corresponding candidate manifest.
- Normalizes fields without guessing their meaning.
- Preserves source-specific metadata where supported.
- Can perform read-only verification independently.
### `scripts/unified/build_unified.py`
- Registers the nine supported source adapters.
- Builds one dataset output at a time.
- Checks expected row counts and identity keys.
- Produces deterministic Unified CSV artifacts.
- Refuses accidental overwrite of an existing output.
### `scripts/unified/merge_unified.py`
- Merges the nine individual v1 outputs in a fixed order.
- Rechecks schema and source provenance.
- Ensures master candidate UID integrity.
- Refuses to replace an existing master silently.
### `scripts/unified/validate_master.py`
- Compares the master with all nine source files.
- Validates rows across all 19 fields.
- Checks source counts, identity keys, and uniqueness.
- Validates the expected v1 checksum.
- Leaves input data unchanged.

---

## Running the Validated Pipeline
Use the prepared Lightning Studio workspace with its original authorized datasets and audit artifacts.
```bash
cd ~/DG_ReID_Audit
```
### Inspect schema
```bash
python scripts/unified/schema.py
```
### Check one adapter without generating files
```bash
python -m scripts.unified.adapters.cuhk03
```
Other registered adapters: `iust`, `last`, `deepchange`, `lpw`, `occluded_posetrack`, `orbench`, `prid2011`, and `rpifield`.
### Build a per-source manifest (only when absent)
```bash
python -m scripts.unified.build_unified --dataset iust
```
### Merge per-source outputs (only when the target does not already exist)
```bash
python -m scripts.unified.merge_unified
```
### Validate the existing published master
```bash
python -m scripts.unified.validate_master
```
The last step is the appropriate **read-only verification** for the current frozen v1 master.
Do not delete or overwrite previously validated v1 CSVs just to repeat a build command.
The current code is not a complete installer: the source manifests, images, and audit evidence are required for end-to-end reproduction.

---

## Reproducibility Snapshot
**Validated master location (local only):**
```text
unified/v1/all_sources_unified_v1.csv
```
**Master SHA-256:**
```text
4fba9e5578b19679275f061dc97740518d300830a0306b725c39e0dfc15e34ce
```
**Independent validator results:**
```text
Datasets verified:       9
Rows compared:       940,809
Unique sample UIDs:  940,809
Source-scoped IDs:    11,297
Schema columns:           19
MASTER VALIDATION:       PASS
```
The successful comparison establishes internal consistency of the generated metadata at this version.
It does **not** establish globally unique human identities, legal sharing rights, or freedom from all acquisition overlap.
### Small Unified previews
Nine preview CSVs have been prepared with **10 sample rows and 19 columns each**.
They illustrate the schema without mirroring full, large generated manifests.
The previews are examples, not an unbiased random sample or the training dataset.
The local full master is approximately 542 MB; the merged and individual files together are much larger.
Full generated outputs are intentionally excluded from normal Git tracking.

---

## Data and Git Publication Policy
### Intended for Git
- Python auditing and adapter scripts.
- Project README and pipeline documentation.
- Reviewed Markdown dataset audit reports.
- Small, necessary evidence or summary CSVs.
- Reviewed metadata previews and reproducibility information.
### Kept outside ordinary Git tracking
- Original dataset images and archives.
- Large generated manifests and audit tables.
- Full Unified v0/v1 CSV artifacts.
- Model checkpoints, embedding caches, and large intermediate arrays.
- Unreviewed visual samples or restricted annotations.
- Tokens, credentials, and machine-local configuration secrets.
A private GitHub repository still requires appropriate data-use permissions.
A `.gitignore` rule prevents new files from being staged normally; it does not certify that a file is safe or legally shareable.
### Branching
The research workspace is published on the dedicated branch:
```text
research/dg-reid-audit
```
It is an independent research branch within the team's private repository.
The branch was initialized with its own Git history rather than from the team's `main` branch.
Combining these histories later is a separate decision, not part of this release.

---

## Remaining Leakage and Protocol Risks
Local cleaning and successful CSV validation are **not sufficient** for final model training.
Important outstanding checks include:
- Cross-source exact-byte duplicates.
- Cross-source decoded-pixel duplicates.
- Near-duplicate acquisitions or repeated recordings.
- Source-to-Market1501 and source-to-MARS acquisition relationships.
- Underlying data reused by repackaged benchmark collections.
- Identity collisions and person-label inconsistencies.
- Train/validation identity overlap within source protocols.
- Temporal correlations that can inflate training sample counts.
- Inappropriate use of target test scores for model selection.
For example, ORBench RGB contains material originating from LLCM and SYSU; provenance must remain explicit.
The PRW–Market1501 overlap was investigated specifically and led to the current PRW exclusion.
No equivalent blanket clearance is claimed for the other nine candidate pools.

---

## Planned Experiments and Milestones

| Phase | Planned deliverable | Status |
| --- | --- | --- |
| P0 | Complete dataset audits | **Done** |
| P1 | Local candidate cleaning v0 | **Done** |
| P2 | Unified schema, adapters, master validation | **Done (v1)** |
| P3 | Identity-disjoint source train/validation protocols | Pending |
| P4 | Cross-source and source-target leakage analysis | Pending |
| P5 | Frozen source selection and contiguous class mapping | Pending |
| P6 | Fair and reproducible multi-source sampler | Pending |
| P7 | Pretrained, single-source, and multi-source experiments | Pending |
| P8 | Unseen-domain evaluation and ablation analysis | Pending |
### P3 — Source protocols
- Resolve the unassigned supervised role for PRID2011 and RPIfield.
- Decide whether source validation identities are held out.
- Store deterministic split files and random seeds.
- Check that identity namespaces remain valid after splitting.
### P4 — Leakage clearance
- Build targeted source-source and source-target comparisons.
- Record both detected problems and evidence limitations.
- Keep original dataset artifacts unchanged.
### P5 and P6 — Training design
- Freeze approved source domains and identity labels.
- Map each training identity to a contiguous classifier class ID.
- Define source balancing, identity sampling, and temporal control.
- Fix batch size, steps, epoch semantics, and oversampling rules.
### P7 — Model experiments
- Establish a pretrained-only feature extractor baseline.
- Run controlled single-source fine-tuning where appropriate.
- Run supervised multi-source fine-tuning under matched budgets.
- Evaluate sampler and source-composition ablations.
- Track seeds, hyperparameters, checkpoints, and training provenance.
### P8 — Final evaluation
- Use the official Market1501 image-based protocol.
- Use the official MARS tracklet-based protocol.
- Report mAP and CMC Rank-1 with clear dataset and protocol labeling.
- Avoid selecting the best checkpoint through repeated target-test tuning.
- Separate experimental conclusions from engineering-validation facts.

---

## Research Integrity and Attribution
This repository is a research work in progress, not a claim of a finished universal ReID system.
Conclusions about generalization require controlled training runs and properly isolated evaluation targets.
Raw image rights, source-dataset conditions, and privacy obligations must be respected.
When a figure or result is reported, preserve the script, dataset version, source role, and checksum needed to interpret it.
Changes to a frozen manifest should create a new reviewed artifact version rather than silently replacing existing evidence.
**Maintained for:** the DG-ReID research team  
**Snapshot:** 12 October 2026  
**Current stage:** Unified Candidate Metadata v1 validated; training protocols and experiments pending.
