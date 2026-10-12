# Domain-Generalizable Person Re-Identification (DG-ReID)

**Research repository | Multi-source supervised learning | Reproducible data engineering**

> **Project status (12 October 2026):** Dataset audits, local candidate cleaning, and the
> Unified Candidate Manifest v1 are complete. The multi-source training protocol,
> cross-dataset leakage clearance, final source selection, fine-tuning, and final
> unseen-domain evaluation are **not** complete.

---

## 1. Project at a Glance
This repository documents our research workflow toward a **domain-generalizable
Person Re-Identification (ReID)** model.

Person ReID attempts to match images or tracklets of the same person across
observations. Real-world ReID is difficult because camera environments, clothing,
viewpoint, resolution, lighting, occlusion, and acquisition conditions change.

Our research investigates whether **supervised fine-tuning on multiple diverse
source domains** can improve performance on **unseen target domains** compared
with suitable pretrained-only and single-source baselines.

We are deliberately treating **data correctness and experimental validity** as
first-class research problems, not just preliminary preprocessing tasks.
### Current snapshot
| Item | Verified / current status |
| --- | --- |
| Candidate datasets audited | **10 / 10** |
| Candidate source pools prepared | **9 / 9**, local cleaning v0 |
| Datasets excluded from the proposed source pool | **PRW** (Market1501 acquisition overlap) |
| Path-resolution checks | **9 / 9 PASS** |
| Dataset-specific unified adapters | **9 / 9 validated** |
| Individual Unified Manifests v1 | **9 / 9 generated** |
| Master Unified Candidate Manifest v1 | **Generated and independently validated** |
| Candidate image rows | **940,809** |
| Source-scoped identity keys | **11,297** |
| Unified metadata columns | **19** |
| Approved final training sources | **Not frozen** |
| Multi-source fine-tuning | **Not started** |

The **11,297** figure counts distinct identity keys within source namespaces.
It is **not** a proof that all of these represent unique real people across
independently acquired datasets.
### What this repository is — and is not
- **It is:** an auditable record of dataset investigation, evidence-based source
  candidate selection, schema design, reproducible conversion, and validation.
- **It is becoming:** an end-to-end, versioned multi-source DG-ReID experiment.
- **It is not yet:** a released trained DG model, a finished benchmark study,
  or a leakage-cleared final training set.

---

## 2. Research Objective
### Core research question
> Does supervised multi-source fine-tuning of a ReID feature extractor improve
> generalization to unseen domains compared with pretrained-only and
> appropriately controlled single-source fine-tuning?
### Planned experimental logic
```text
Audited candidate datasets (10)
            |
            v
Evidence-based local cleaning + source eligibility decisions
            |
            v
Nine dataset-specific candidate manifests
            |
            v
Adapters -> Unified Schema v1 -> Master Candidate Manifest
            |
            v
Identity-disjoint protocols + cross-dataset leakage checks
            |
            v
Freeze safe training sources, identity labels, and sampling policy
            |
            v
Pretrained baseline / single-source / multi-source fine-tuning
            |
            v
Unseen-domain evaluation on official target protocols
            |
            v
Ablations, analysis, and research reporting
```
### Primary evaluation targets
| Target | Planned role | Current note |
| --- | --- | --- |
| **Market1501** | Unseen image-based ReID evaluation | Available in the research workspace |
| **MARS** | Unseen tracklet-based ReID evaluation | Package/protocol preparation checked |
| **MSMT17 V2** | Additional possible target | Deferred after download/CRC problems |

**Target images and test labels must not become supervised source training data.**
Target test results must not be used as an undocumented hyperparameter-selection
mechanism; a separately defined validation policy is required.

The final evaluation must respect each dataset's official protocol. In
particular, **MARS is tracklet-based**, whereas **Market1501 is image-based**.

---

## 3. Milestone Tracker
This table distinguishes finished engineering work from future research decisions.

| Phase | Scope | Status |
| --- | --- | --- |
| **P0** | Audit ten datasets and preserve evidence | **Complete** |
| **P1** | Local cleaning and candidate pool selection | **Complete (v0)** |
| **P2.1** | Path resolution across nine candidates | **Complete** |
| **P2.2** | Unified metadata schema and nine adapters | **Complete (v1)** |
| **P2.3** | Master merge and independent validation | **Complete (v1)** |
| **P3** | Training/validation source protocols | **Pending** |
| **P4** | Cross-source / source-target leakage safety | **Pending** |
| **P5** | Final source freeze and class-ID mapping | **Pending** |
| **P6** | Fair, reproducible multi-source sampler | **Pending** |
| **P7** | Controlled fine-tuning and ablations | **Pending** |
| **P8** | Unseen-domain evaluation and analysis | **Pending** |

The phase names describe **workflow milestones**, not claims of model accuracy.
The order of P3/P4 investigation can be adjusted when safety findings require it;
**both must be cleared before P5 freezes training inputs**.
### Completed milestone: P0 — Dataset audits
We investigated dataset structure, identities, camera/view semantics, training
and evaluation roles, images and paths, geometry, duplicates, and relevant
cross-dataset acquisition relationships.

Artifacts include dataset-specific CSV manifests, audit flags, evidence, and
Markdown reports in the local `manifests/` and `reports/` directories.
### Completed milestone: P1 — Local cleaning v0
We generated candidate pools without altering raw images or official
query/gallery protocols. Hard exclusions were evidence-based, while normal
sampling redundancy and potentially useful difficult images were generally
retained for later sampler decisions.

A *clean candidate pool* is **not** necessarily a finalized supervised
training subset or a leakage-safe domain.
### Completed milestone: P2 — Unified metadata preparation v1
We generated nine schema-compatible CSVs, merged them into one master file,
then verified the master against the individual sources **row by row across
all 19 columns**.

Validation passed for **940,809 rows**, **940,809 unique sample UIDs**, and
**11,297 source-scoped identity keys**.

---

## 4. Audited Datasets and Current Candidate Pools
### Candidate source summary
| Source | Candidate images | Candidate IDs | Selection / role |
| --- | ---: | ---: | --- |
| CUHK03-NP | 7,365 | 767 | `detected`, official train candidates |
| IUST-PersonReID | 72,393 | 1,193 | Official train candidates |
| LaST | 71,248 | 5,000 | Official train candidates |
| DeepChange | 75,009 | 450 | Clean official train candidates |
| LPW | 418,447 | 1,975 | Clean official train candidates |
| Occluded-PoseTrack-ReID | 17,898 | 1,000 | Official train candidates |
| ORBench RGB | 26,932 | 600 | Clean RGB/VIS train candidates |
| PRID2011 | 40,033 | 200 | Matched multi-shot candidate pool; split unassigned |
| RPIfield | 211,484 | 112 | Known-actor candidate pool; split unassigned |
| **TOTAL** | **940,809** | **11,297** | **Candidate manifests only** |

The first seven contribute **689,292** candidate images. PRID2011 and RPIfield
contribute an additional **251,517** images, but their custom supervised
train/validation protocols have not yet been finalized.
### CUHK03-NP
- Selected the **detected** variant, not the alternative **labeled** crops.
- Kept **7,365 official training images / 767 identities**.
- The dataset has **five physical camera pairs**, but filename `c1`/`c2`
  refers to a **pair-local view**, not a globally shared physical camera ID.
- The unified representation uses `camera_semantics = pair_local_view`.
- The physical camera-pair context is not inferred without supporting data.
- Image indices are retained as source indices, **not assumed video frames**.
- No mandatory training-image hard exclusion was established for this pool.
### IUST-PersonReID
- Kept **72,393 official training images / 1,193 identities**.
- Contains **17 camera codes** with sequence and frame metadata.
- Observed strong near-frame redundancy: **92.79%** of consecutive within-
  identity/camera frame gaps were at most six in the audit.
- We keep valid examples and plan to control temporal oversampling later.
- Query imagery also appears in the test-image collection; this must not
  be misinterpreted as independent extra training imagery.
- Sequence metadata is preserved without automatically declaring a tracklet.
### LaST
- Kept **71,248 training images / 5,000 identities**.
- **3,180** training identities have multiple clothing labels.
- Preserved `video_id`, `frame_id`, `clothes_id`, `image_id`, and `bbox_id`.
- Do not interpret the provided video IDs as confirmed physical camera IDs.
- Audit anomalies and unusual crop geometry are documented; valid difficult
  examples are not removed merely because their shapes are unusual.
- Clothing-aware sampling is **planned**, not implemented yet.
### DeepChange
- Original official training subset: **75,083 images / 450 identities**.
- Local cleaning excluded **74** training rows with documented conflicts or
  train-to-evaluation exact duplicate evidence.
- Current pool: **75,009 images / 450 identities**.
- Camera, frame, date/time, and tracklet information are retained.
- The clean candidate CSV has **no direct `path` column**.
- Its adapter joins to verified `reports/DeepChange/image_hashes.csv` evidence.
- The join key uses filename, person ID, camera ID, and tracklet ID.
- The adapter checked all **75,009** candidate-to-path matches.
- There are **16 camera codes in train**, despite 17 in the broader dataset.
### LPW
- Original official training subset: **418,739 images / 1,975 identities**.
- Two conflicting `scen2/view2` tracklets were excluded: **292 rows**.
- Current pool: **418,447 images / 1,975 identities**.
- Identity labels are **scene-scoped**, e.g. `scen2-10`.
- Candidate pool retains **5,864** tracklets across `scen2` and `scen3`.
- `image_number` remains a source image index, **not a verified timestamp**.
- View semantics are `scene_local_view`, not a cross-dataset physical camera.
- Source/scene balance and sequential redundancy remain sampler concerns.
### Occluded-PoseTrack-ReID
- Kept **17,898 train crops / 1,000 identities**.
- Training data spans **474 logical videos**.
- `logical_camid` is a **video proxy**, not evidence of a physical camera.
- `posetrack_image_id` and annotation IDs are preserved as source metadata;
  they are not treated as verified temporal frame numbers.
- Identity keys include the **train partition**, because reused numeric PIDs
  across training and evaluation can refer to different actual people.
- One same-person duplicate pair was observed, but no mandatory hard
  deletion was applied for this pool.
### ORBench RGB
- Current experiment uses **RGB/VIS**, not the other ORBench modalities.
- The original RGB training set had **27,063** rows.
- Existing audited cleaning excluded **111** landscape outliers and **20**
  redundant exact copies, for **131 exclusions** total.
- The clean candidate pool has **26,932 images / 600 identities**.
- Sub-source composition: **LLCM 9,643**, **SYSU 17,289**.
- `orbench_id` is the current pool identity; original sub-source person IDs
  and JSON labels are preserved for provenance.
- Camera context is scoped by the original sub-source, e.g. LLCM versus SYSU.
- LLCM frame numbers are retained where supported; SYSU fields are not invented.
- **Do not clean this pool a second time** with the same exclusions.
### PRID2011
- Multi-shot **matched identities only**: **40,033 images / 200 IDs**.
- **23,330** images from `cam_a` and **16,703** from `cam_b`.
- These identities yield **400 camera-specific trajectories**.
- Single-shot content duplicates images already found in multi-shot, so the
  two acquisition variants are not combined as independent examples.
- Camera-only identities remain in the original data, not this initial pool.
- No fixed official supervised source train/validation split was adopted.
- Unified rows therefore have **`source_split = unassigned`**.
- A custom identity-disjoint source protocol must be decided before training.
### RPIfield
- Original pool: **601,581 images / 4,108 identities** across 12 cameras.
- Audit identified **1,262 cross-person pixel-conflict rows** and one
  landscape anomaly for **1,263 hard-exclusion flags**.
- After these flags: **600,318** dataset-wide candidates, not an approved
  supervised training set.
- Our conservative initial pool uses **known actors only**.
- Known-actor pool: **211,484 images / 112 identities** after 160 exclusions.
- The **3,996 original distractor identities** were not automatically used
  as supervised classes; three are fully removed by exclusion flags.
- High temporal redundancy remains a sampling and protocol issue.
- `appearance_label` is **not automatically** a clothes label, and
  `sequence_id` is **not automatically** a tracklet identifier.
- Unified rows have **`source_split = unassigned`** pending protocol design.
### PRW — audited but deliberately excluded
PRW was audited as the **tenth dataset**; it is **not** one of the nine
candidate training sources.

A specific acquisition-overlap investigation found **3,886 of 5,663 PRW
source video frames (68.62%)** exactly overlapping Market1501 raw frames.
The overlap affected a substantial portion of PRW source crops and identities.

Because **Market1501 is intended as an unseen evaluation target**, PRW is
excluded at the **experiment source-role level**. Merely dropping individual
exact matches does not establish acquisition independence.

This is a **leakage-control decision**, not a judgment that PRW is poor data.
The original PRW audit and evidence remain useful research records.

---

## 5. Data Engineering Principles
Our current pipeline follows these constraints:

1. **Raw images are immutable.** Never edit, rename, or overwrite source images.
2. **Official evaluation protocols are immutable.** Do not relabel query/gallery.
3. **Auditing and cleaning are separate.** An audit flag is not automatically a
   new training-label decision unless the cleaning policy authorizes it.
4. **Hard exclusions require evidence.** Prefer documented row-level rules.
5. **Difficult samples are not automatically bad samples.** Occlusion, small
   boxes, viewpoint changes, and clothing changes may be valuable for DG.
6. **Identity IDs are source-scoped.** Raw ID `1` is not globally unique.
7. **Camera semantics are explicit.** A view, logical video, and physical camera
   are not interchangeable fields.
8. **Unsupported metadata stays missing.** Never invent a frame or clothes ID.
9. **Paths are portable.** Store dataset-root keys plus relative image paths.
10. **Derived artifacts are versioned and validated.** Keep previous versions.
### Four distinct duplicate concepts
- **Filename collision:** same basename does not establish same image.
- **File-byte duplicate:** exact bytes match.
- **Decoded-pixel duplicate:** image contents match even if encoding differs.
- **Near duplicate:** images are visually similar but not necessarily equal.

These categories cannot be substituted for one another in a leakage claim.

---

## 6. Repository Layout
This tree describes the current research workflow. Some directories are
**local-only** and deliberately excluded from Git; they are shown for context.
```text
DG_ReID_Audit/
├── .gitignore                       # Git tracking policy
├── README.md                        # Project orientation (this file)
├── scripts/
│   ├── CUHK03_NP/                  # Existing per-dataset audit tools, as present
│   ├── DeepChange/                 # Existing per-dataset audit tools
│   ├── LPW/                        # Existing per-dataset audit tools
│   ├── .../                        # Other audit scripts
│   └── unified/
│       ├── schema.py               # Schema v1: 19 columns and header check
│       ├── build_unified.py        # Registered adapter runner / CSV writer
│       ├── merge_unified.py        # Merge nine versioned unified manifests
│       ├── validate_master.py      # Independent row-by-row master validation
│       └── adapters/
│           ├── cuhk03.py
│           ├── iust.py
│           ├── last.py
│           ├── deepchange.py
│           ├── lpw.py
│           ├── occluded_posetrack.py
│           ├── orbench.py
│           ├── prid2011.py
│           └── rpifield.py
├── datasets/                        # LOCAL ONLY: original image data
├── manifests/                       # LOCAL ONLY: audit and candidate CSVs
├── reports/                         # LOCAL ONLY: audit reports and evidence
└── unified/                         # LOCAL ONLY: generated unified manifests
    ├── v0/                          # Early 17-column CUHK03 prototype
    └── v1/
        ├── cuhk03_np_unified_v1.csv
        ├── iust_unified_v1.csv
        ├── last_unified_v1.csv
        ├── deepchange_unified_v1.csv
        ├── lpw_unified_v1.csv
        ├── occluded_posetrack_unified_v1.csv
        ├── orbench_rgb_unified_v1.csv
        ├── prid2011_unified_v1.csv
        ├── rpifield_unified_v1.csv
        └── all_sources_unified_v1.csv
```
A few names under older dataset-specific audit-script directories may differ
from this illustrative tree; consult the actual committed `scripts/` directory.
### Git policy at the time of this README
The existing `.gitignore` excludes these entire directories:
```gitignore
/datasets/
/manifests/
/reports/
/unified/
```
It also ignores environments, caches, model weights, notebook files, archives,
temporary files, credentials, and several large binary formats.

Consequently, **cloning the code alone is not enough to reproduce the
940,809-row data build**. A collaborator also needs authorized access to the
required dataset files, manifests, and audit evidence.

A future release should provide a documented, rights-compliant artifact
manifest and reproducible data-access instructions. Do **not** bypass these
requirements by indiscriminately committing raw datasets or full CSV dumps.

---

## 7. Unified Schema v1
All nine individual outputs and the master share the same **19 ordered columns**.

| # | Column | Meaning |
| ---: | --- | --- |
| 1 | `sample_uid` | Stable ID derived from dataset name and relative path |
| 2 | `dataset` | Canonical dataset name |
| 3 | `candidate_pool` | Source candidate manifest used for conversion |
| 4 | `dataset_root_id` | Key used to locate the dataset root |
| 5 | `relative_path` | Portable image path relative to that root |
| 6 | `source_split` | Original train role or `unassigned` |
| 7 | `raw_pid` | Original or audited source-scoped person identifier |
| 8 | `identity_namespace` | Scope that prevents ID collisions |
| 9 | `identity_key` | Namespace-qualified identity identifier |
| 10 | `raw_camera_or_view` | Original camera/view/video proxy code |
| 11 | `camera_semantics` | Meaning of the preceding code |
| 12 | `camera_context_id` | Source-scoped camera or view context, if supported |
| 13 | `video_id` | Original video code, if available |
| 14 | `tracklet_id` | Original trajectory/tracklet code, if supported |
| 15 | `frame_id` | Verified source frame identifier, if available |
| 16 | `clothes_id` | Source clothing label, if available |
| 17 | `source_row_number` | Original candidate-CSV line number |
| 18 | `source_image_index` | Dataset-specific image or sequence index |
| 19 | `source_metadata_json` | Source metadata not otherwise represented |
### Example identity keys
```text
CUHK03_NP::detected::train::1
IUST_PersonReID::train::1
LaST::train::2
LPW::train::scen2-10
Occluded-PoseTrack-ReID::train::2125
PRID2011::matched_multishot::shared-0001
RPIfield::known_actor::1
```
These are **source identity keys**, not the final contiguous classification
labels for training. The latter are planned for **P5**, after source selection
and safe train/validation protocols are frozen.
### Stable sample UID versus pixel hash
`sample_uid` is computed from a dataset label plus the portable relative path,
using SHA-256. It is **not** a content hash of the image pixels.

Two identical images stored at two different paths can have different UIDs.
Cross-dataset duplicates therefore still require independent byte, decoded-
pixel, near-duplicate, and acquisition-overlap checks.
### Camera semantics: a critical caveat
| Dataset | Current semantics | Important limitation |
| --- | --- | --- |
| CUHK03 | `pair_local_view` | `c1` across different pairs is not a global camera |
| IUST | `physical_camera_code` | Camera IDs are valid only inside IUST |
| LaST | `not_available` | No verified physical camera ID inferred |
| DeepChange | `physical_camera_code` | Camera context is scoped to DeepChange |
| LPW | `scene_local_view` | Scene/view is not assumed globally physical |
| Occluded-PoseTrack | `logical_video_proxy` | Video ID is not a physical camera |
| ORBench | `source_scoped_camera_code` | LLCM and SYSU camera namespaces differ |
| PRID2011 | `physical_camera_code` | Two known cameras: `cam_a` and `cam_b` |
| RPIfield | `physical_camera_code` | Twelve dataset-local camera codes |

Future sampling code must consult `camera_semantics` instead of assuming every
non-empty camera field has the same physical interpretation.
### Explicit missing values
Empty cells mean information is **not available or not validated** in that
source. They do not automatically mean zero.

For instance, LPW image number is not a confirmed video frame timestamp,
and PRID2011 sequence index is an ordering within a trajectory.

---

## 8. How the Unified Pipeline Works
### Overview
```text
manifests/<dataset>/<clean_candidates>.csv
                 |
                 v
     scripts/unified/adapters/<dataset>.py
                 |
                 v
       adapt_row(source_row, row_number)
                 |
                 v
        Unified Schema v1 dictionary
                 |
                 v
        scripts/unified/build_unified.py
                 |
                 v
     unified/v1/<dataset>_unified_v1.csv
                 |
                 v
        scripts/unified/merge_unified.py
                 |
                 v
      unified/v1/all_sources_unified_v1.csv
                 |
                 v
       scripts/unified/validate_master.py
```
### Responsibilities
**`schema.py`**

- Defines the 19-column contract and version.
- Validates CSV header names and their order.
- Can check the existing CUHK03 v1 reference header.

**Dataset adapters**

- Validate dataset-specific fields, training/candidate role, and path structure.
- Convert source rows into the 19-column dictionary.
- Preserve source-specific information in `source_metadata_json`.
- Support read-only validation when run as Python modules.
- Do **not** write final CSVs themselves.

**`build_unified.py`**

- Holds the current `DATASETS` registry of nine configured adapters.
- Streams source rows through each adapter and checks counts/identity keys.
- Writes to a temporary file before publishing the CSV.
- Computes a SHA-256 checksum of the final output.
- Refuses to overwrite an existing output file.

**`merge_unified.py`**

- Reads the nine previously created unified files in a fixed source order.
- Rechecks schema, provenance, sample UID construction, and row totals.
- Emits the master candidate CSV only after validation.
- Refuses to overwrite an existing master manifest.

**`validate_master.py`**

- Reopens all individual unified files and the published master.
- Checks each master row against the corresponding source row, all 19 fields.
- Checks total counts, source-level identity counts, UID/path uniqueness,
  split conventions, metadata parsing, and the published master checksum.
- Does not modify any file.
### What validation does not prove
A complete adapter and master validation **does not prove**:

- There are no visually or temporally related samples across datasets.
- Source and target acquisition conditions are fully independent.
- A source's person IDs are correct beyond the audited evidence.
- Training and validation partitions are already identity-disjoint.
- A source will improve generalization when included in training.

Those are separate research questions addressed by later phases.

---

## 9. Working with the Code
### Development environment
The verified workflows were run on **Lightning Studio** in a project folder:
```bash
cd ~/DG_ReID_Audit
```
Python's standard library is sufficient for the unified CSV conversion
modules shown here (`csv`, `pathlib`, `hashlib`, `json`, etc.).

The original audit and benchmarking utilities may need additional libraries.
**There is not yet a frozen, fully validated project-wide requirements file**;
do not assume an arbitrary `pip install -r requirements.txt` command works.

Run commands from the project root so `scripts.unified` imports resolve.
### Inspect the canonical header
```bash
python scripts/unified/schema.py
```
Expected final line:
```text
CUHK03 header validation: PASS
```
This checks the reference header, not all rows.
### Validate an adapter without generating a CSV
```bash
python -m scripts.unified.adapters.cuhk03
python -m scripts.unified.adapters.iust
python -m scripts.unified.adapters.last
python -m scripts.unified.adapters.deepchange
python -m scripts.unified.adapters.lpw
python -m scripts.unified.adapters.occluded_posetrack
python -m scripts.unified.adapters.orbench
python -m scripts.unified.adapters.prid2011
python -m scripts.unified.adapters.rpifield
```
These commands depend on their original candidate manifests, evidence files,
and image roots being present in the expected workspace structure.
### Build one dataset's Unified Manifest
For a source **without an existing output file**:
```bash
python -m scripts.unified.build_unified --dataset iust
```
Replace `iust` with another registered source key as needed.

**Important:** This is a creation command, not an idempotent overwrite.
It intentionally raises `FileExistsError` if the target CSV already exists.
Do not delete a validated result just to make the command succeed.
### Create the master candidate manifest
Once all nine individual outputs exist:
```bash
python -m scripts.unified.merge_unified
```
The master builder also refuses to overwrite its existing output.
It is not a command that should be rerun casually on the current published v1.
### Validate the published master
```bash
python -m scripts.unified.validate_master
```
The current validator intentionally checks a **specific frozen v1 checksum**.
If a future legitimate source change produces new files, it must be handled
as a reviewed, versioned release—not by silently disabling the checksum.
### Path configuration and portability
Every row contains `dataset_root_id` and `relative_path` instead of depending
on a Lightning-specific absolute path in the unified output.

For example:
```text
dataset_root_id = CUHK03_NP
relative_path   = extracted/cuhk03-np/detected/bounding_box_train/0001_c1_1.png
```

A future DataLoader can resolve paths using a machine-local root registry:
```python
image_path = dataset_roots[row["dataset_root_id"]] / row["relative_path"]
```

This is a **design illustration**, not a claim that the final training
DataLoader or its root configuration file has already been implemented.

---

## 10. Reproducibility and Validated Artifacts
### Master Candidate Manifest v1
**Location on the research workspace:**
```text
unified/v1/all_sources_unified_v1.csv
```

**Published local SHA-256 (12 October 2026):**
```text
4fba9e5578b19679275f061dc97740518d300830a0306b725c39e0dfc15e34ce
```

Independent validation reported:
```text
Datasets verified:         9
Rows compared:       940,809
Unique sample UIDs:  940,809
Source-scoped IDs:    11,297
Schema columns:           19
All row comparisons:    PASS
MASTER VALIDATION:       PASS
```
### Recorded per-source v1 checksums
| Source | SHA-256 of local Unified v1 CSV |
| --- | --- |
| CUHK03-NP | Not recorded in this README; validate against local artifact |
| IUST | `ae577916bf2fd79cdb275b6c36bf2b223243ae4228af16499b7f28a2008a3503` |
| LaST | `10f69e3cc005db46f111f49ae9f1872b01b0cc74713c744e61c08dc545f1d256` |
| DeepChange | `4df5ecab5acc69909c98616f5d0d9320606976f6c94ffad68fb793e721d17a92` |
| LPW | `0f9c8d5808b3b935750dae0136ac88b4670e514c323d75cd28f0cd78969a0354` |
| Occluded-PoseTrack | `fdd4c5f501864cdb985113c78588066fd5f5224e16b5ef77dbfe55b463a096cb` |
| ORBench RGB | `3ea915f1e43198acafb582895819c6111fea0f17004ca9252f871eb4221d30dc` |
| PRID2011 | `f9e5877bf53da59f525d863a4425f0b97939dc890cff4f49fd207f4e6824f6b7` |
| RPIfield | `4c0de5d7fced5738b24d97c3fb1309977763b791f18289cf4f3c1478eb7cddcc` |

These hashes identify **specific local artifacts**; they are not pixel hashes,
source licensing statements, or claims of cross-domain independence.
### Versioning approach
- `unified/v0/` preserves the earlier 17-column CUHK03 prototype.
- `unified/v1/` is the validated 19-column schema and current candidate set.
- Changes in data policy, protocol, or schema should create a reviewed new
  version or an explicitly documented migration rather than silent edits.
- Do not mix old outputs with new metadata rules under identical filenames.
### Reproducibility still to be completed
- Dataset access and license inventory for collaborators.
- Machine-readable checksums for ignored audit and source candidate files.
- A frozen runtime/environment specification for all research stages.
- A versioned protocol and sampler configuration.
- Deterministic train/validation and class-ID assignment.
- Experiment seeds, model/checkpoint provenance, and full training logs.
- Final benchmark tables with traceable evaluation scripts.

---

## 11. Experimental Safety and Leakage Policy
### Already confirmed
The **PRW–Market1501** acquisition overlap is substantial and documented.
PRW is consequently excluded while Market1501 remains an unseen target.
### Not yet confirmed
The other nine candidate pools have **not** received a complete, frozen
cross-source and source-target leakage clearance for the final experiment.

Planned checks must distinguish:

- Exact bytes versus decoded-pixel identity.
- Visually near-identical crops derived from the same acquisition.
- Shared underlying datasets or original source datasets.
- Video/frame/time overlap where acquisition metadata permits it.
- Person identity overlap and source/target data-role overlap.

Special attention is needed when a dataset repackages another collection,
such as ORBench's LLCM and SYSU origin components.
### Never optimize against a contaminated target
Do not select training sources, tune hyperparameters, or choose checkpoints
because of repeated access to the unseen target test labels.

If evidence of contamination is found, document and review it before the
training manifest is frozen. Preserve the original audit evidence.

---

## 12. What Happens Next
### P3 — Source training/validation protocols
Specify which identities and samples are eligible for source training,
which are reserved for validation, and how these choices are reproduced.

Particular open questions:

- PRID2011: use all 200 matched identities or reserve some for validation?
- RPIfield: how to define an identity-disjoint known-actor protocol?
- What source validation policy avoids repeated target-test tuning?
- How are custom splits seeded, stored, and checked for collisions?
### P4 — Cross-dataset leakage safety
Run source-versus-target and relevant source-versus-source checks; record
source accept/exclude decisions with evidence and limitations.
### P5 — Freeze source selection and labels
Choose a final set of valid source domains and build a deterministic mapping
from source-scoped identity keys to contiguous training class IDs.

The **nine candidate datasets need not all be used** in every experiment.
### P6 — Multi-source sampler and budget
Control imbalances across datasets and people, and avoid sampling enormous
numbers of redundant near-neighbor frames.

Compare sampling choices such as identity balance, source quotas, camera/view
context, temporal separation, and (where available) clothing diversity.

Define batch construction, optimizer steps, epoch semantics, oversampling
policy, seed, and compute budget before launching long training runs.
### P7 — Fine-tuning and ablations
Planned comparisons include:

- Pretrained-only feature extractor baseline.
- Controlled single-source fine-tuning baseline(s).
- Balanced multi-source supervised fine-tuning.
- Sampler and source-composition ablations.
- Repeated seeds or other uncertainty analysis where feasible.

No result of these experiments is claimed as complete here.
### P8 — Unseen-domain evaluation
Evaluate using the target's appropriate official setup, including **mAP**
and **CMC Rank-1**, and clearly distinguish image-based from tracklet-based
protocols. Archive configs, checkpoints, rankings, and statistical analyses.

---

## 13. Background and Team Collaboration
Earlier work explored CION/ReIDZoo benchmarking with GhostNet, EdgeNet, and
RepViT model families. Model-loading and embedding smoke tests were performed,
but the full final benchmark matrix and multi-source fine-tuning results have
**not** been verified as complete.
### Contribution workflow
1. Open a scoped issue or agree on the research question and evidence.
2. Implement one focused, reviewable change at a time.
3. Maintain correct PID, camera, video, frame, and split semantics.
4. Test against the corresponding source audit and candidate manifest.
5. Preserve raw images, official evaluation protocols, and earlier versions.
6. Record the important output counts, SHA-256 checksums, and limitations.
7. Have a teammate review a change before promoting it to the baseline.
### Quick answers for new team members
**Is the master CSV the final training set?** No. It is the versioned
**candidate** metadata pool, prior to final leakage and protocol clearance.

**Does the repository contain raw images?** No; `datasets/`, `manifests/`,
`reports/`, and `unified/` are locally ignored by Git at this stage.

**Why keep nine individual files after merging?** To validate each adapter,
trace provenance, and isolate failures without reprocessing all sources.

**Why is PRW excluded?** Verified acquisition overlap with the Market1501
evaluation target threatens the intended unseen-domain experiment.

**Why are PRID2011/RPIfield marked `unassigned`?** Their supervised source
train/validation protocols have not yet been frozen.

**Has multi-source training started?** No. Dataset engineering is complete
for v1; the training experiment is still being designed.
### Responsible research and publication
Respect dataset-specific permissions, privacy requirements, and ethics rules.
Do not publish raw surveillance imagery, credentials, authenticated download
URLs, restricted annotations, or identifiable private metadata without proper
rights and authorization.

For a research paper, report only what the evidence supports: a unified
metadata validation **does not establish** cross-domain independence, a
finished benchmark, or a model accuracy improvement.

---

## 14. Current Handoff
**Last verified operation:**
```bash
python -m scripts.unified.validate_master
```

**Observed result:**
```text
MASTER VALIDATION: PASS
```

**Research engineering milestone reached:**

- Ten audits and their evidence exist in the working environment.
- Nine candidate pools have been locally prepared and path-validated.
- Nine adapters and individual Unified v1 CSVs have been validated.
- The nine sources were merged into one independently validated master CSV.
- The master content is versioned and identified by its SHA-256 checksum.

**Immediate project-management activity:** establish a clean, private Git
repository, review tracked files, and commit the source code and this README.

**Next research milestone:** **P3 — training/validation protocol design**, in
coordination with **P4 — cross-dataset leakage safety**.

The intent is to produce a research artifact that the entire team can
understand, inspect, reproduce under the proper data permissions, and cite
accurately when the experimental results are ready.

---

**Maintained for:** the DG-ReID research team  
**Snapshot date:** 12 October 2026  
**Stage:** Candidate data engineering complete (v1); experiment protocols pending
