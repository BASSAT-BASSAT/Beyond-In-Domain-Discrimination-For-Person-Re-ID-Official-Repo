from pathlib import Path
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

QUALITY = (
    BASE
    / "manifests/ORBench/orbench_rgb_quality_manifest.csv"
)

FLAGS = (
    BASE
    / "manifests/ORBench/orbench_rgb_audit_flags.csv"
)

SOURCE_OUT = (
    BASE
    / "manifests/ORBench/orbench_rgb_clean_source_manifest.csv"
)

REPORT_OUT = (
    BASE
    / "reports/ORBench/ORBench_Audit_Report.md"
)


quality = pd.read_csv(
    QUALITY,
    low_memory=False
)

flags = pd.read_csv(
    FLAGS,
    low_memory=False
)


# ============================================================
# BUILD CLEAN TRAINING SOURCE
# ============================================================

exclude_paths = set(
    flags.loc[
        flags["recommended_training_exclusion"] == True,
        "file_path",
    ].astype(str)
)


source = quality[
    quality["split"] == "train"
].copy()


source["recommended_training_exclusion"] = (
    source["file_path"]
    .astype(str)
    .isin(exclude_paths)
)


clean = source[
    ~source["recommended_training_exclusion"]
].copy()


if len(source) != 27063:
    raise RuntimeError(
        f"Expected 27063 official train rows, got {len(source)}"
    )

if len(clean) != 26932:
    raise RuntimeError(
        f"Expected 26932 clean source rows, got {len(clean)}"
    )


clean.to_csv(
    SOURCE_OUT,
    index=False
)


# ============================================================
# SUMMARY NUMBERS
# ============================================================

test = quality[
    quality["split"] == "test_gallery"
].copy()


train_origin = (
    source.groupby(
        "source_dataset"
    )
    .agg(
        images=("file_path", "size"),
        ids=("json_label", "nunique"),
    )
)


clean_origin = (
    clean.groupby(
        "source_dataset"
    )
    .agg(
        images=("file_path", "size"),
        ids=("json_label", "nunique"),
    )
)


landscape_train = int(
    (
        (quality["split"] == "train")
        &
        (quality["width"] > quality["height"])
    ).sum()
)

landscape_test = int(
    (
        (quality["split"] == "test_gallery")
        &
        (quality["width"] > quality["height"])
    ).sum()
)

square_train = int(
    (
        (quality["split"] == "train")
        &
        (quality["width"] == quality["height"])
    ).sum()
)

square_test = int(
    (
        (quality["split"] == "test_gallery")
        &
        (quality["width"] == quality["height"])
    ).sum()
)

narrow_train = int(
    (
        (quality["split"] == "train")
        &
        (quality["aspect_ratio"] < 0.18)
    ).sum()
)

narrow_test = int(
    (
        (quality["split"] == "test_gallery")
        &
        (quality["aspect_ratio"] < 0.18)
    ).sum()
)


dup_sizes = (
    quality.groupby(
        "pixel_sha256"
    )
    .size()
)

dup_hashes = set(
    dup_sizes[
        dup_sizes > 1
    ].index
)

dups = quality[
    quality["pixel_sha256"].isin(
        dup_hashes
    )
].copy()


train_dup_groups = 0
test_dup_groups = 0

for _, g in dups.groupby(
    "pixel_sha256"
):

    if (
        g["split"] == "train"
    ).sum() > 1:
        train_dup_groups += 1

    if (
        g["split"] == "test_gallery"
    ).sum() > 1:
        test_dup_groups += 1


# ============================================================
# REPORT
# ============================================================

report = f"""# ORBench RGB Dataset Audit Report

## 1. Audit Status

**Dataset:** ORBench  
**Modality audited:** RGB / VIS only  
**Audit status:** COMPLETE  
**Raw dataset:** Preserved unchanged  

### Final project role

**Accepted as a supervised DG source candidate.**

Only the RGB/VIS official training split is used.

ORBench must be treated as a composite source because its RGB imagery originates from two source datasets:

- LLCM
- SYSU-MM01

These source namespaces must remain distinct during sampling and analysis.

---

## 2. Filesystem Structure

The distributed ORBench package contains:

- `vis/`
- `nir/`
- `cp/`
- `sk/`
- `train_annos.json`
- `test_gallery_and_queries.json`

The current project uses only:

`vis/`

The NIR, color-pencil, sketch, and text modalities are not used in the current RGB DG training experiment.

---

## 3. RGB Dataset Size

Total RGB/VIS images:

**45,113**

Official split:

- Train: **27,063 images**
- Test RGB gallery: **18,050 images**

Identity split:

- Train: **600 identities**
- Test: **400 identities**

Total:

- **1,000 ORBench identities**

All 45,113 RGB images are covered by the official JSON protocol.

No RGB filesystem images are left outside the protocol, and no JSON-referenced RGB files are missing.

---

## 4. Label Mapping

The numeric `id` field in the JSON files is not generally equal to the ORBench folder/image identity number.

This is an official label remapping rather than an annotation error.

Validation found:

- every JSON label maps to exactly one ORBench identity
- every ORBench identity maps to exactly one JSON label
- folder ID and filename ORBench ID always match

Therefore the remapped JSON labels are internally consistent.

---

## 5. Train / Test Identity Separation

Train and test separation is clean.

Results:

- ORBench-ID train/test overlap: **0**
- Original source-identity overlap: **0**

The second check compared identities using:

`(source_dataset, original_source_pid)`

This is important because ORBench remaps identities from LLCM and SYSU.

No original person identity is shared between the official train and test splits.

---

## 6. RGB Source Origins

ORBench RGB is a composite dataset built from LLCM and SYSU imagery.

### Official training split

| Origin | Images | Identities |
|---|---:|---:|
| LLCM | 9,650 | 301 |
| SYSU | 17,413 | 299 |

Total:

- 27,063 images
- 600 identities

The camera namespaces are source-specific.

For example:

`LLCM camera 1 != SYSU camera 1`

Therefore the correct camera identity is conceptually:

`(source_dataset, camera_id)`

ORBench should not be treated as an independently captured single-domain dataset.

---

## 7. Image Integrity

All RGB images were decoded successfully.

Results:

- Readable RGB images: **45,113**
- Corrupt/unreadable images: **0**
- PIL mode: RGB for all images

No corruption exclusion was required.

---

## 8. Image Geometry

RGB crop geometry is highly variable.

Width:

- Min: 21 px
- Median: 87 px
- Mean: 95.53 px
- Max: 1,198 px

Height:

- Min: 49 px
- Median: 227 px
- Mean: 239.51 px
- Max: 879 px

Aspect ratio:

- Median: 0.388
- Mean: 0.403
- Maximum: 4.870

Unique width/height combinations:

**17,512**

This reflects the heterogeneous acquisition origins of ORBench.

---

## 9. Landscape Geometry Outliers

Total landscape RGB images:

**186**

Distribution:

- Train: **{landscape_train}**
- Test/gallery: **{landscape_test}**

Training landscape images by origin:

- LLCM: 7
- SYSU: 104

Visual inspection showed that many of these landscape samples are background-heavy or are not clean person-centric ReID crops.

### Training policy

The **111 landscape training images** are excluded from the custom training source manifest.

### Evaluation policy

The **75 test/gallery landscape images remain untouched**.

The official evaluation data is not modified.

Raw files are never deleted.

---

## 10. Square and Extremely Narrow Images

Square images:

- Train: **{square_train}**
- Test: **{square_test}**

Extremely narrow images with aspect ratio < 0.18:

- Train: **{narrow_train}**
- Test: **{narrow_test}**

These samples are flagged but are not automatically excluded.

Visual inspection showed that many small or narrow crops still contain valid pedestrians.

The audit therefore avoids deleting difficult examples solely because of crop geometry.

---

## 11. Exact Duplicate Audit

Exact decoded RGB-pixel duplicate audit found:

- Duplicate groups: **27**
- Images in duplicate groups: **54**
- Extra copies: **27**

All groups contain exactly two images.

Duplicate safety checks:

- Cross-split groups: **0**
- Cross-ORBench-ID groups: **0**
- Cross-original-identity groups: **0**

Therefore no identity conflict or train/test leakage is caused by these duplicates.

Distribution:

- Train duplicate groups: **{train_dup_groups}**
- Test duplicate groups: **{test_dup_groups}**

### Training policy

For the 20 training duplicate groups:

- keep one deterministic representative
- exclude one extra copy

Therefore:

**20 redundant training rows are excluded.**

### Test policy

The 7 duplicate groups in the official test/gallery split remain unchanged.

---

## 12. Training Identity Balance

Official train:

- 600 identities
- 27,063 images

Images per identity:

- Minimum: 8
- Median: 28.5
- Mean: 45.1
- 95th percentile: 104
- Maximum: 153

No training identity has only one image.

No training identity has fewer than four images.

However, image counts vary substantially, so identity-balanced sampling is recommended.

---

## 13. Cross-Camera Coverage

Across the training set, mean camera coverage is approximately:

**2.32 cameras per identity**

Overall distribution:

- 1 camera: 176 identities
- 2 cameras: 237 identities
- 3 cameras: 26 identities
- 4 cameras: 150 identities
- 5 cameras: 8 identities
- 6 cameras: 1 identity
- 7 cameras: 1 identity
- 8 cameras: 1 identity

### LLCM

- 301 train identities
- mean camera coverage: about 1.69
- median: 1 camera
- maximum: 8 cameras

Many LLCM identities are single-camera.

### SYSU

- 299 train identities
- mean camera coverage: about 2.94
- median: 2 cameras
- maximum: 4 cameras

SYSU provides stronger cross-camera coverage.

Visual audit confirmed reasonable identity consistency across available camera views.

---

## 14. Temporal Metadata

Temporal metadata availability differs strongly by source.

### LLCM

All LLCM RGB images contain parseable:

- camera
- sequence
- frame number

### SYSU

Camera information is available, but filenames do not provide usable sequence/frame-number metadata in the same form.

Therefore temporal redundancy can be measured directly for LLCM but not for SYSU using the current filenames.

This limitation is explicitly retained in the audit rather than inferred or artificially reconstructed.

---

## 15. LLCM Temporal Redundancy

Official ORBench training contains:

**9,650 LLCM RGB images**

Adjacent same-identity / same-camera / same-sequence frame gaps:

- Median gap: **15**
- 94.81% have gap <= 20
- 98.98% have gap <= 50
- 99.38% have gap <= 100

This is strong temporal redundancy.

Therefore ORBench training should not sample LLCM frames uniformly without temporal awareness.

Recommended training strategy:

- identity-balanced sampling
- source-aware sampling
- camera diversity when possible
- temporal spacing / near-neighbor suppression for LLCM

No raw frames are deleted because of temporal redundancy.

---

## 16. Visual Audit

Visual audit covered:

- source/camera overview
- smallest crops
- extreme aspect-ratio crops
- exact duplicate pairs
- same identities across cameras

Findings:

- both indoor and outdoor conditions are present across the composite source
- strong variation exists in viewpoint, scale, illumination, and background
- some very small pedestrian crops remain valid
- exact duplicate pairs are visually consistent with the hash audit
- cross-camera identity appearance is generally consistent
- landscape outliers are often background-heavy and unsuitable as person-centric training crops
- SYSU contributes stronger multi-camera coverage than LLCM

---

## 17. Custom Training Flags

Training exclusions are applied only through reproducible metadata flags.

### Hard custom-training exclusions

- Landscape training outliers: **111**
- Redundant exact duplicate copies: **20**

Total:

**131 training rows**

Official RGB train:

**27,063**

Final custom clean source:

**26,932 RGB images**

Identity count remains:

**600 identities**

No raw image is deleted or modified.

---

## 18. Final Training Source

Final source manifest:

`orbench_rgb_clean_source_manifest.csv`

Final size:

- **26,932 RGB images**
- **600 identities**

Origin identities:

- LLCM: 301 IDs
- SYSU: 299 IDs

The source remains a composite LLCM + SYSU acquisition domain.

---

## 19. DG Role Decision

### Accepted as DG source candidate

ORBench RGB provides useful variation in:

- acquisition source
- camera
- scale
- viewpoint
- indoor/outdoor appearance
- image resolution

Internal train/test identity separation is clean.

No cross-split exact duplicates were found.

No original-source identity leakage was found.

### Important interpretation constraint

ORBench must not be counted as fully independent from LLCM or SYSU if either of those original datasets is also used separately as another training source.

Doing so would double-count the same acquisition domains and potentially the same underlying imagery.

For the current project, ORBench should be represented as:

**ORBench-RGB composite source: LLCM + SYSU**

rather than as an unrelated new camera network.

---

## 20. Final Status

| Audit item | Result |
|---|---|
| RGB protocol coverage | Complete |
| Missing RGB files | 0 |
| Corrupt RGB files | 0 |
| Train/test ORBench-ID overlap | 0 |
| Original identity overlap | 0 |
| Exact duplicate groups | 27 |
| Cross-split duplicate groups | 0 |
| Landscape train outliers | 111 |
| Landscape test outliers | 75, preserved |
| Temporal redundancy | Strong in LLCM |
| SYSU temporal metadata | Not available from filenames |
| Official RGB train | 27,063 |
| Custom clean source | **26,932** |
| Train identities | **600** |
| Final DG role | **Accepted source candidate** |

---

# Final Conclusion

ORBench RGB passed the dataset audit as a usable supervised source for the current domain-generalization Person ReID project.

The official RGB protocol contains **27,063 training images from 600 identities** and **18,050 test/gallery images from 400 identities**, with no train/test identity leakage.

The RGB data is a composite of LLCM and SYSU rather than a newly captured independent domain.

The audit identified several important issues:

- 186 landscape geometry outliers
- 27 exact duplicate pairs
- strong temporal redundancy in LLCM
- unequal camera coverage between LLCM and SYSU
- no usable temporal sequence/frame metadata for SYSU

For custom supervised training, **131 rows are excluded reproducibly**, producing a final source manifest of:

**26,932 RGB images from 600 identities.**

Raw ORBench remains unchanged, and all discovered problems remain documented in the audit report.
"""


REPORT_OUT.write_text(
    report,
    encoding="utf-8"
)


print("=" * 100)
print("ORBENCH FINALIZATION")
print("=" * 100)

print(
    "Official train:",
    len(source)
)

print(
    "Excluded:",
    len(source) - len(clean)
)

print(
    "Final clean source:",
    len(clean)
)

print(
    "Final identities:",
    clean["json_label"].nunique()
)

print("\nClean source by origin:")
print(clean_origin)

print("\nSaved clean source manifest:")
print(SOURCE_OUT)

print("\nSaved final report:")
print(REPORT_OUT)

print("\nDONE")
