# CUHK03-NP — Dataset Audit Report

## 1. Overview

**Dataset:** CUHK03-NP  
**Task:** Person Re-Identification (ReID)  
**Type:** Conventional image-based ReID dataset  
**Available variants:** `detected` and `labeled`

Structure:

```text
cuhk03-np/
├── detected/
│   ├── bounding_box_train/
│   ├── query/
│   └── bounding_box_test/
└── labeled/
    ├── bounding_box_train/
    ├── query/
    └── bounding_box_test/
```

All audited files are PNG images.

---

## 2. Dataset Size

### Detected Variant

| Split | Images | IDs | Views |
|---|---:|---:|---:|
| Train | 7,365 | 767 | 2 |
| Query | 1,400 | 700 | 2 |
| Gallery | 5,332 | 700 | 2 |

Total detected images:

```text
14,097
```

### Labeled Variant

| Split | Images | IDs | Views |
|---|---:|---:|---:|
| Train | 7,368 | 767 | 2 |
| Query | 1,400 | 700 | 2 |
| Gallery | 5,328 | 700 | 2 |

Total labeled images:

```text
14,096
```

Total physical files across both variants:

```text
28,193
```

The two variants are alternative crops of the same benchmark and should not be treated as independent training datasets.

---

## 3. Filename Structure

Example:

```text
0001_c1_1.png
```

Parsed as:

```text
0001 → Person ID
c1   → View label
1    → Image index
```

The audit found only two view labels:

```text
c1
c2
```

All filenames were parsed successfully:

```text
Invalid filenames = 0
```

For this audit, `c1` and `c2` are treated as relative view labels rather than globally unique physical-camera IDs.

---

## 4. Identity and View Distribution

### Detected Training Split

```text
IDs    = 767
Min    = 6 images / ID
Mean   = 9.60
Median = 10
Max    = 10
Std    = 0.90
```

View distribution:

```text
View 1 = 3,601 images
View 2 = 3,764 images
```

Cross-view coverage:

```text
767 / 767 train IDs appear in both views
```

No train identity is limited to a single view.

### Detected Gallery Split

```text
IDs    = 700
Min    = 4 images / ID
Mean   = 7.62
Median = 8
Max    = 8
Std    = 0.81
```

View distribution:

```text
View 1 = 2,617 images
View 2 = 2,715 images
```

All 700 gallery identities appear in both views.

### Labeled Variant Note

The labeled split is very similar overall, but one training identity (`PID 1351`) appears only in View 2:

```text
PID 1351:
View 1 = 0 images
View 2 = 5 images
```

The detected variant contains both views for this identity.

---

## 5. Train/Test Integrity

Identity separation is clean in both variants:

```text
Train ∩ Query IDs   = 0
Train ∩ Gallery IDs = 0
```

Query and gallery contain the same 700 test identities:

```text
Query - Gallery IDs = 0
Gallery - Query IDs = 0
```

Therefore, the test identities are unseen during training.

---

## 6. Query/Gallery Protocol

For both variants:

```text
Query images = 1,400
Query IDs    = 700
```

Each test identity contributes:

```text
2 query images
```

with exactly:

```text
700 queries from View 1
700 queries from View 2
```

Filename overlap between splits:

```text
Train ∩ Query filenames   = 0
Train ∩ Gallery filenames = 0
Query ∩ Gallery filenames = 0
```

Every query has at least one valid gallery match:

```text
Queries with ZERO gallery positives       = 0
Queries with ZERO opposite-view positives = 0
```

For the detected variant, opposite-view positives per query are:

```text
1 positive  →   23 queries
2 positives →   51 queries
3 positives →   97 queries
4 positives → 1,229 queries
```

Summary:

```text
Mean   = 3.81
Median = 4
Min    = 1
Max    = 4
```

The evaluation protocol is therefore valid for cross-view ReID.

---

## 7. Image Quality

### Detected Variant

All images were successfully decoded:

```text
Readable images  = 14,097
Corrupted images = 0
RGB images       = 14,097
```

Training image geometry:

```text
Width:
Min    = 41 px
Mean   = 90.41 px
Median = 93 px
Max    = 185 px

Height:
Min    = 121 px
Mean   = 266.07 px
Median = 277 px
Max    = 515 px

Aspect ratio (W/H):
Min    = 0.252
Mean   = 0.340
Median = 0.336
Max    = 0.579
```

Typical crops are narrow, vertical person crops.

---

## 8. Detected vs. Labeled Visual Audit

Visual inspection showed that the two variants are highly similar in identity content and scene conditions.

### Detected

- Slight bounding-box and alignment noise
- Small variations in crop tightness
- Some additional background around the person
- No obvious large-scale detection failures in inspected samples

### Labeled

- Manually defined crops
- Generally clean person localization
- Still contains some extreme aspect-ratio cases
- One training identity lacks one of the two views

The difference is mainly in crop boundaries rather than identity or domain content.

---

## 9. Main Strengths

- Clean train/test identity separation
- Balanced identity representation
- Nearly balanced view distribution
- Every detected training identity appears in both views
- Valid cross-view query/gallery protocol
- No corrupted images
- Conventional low-resolution surveillance ReID setting
- Useful viewpoint and crop variation

---

## 10. Main Risks

- Relatively small training set
- Only two relative views per identity
- Limited images per identity
- Typical aspect ratio (~0.34) differs from common ReID model input ratios such as 128×256
- The two variants are highly redundant and must not be merged as separate source domains

---

## 11. DG Fine-Tuning Decision

**Current role:** Source candidate for multi-source Domain-Generalizable ReID.

### Selected Variant

```text
detected
```

Reason:

The detected version preserves realistic person-detector crop variation while maintaining clean identity labels, full cross-view coverage, and a valid evaluation protocol.

The `detected` and `labeled` variants should **not** be used together as independent datasets because they represent alternative crops of the same underlying benchmark.

### Training Consideration

Identity-balanced sampling is appropriate:

```text
P identities × K images
```

No dataset-specific cleaning is required at this stage.

---

# Final Audit Summary

| Property | Result |
|---|---|
| Selected Variant | Detected |
| Train Images | 7,365 |
| Train IDs | 767 |
| Test IDs | 700 |
| Query Images | 1,400 |
| Gallery Images | 5,332 |
| Relative Views | 2 |
| Train/Test PID Leakage | None |
| Query/Gallery Filename Overlap | None |
| Queries Without Cross-View Positive | 0 |
| Train Images / ID | 6–10 |
| Corrupted Images | 0 |
| Mean Train Aspect Ratio | 0.340 |
| Cross-View Train Coverage | 100% |
| DG Role | Source Candidate |

## Final Decision

**CUHK03-NP is retained as a source candidate for multi-source Domain-Generalizable Person Re-Identification, using the `detected` variant only.**

The dataset has a clean protocol, balanced identity distribution, complete cross-view training coverage, and realistic detector-crop variation. No image removal is required at this stage.
