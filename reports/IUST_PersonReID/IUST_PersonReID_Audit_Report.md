# IUST_PersonReID — Dataset Audit Report

## 1. Overview

**Dataset:** IUST_PersonReID  
**Task:** Person Re-Identification (ReID)  
**Type:** Image-based ReID dataset with strong temporal/sequence characteristics.

Structure:

```text
IUSTPersonReID/
├── bounding_box_train/
├── query/
└── bounding_box_test/
```

All audited files are JPEG RGB images and no corrupted images were found.

---

## 2. Dataset Size

| Split | Images | IDs | Cameras |
|---|---:|---:|---:|
| Train | 72,393 | 1,193 | 17 |
| Query | 1,428 | 654 | 17 |
| Test | 45,062 | 654 | 17 |

Total physical files: **118,883**

All query files are exact copies of images already present inside `bounding_box_test`, so the number of unique benchmark images is:

```text
72,393 + 45,062 = 117,455
```

---

## 3. Filename Structure

Example:

```text
0001_c2s1_024962_01.jpg
```

Parsed as:

```text
0001     → Person ID
c2       → Camera ID
s1       → Sequence field
024962   → Frame ID
01       → Suffix
```

In this dataset version:

- Sequence value is always `1`
- Suffix value is always `1`
- Useful metadata fields are mainly PID, Camera ID, and Frame ID
- Invalid filenames: **0**

---

## 4. Camera and Identity Distribution

The dataset contains **17 camera codes**:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
21, 22, 23, 24, 31, 32
```

Camera representation is noticeably imbalanced. For example:

```text
C2  → 16,053 train images
C4  → 13,667
C6  →  9,246

C23 →    507
C32 →    625
C11 →    683
```

Training images per identity:

```text
Min    = 8
Mean   = 60.68
Median = 56
Max    = 200
```

Camera coverage per training identity:

```text
2 cameras → 1,090 IDs
3 cameras →    73 IDs
4 cameras →    30 IDs
```

No training identity appears in only one camera.

---

## 5. Train/Test Integrity

No identity leakage was found:

```text
Train ∩ Query IDs = 0
Train ∩ Test IDs  = 0
```

So the test identities are unseen during training.

---

## 6. Query/Test Protocol

The audit showed:

```text
Query images = 1,428
Query IDs    = 654
```

There is exactly **one query image per (PID, Camera) pair**.

All query files are also present inside `bounding_box_test`, but every query has at least one valid cross-camera positive.

```text
Queries with ZERO cross-camera positives = 0
```

Cross-camera positives per query:

```text
Min    = 2
Mean   = 42.03
Median = 38
Max    = 227
```

This confirms that the evaluation protocol supports valid cross-camera ReID.

---

## 7. Temporal Redundancy

IUST contains strong temporal redundancy.

Training data contains:

```text
2,519 PID-camera groups
```

Images per PID-camera group:

```text
Mean   = 28.74
Median = 26
Max    = 50
```

Consecutive frame-gap analysis:

```text
gap <= 6   → 92.79%
gap <= 12  → 97.85%
gap <= 30  → 98.78%
```

The most common frame gap is:

```text
6 → 64,277 occurrences
```

This indicates that many images are temporally close and visually similar, so the raw image count overstates the true visual diversity.

---

## 8. Image Quality

All images were successfully decoded:

```text
Readable images  = 118,883
Corrupted images = 0
RGB images       = 118,883
```

Training image geometry:

```text
Width:
18 → 1,243 px
Median = 264 px

Height:
73 → 1,413 px
Median = 586 px

Aspect ratio (W/H):
0.134 → 4.507
Median = 0.436
```

This shows very large variation in crop size and shape.

Quality flags in the training split:

```text
tiny        = 2.54%
very_narrow = 0.91%
horizontal  = 3.36%
very_wide   = 0.02%
any extreme = 6.52%
```

Some cameras contain much more extreme crop geometry than others, especially cameras `21–24`. This suggests that many extreme crops are part of the camera/domain characteristics rather than random corruption.

---

## 9. Main Strengths

- 1,193 training identities
- 17 camera domains
- Every training identity appears in at least 2 cameras
- Valid cross-camera evaluation protocol
- Strong viewpoint, background, lighting, and crop variation
- Useful partial-body and low-resolution cases
- No corrupted images

---

## 10. Main Risks

- Strong temporal redundancy
- Unequal number of images per identity
- Camera imbalance
- Very large crop-size and aspect-ratio variation
- Small and partial-person crops are present

---

## 11. DG Fine-Tuning Decision

**Current role:** Strong source candidate for multi-source Domain-Generalizable ReID.

The dataset should not be used with naive uniform image sampling because temporally repeated samples and heavily represented identities/cameras could dominate training.

Recommended considerations for later training design:

```text
Dataset-balanced sampling
+
Identity-balanced P×K sampling
+
Camera-aware sampling
+
Temporal-diversity control
```

No images should be removed at this stage. Extreme samples should remain flagged as metadata until a global preprocessing policy is decided after auditing all datasets.
