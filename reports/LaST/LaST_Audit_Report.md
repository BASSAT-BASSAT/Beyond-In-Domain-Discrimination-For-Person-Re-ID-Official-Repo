# LaST — Dataset Audit Report

## 1. Overview

**Dataset:** LaST  
**Task:** Person Re-Identification (ReID)  
**Type:** Image-based ReID dataset with video/frame and clothes metadata

Dataset structure:

```text
last/
├── train/
├── val/
│   ├── query/
│   └── gallery/
├── test/
│   ├── query/
│   └── gallery/
└── readme.txt
```

Filename schema:

```text
PID_IMAGEID_VIDEOID_FRAMEID_BBOXID_CLOTHESID.jpg
```

Example:

```text
000051_003_00013_0270_04_001.jpg
```

The fields encode identity, image index, video ID, frame ID, bounding-box ID, and clothes ID.

In validation/test galleries, folder `000000` contains unlabeled distractor images and is not treated as a valid identity.

---

## 2. Dataset Size

| Split | Images |
|---|---:|
| Train | 71,248 |
| Validation Query | 100 |
| Validation Gallery | 21,279 |
| Test Query | 10,176 |
| Test Gallery | 125,353 |
| **Total** | **228,156** |

Training identities:

```text
5,000
```

Distractor gallery images:

```text
Validation Gallery = 20,000
Test Gallery       = 20,275
```

Therefore, labeled gallery sizes are:

```text
Validation = 1,279
Test       = 105,078
```

---

## 3. Manifest and Metadata Integrity

All images were successfully represented in the final manifest:

```text
Parsed   = 228,156
Unparsed = 0
```

The audit found:

```text
31 filenames with a double underscore after the PID
30 gallery files with folder-PID / filename-PID mismatch
```

The mismatch is concentrated in a known test-set anomaly involving filename PID `005859` stored under folder `005895`.

The raw data was not modified. The manifest preserves both the filename PID and folder PID and records anomaly flags.

No such filename anomaly affects the training split.

---

## 4. Training Identity Distribution

Training set:

```text
Images = 71,248
IDs    = 5,000
```

Images per identity:

```text
Min    = 5
Mean   = 14.25
Median = 9
Std    = 14.24
Max    = 140
```

Percentiles:

```text
1%  = 5
5%  = 5
25% = 6
50% = 9
75% = 16
95% = 44
99% = 75
```

The distribution is noticeably imbalanced, so raw image-uniform sampling would give identities with many images more influence.

---

## 5. Clothes Diversity

All training images have known clothes labels:

```text
Known clothes images   = 71,248 (100%)
Unknown clothes images = 0
```

Known clothes IDs per training identity:

```text
Mean   = 3.21
Median = 2
Min    = 1
Max    = 36
```

Identity-level summary:

```text
1 clothes ID   = 1,820 identities
>=2 clothes IDs = 3,180 identities
```

Therefore:

```text
63.60% of training identities have multiple known clothes IDs
```

This confirms substantial clothes-change diversity in the training set.

---

## 6. Video Structure

Every training identity is associated with exactly one `video_id`:

```text
5,000 / 5,000 IDs appear in one video
```

Even identities with multiple clothes labels remain associated with a single video ID.

Therefore, the training metadata provides strong clothes-state diversity but does not provide cross-video positives for an identity according to the supplied `video_id`.

---

## 7. Temporal Redundancy

Images per `(PID, clothes ID)`:

```text
Mean   = 4.44
Median = 3
25%    = 1
75%    = 6
95%    = 13
99%    = 26
Max    = 90
```

Frame-gap statistics within `(PID, clothes ID)` groups:

```text
Mean   = 70.37
Median = 12
25%    = 4
75%    = 44
95%    = 348
99%    = 996
```

Temporal proximity:

```text
Gap <= 1 frame   : 11.25%
Gap <= 5 frames  : 33.42%
Gap <= 10 frames : 47.12%
Gap <= 20 frames : 61.29%
Gap <= 60 frames : 79.75%
```

This indicates meaningful temporal redundancy, although images are not uniformly consecutive.

The audit also found:

```text
16 same-PID/video/frame groups
32 images involved
```

These are separate bounding boxes rather than exact duplicate files.

A small number of same-frame groups have inconsistent clothes labels, so clothes metadata should be treated as useful but not perfectly noise-free.

---

## 8. Split and Evaluation Integrity

Identity overlap:

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

### Validation

```text
Query images       = 100
Query IDs          = 56
Labeled gallery    = 1,279
Gallery IDs        = 56
Missing query IDs  = 0
Zero-positive queries = 0
```

### Test

```text
Query images       = 10,176
Query IDs          = 5,805
Labeled gallery    = 105,078
Gallery IDs        = 5,806
Missing query IDs  = 0
Zero-positive queries = 0
Gallery-only IDs   = 1 (PID 10900)
```

There is no exact query/gallery filename overlap and no identical `(PID, video, frame, bbox)` overlap.

---

## 9. Query–Gallery Temporal Structure

The evaluation protocol is dominated by same-video positives.

Validation:

```text
Queries with zero different-video positives = 100 / 100
Median nearest same-video positive gap       = 10 frames
Nearest positive <= 10 frames                = 51.00%
Nearest positive <= 60 frames                = 82.00%
```

Test:

```text
Queries with zero different-video positives = 10,172 / 10,176
Median nearest same-video positive gap       = 10 frames
Nearest positive <= 10 frames                = 50.66%
Nearest positive <= 60 frames                = 82.64%
Nearest positive <= 100 frames               = 88.26%
```

Same-frame query/gallery positive cases are rare:

```text
Validation = 1 query
Test       = 2 queries
```

The official protocol remains usable, but this temporal structure should be considered when interpreting evaluation results.

---

## 10. Image Integrity and Geometry

All images were successfully decoded:

```text
Readable images  = 228,156
Corrupted images = 0
RGB images       = 228,156
```

### Training Geometry

```text
Width:
Min    = 25
Mean   = 255.16
Median = 236
Max    = 1189

Height:
Min    = 57
Mean   = 468.64
Median = 478
Max    = 1058

Aspect ratio (W/H):
Min    = 0.142
Mean   = 0.544
Median = 0.507
Max    = 3.414
```

Training diagnostic flags:

```text
Tiny         = 197   (0.28%)
Very narrow  = 131   (0.18%)
Horizontal   = 2,176 (3.05%)
Very wide    = 18    (0.03%)
Extreme-any  = 2,495 (3.50%)
```

These flags are diagnostic only and are not removal rules.

---

## 11. Visual Audit

Visual inspection showed broad variation in:

- full-body and partial-body crops
- upright, sitting, lying, and unusual poses
- indoor and outdoor scenes
- person scale
- lighting
- background complexity
- clothing state

The multi-clothes contact sheet visually confirms meaningful clothing changes for the same identity.

The narrowest crops are mostly valid person crops rather than obvious annotation failures.

The tiny crops are low-resolution and difficult, but the person remains visually present in the inspected examples.

The widest crops mainly correspond to valid horizontal or reclining body poses rather than failed detections.

Therefore, no automatic removal of extreme-aspect-ratio or tiny samples is recommended at this stage.

---

## 12. Main Strengths

- Large identity count
- Strong clothes-change diversity
- 100% clothes-label coverage in training
- Clean train/validation/test identity separation
- Large gallery with distractors
- No corrupted images
- Broad pose, scale, scene, and appearance variation
- Rich metadata for temporal and clothing-aware sampling

---

## 13. Main Risks

- Strong training identity-size imbalance
- Meaningful temporal redundancy
- Each training identity is associated with only one video ID
- Evaluation is overwhelmingly same-video
- Many query positives are temporally close
- Small number of metadata anomalies and clothes-label inconsistencies
- Some tiny and extreme-aspect-ratio samples

---

## 14. DG Fine-Tuning Decision

**Current role:** Strong source candidate for multi-source Domain-Generalizable Person Re-Identification.

Recommended later training strategy should consider:

```text
identity-balanced sampling
+ clothes-aware sampling
+ temporal-diversity control
```

Raw uniform image sampling is not preferred because identities have substantially different image counts and temporally adjacent samples may be repeatedly selected.

No image deletion is recommended at this stage.

The official raw data should remain unchanged; known metadata anomalies should be handled through the manifest.

---

# Final Audit Summary

| Property | Result |
|---|---:|
| Total Images | 228,156 |
| Train Images | 71,248 |
| Train IDs | 5,000 |
| Multi-Clothes Train IDs | 3,180 (63.60%) |
| Train Clothes Coverage | 100% |
| Train Videos per ID | 1 |
| Median Train Images / ID | 9 |
| Corrupted Images | 0 |
| Train Extreme-Flag Rate | 3.50% |
| Split PID Leakage | None |
| Test Query IDs | 5,805 |
| Test Gallery IDs | 5,806 |
| Test Queries Without Positive | 0 |
| Test Distractors | 20,275 |
| Filename Anomalies | 31 |
| Folder-PID Mismatches | 30 |
| Median Test Nearest Positive Gap | 10 frames |
| DG Role | Strong Source Candidate |

## Final Decision

**LaST is retained as a strong source candidate for multi-source Domain-Generalizable Person Re-Identification.**

Its main value is the combination of a large identity set, substantial clothes-change diversity, diverse visual conditions, and rich metadata. The main controls required for later fine-tuning are identity balance and temporal/clothing-aware sampling. No automatic cleaning or sample removal is currently justified.
