# DeepChange — Dataset Audit Report

## 1. Overview

**Dataset:** DeepChange  
**Task:** Long-Term Person Re-Identification  
**Data type:** Image-based ReID benchmark derived from person tracklets across outdoor surveillance cameras.

Each image filename follows:

```text
pXXXX_cXX_YYYYMMDD_HHMMSS_FRAME_bbox.jpg
```

Example:

```text
p0003_c01_20201015_172644_000406_bbox.jpg
```

The metadata fields used in this audit are:

- `pid`: person identity
- `camera_id`: surveillance camera
- `date`
- `time`
- `frame_id`
- `tracklet_id`: supplied by the accompanying TXT annotation file

The TXT annotation has one row per image and the second field is the tracklet identifier.

---

## 2. Dataset Size

| Split | Images | IDs |
|---|---:|---:|
| Train | 75,083 | 450 |
| Validation Query | 4,976 | 150 |
| Validation Gallery | 17,865 | 150 |
| Test Query | 17,527 | 521 |
| Test Gallery | 62,956 | 521 |
| **Total** | **178,407** | **1,121 split-specific identities** |

The number of annotation rows exactly matches the number of physical images in every archive.

---

## 3. Manifest Integrity

The final manifest contains:

```text
Parsed  = 178,407
Invalid = 0
```

All filenames conform to the expected metadata format.

The dataset contains camera IDs:

```text
1 through 17
```

Global temporal coverage in the supplied data:

```text
Earliest = 2020-08-13 07:00:17
Latest   = 2021-07-01 06:58:28
```

No invalid timestamps were found.

---

## 4. Split Integrity

Identity separation is clean:

```text
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0
```

Validation:

```text
Query IDs   = 150
Gallery IDs = 150
Missing query IDs = 0
Gallery-only IDs  = 0
```

Test:

```text
Query IDs   = 521
Gallery IDs = 521
Missing query IDs = 0
Gallery-only IDs  = 0
```

Therefore, there is no PID-level train/validation/test leakage according to the provided labels.

---

## 5. Camera Coverage

The dataset contains 17 cameras overall.

Training uses:

```text
16 cameras
```

Camera `2` is absent from the training split.

It appears only sparsely in evaluation:

```text
Validation:
5 query images
27 gallery images
1 PID

Test:
15 query images
62 gallery images
3 PIDs
```

Therefore, camera 2 is technically unseen during training, but it forms only a very small evaluation subset.

---

## 6. Training Identity Distribution

Training:

```text
75,083 images
450 identities
3,304 tracklets
```

Images per identity:

```text
Min    = 13
Mean   = 166.85
Median = 94
Std    = 173.95
Max    = 1,446
```

This is strongly imbalanced.

A small number of identities contain many more images than others, so raw image-uniform sampling would over-represent them.

---

## 7. Tracklet Diversity

Tracklets per PID:

```text
Min    = 2
Mean   = 7.34
Median = 5
Max    = 63
```

All training identities have at least two tracklets:

```text
450 / 450 = 100%
```

Cameras per PID:

```text
Min    = 2
Mean   = 3.26
Median = 3
Max    = 9
```

All training identities are multi-camera:

```text
450 / 450 = 100%
```

This provides strong multi-observation and cross-camera diversity.

---

## 8. Date and Long-Term Diversity

Unique dates per PID:

```text
Min    = 1
Mean   = 2.71
Median = 2
Max    = 11
```

Identities with at least two dates:

```text
278 / 450 = 61.78%
```

Temporal span per PID:

```text
Mean   = 69.44 days
Median = 22.47 days
Max    = 321.99 days
```

Identity-level temporal coverage:

```text
>=   1 day   : 61.11%
>=   7 days  : 58.67%
>=  30 days  : 48.89%
>=  90 days  : 24.44%
>= 180 days  : 15.78%
>= 270 days  :  7.56%
```

Therefore, long-term diversity is substantial but not equally strong for every identity.

---

## 9. Session and Temporal Structure

Each tracklet is internally consistent:

```text
Tracklets with >1 PID       = 0
Tracklets with >1 camera    = 0
Tracklets with >1 date      = 0
Tracklets with >1 timestamp = 0
```

Only one `(PID, timestamp)` group contains more than one tracklet.

Thus, tracklets correspond almost one-to-one with separate observation sessions in the supplied metadata.

Consecutive session-gap distribution:

```text
< 1 hour           : 48.93%
1 hour - <1 day    : 27.30%
1 - <7 days        :  4.21%
7 - <30 days       :  7.43%
30 - <90 days      :  9.25%
90 - <180 days     :  1.65%
>=180 days         :  1.23%
```

About 76% of consecutive session gaps are below one day.

Therefore, the dataset contains both short-term and long-term observations.

---

## 10. Tracklet Pair Diversity

Across all tracklet pairs of the same identity:

```text
Same date + same camera      =  8.70%
Same date + different camera = 20.05%
Different date + same camera = 22.46%
Different date + diff camera = 48.78%
```

At the identity level:

```text
93.78% have at least one same-date / different-camera pair
50.00% have at least one different-date / same-camera pair
61.78% have at least one different-date / different-camera pair
```

The same-camera / different-date subset is especially useful because camera variation is controlled while time and appearance can change.

For these pairs:

```text
Median gap = 41.98 days
Mean gap   = 53.16 days
```

Identity coverage:

```text
>=  7 days : 211 PIDs
>= 30 days : 183 PIDs
>= 90 days :  81 PIDs
>=180 days :  42 PIDs
```

---

## 11. Official Evaluation Structure

For evaluation, same-PID images from the same tracklet are not treated as valid matches; valid positives are cross-tracklet matches of the same identity.

Validation:

```text
Total queries             = 4,976
Queries with valid match  = 4,966
Queries with no valid match = 10
```

Test:

```text
Total queries             = 17,527
Queries with valid match  = 17,462
Queries with no valid match = 65
```

Among test queries with valid matches:

```text
Has cross-camera valid positive     = 99.89%
Has different-date valid positive   = 80.02%
Has >=30-day valid positive         = 70.73%
Has >=90-day valid positive         = 34.92%
Has >=180-day valid positive        = 20.85%
Has same-camera >=30-day positive   = 46.18%
```

Median number of valid positives per test query:

```text
167
```

The nearest valid cross-tracklet positive is often still temporally close:

```text
Median nearest valid gap ≈ 0.0089 days
                          ≈ 12.9 minutes
```

So the benchmark contains both short-term cross-tracklet matches and substantial long-term matches.

---

## 12. Image Integrity

All images were successfully decoded:

```text
Readable images  = 178,407
Missing images   = 0
Corrupted images = 0
RGB images       = 178,407
```

Every image is stored at:

```text
64 × 128
```

with aspect ratio:

```text
0.5
```

Therefore, geometric crop analysis based on width/height is not informative in this released version because all crops were already normalized to the same resolution.

---

## 13. Visual Audit

Visual inspection shows:

- outdoor surveillance scenes
- low-resolution person crops
- viewpoint changes
- back/side/front poses
- walking, cycling, standing and other activity variation
- blur and occlusion
- masks and carried objects
- substantial clothing and appearance change across dates
- clear long-term examples spanning months
- both same-camera and different-camera long-term observations

The timeline sheets visually support the temporal statistics: some identities change clothing and appearance substantially across months.

No obvious systematic crop failure was observed in the sampled sheets.

---

## 14. Exact Duplicate Audit

A full-image SHA-256 and decoded-pixel SHA-256 audit found the same result:

```text
Exact duplicate groups = 67
Images involved        = 134
```

All duplicate groups contain two images.

This affects approximately:

```text
134 / 178,407 ≈ 0.075% of all images
```

The important finding is that most are not harmless same-label repetitions:

```text
Cross-PID exact duplicate groups      = 62
Cross-tracklet exact duplicate groups = 66
Cross-split exact duplicate groups    = 25
Train ↔ Evaluation groups             = 12
Validation Query ↔ Gallery groups     = 1
Test Query ↔ Gallery groups           = 8
```

Cross-PID exact duplicates affect at most 124 image entries, approximately:

```text
0.070% of the dataset
```

Examples show identical pixels assigned to different PIDs, sometimes with matching camera, timestamp, and frame metadata.

Examples also include exact duplicates crossing training and evaluation splits.

This is a real annotation/data-consistency issue, not a perceptual-hash false positive, because both the compressed file bytes and decoded RGB pixels are exactly identical.

---

## 15. Near-Duplicate Audit

64-bit dHash found:

```text
409 same-dHash groups
```

Most are expected visually similar frames from the same tracklet.

After excluding groups confined to one PID and one tracklet:

```text
67 groups cross PID or tracklet
63 cross PID
67 cross tracklet
```

These results almost entirely overlap the exact-duplicate findings.

Therefore, the dominant suspicious issue is the small set of exact duplicates rather than a large hidden near-duplicate population.

---

## 16. Duplicate Handling Recommendation

The original benchmark files should remain unchanged for reproducibility.

For custom multi-source training:

- exact cross-PID duplicate images should be flagged in the manifest
- contradictory duplicate supervision should not be kept blindly
- when a training image exactly duplicates a validation/test image under another PID, excluding the training copy is the safest option for contamination-sensitive experiments
- same-PID redundancy can be handled through tracklet-aware sampling rather than aggressive deletion

For official benchmark reporting:

- preserve the official evaluation protocol for comparability
- report the duplicate issue transparently
- an optional cleaned supplementary evaluation can be added, but should not replace the official benchmark score

---

## 17. Main Strengths

- 450 training identities with strong multi-tracklet coverage
- 100% of training identities appear in multiple cameras
- substantial long-term temporal coverage
- same-camera long-term observations
- realistic surveillance appearance variation
- clean PID-level split separation
- no corrupted or missing images
- rich camera, tracklet, frame and timestamp metadata
- evaluation contains abundant cross-tracklet, cross-camera and long-term positives

---

## 18. Main Risks

- strong identity-size imbalance
- substantial short-term redundancy inside and across tracklets
- low native resolution (`64×128`)
- not every identity has strong long-term coverage
- camera 2 is unseen in training but only sparsely represented in evaluation
- 67 exact duplicate pairs exist
- 62 exact duplicate pairs cross PID labels
- 12 exact duplicate pairs cross training and evaluation
- exact query/gallery duplicate pairs exist in both validation and test

---

## 19. DG Fine-Tuning Decision

**Current role:** Strong source candidate for multi-source Domain-Generalizable Person Re-Identification.

Recommended later training strategy should consider:

```text
identity-balanced sampling
+ tracklet-aware sampling
+ camera diversity
+ temporal diversity when available
```

A strong batch design should avoid repeatedly sampling adjacent frames from the same tracklet when more diverse observations of the same identity are available.

For custom training, exact cross-PID duplicate conflicts should be filtered or explicitly masked through the manifest.

---

# Final Audit Summary

| Property | Result |
|---|---:|
| Total Images | 178,407 |
| Train Images | 75,083 |
| Train IDs | 450 |
| Total Cameras | 17 |
| Train Cameras | 16 |
| Train Tracklets | 3,304 |
| Multi-Tracklet Train IDs | 100% |
| Multi-Camera Train IDs | 100% |
| Multi-Date Train IDs | 61.78% |
| Train IDs with >=30d Span | 48.89% |
| Train IDs with >=90d Span | 24.44% |
| Same-Camera >=30d Train IDs | 183 |
| Corrupted Images | 0 |
| Missing Images | 0 |
| Native Resolution | 64×128 |
| Test Queries | 17,527 |
| Test Queries with Valid Match | 17,462 |
| Test Valid Cross-Camera Coverage | 99.89% |
| Test Valid >=30d Coverage | 70.73% |
| Test Valid >=90d Coverage | 34.92% |
| Test Valid >=180d Coverage | 20.85% |
| Exact Duplicate Groups | 67 |
| Cross-PID Exact Duplicate Groups | 62 |
| Train↔Eval Exact Duplicate Groups | 12 |
| DG Role | Strong Source Candidate |

## Final Decision

DeepChange is retained as a strong source candidate for multi-source Domain-Generalizable Person Re-Identification.

Its strongest contribution is long-term temporal and appearance variation under realistic outdoor surveillance conditions. The main controls needed for later fine-tuning are identity balance, tracklet diversity, temporal diversity, and explicit handling of the small but real exact-duplicate label conflicts.

The raw official dataset should remain untouched; all corrections or exclusions should be implemented through reproducible manifest flags.
