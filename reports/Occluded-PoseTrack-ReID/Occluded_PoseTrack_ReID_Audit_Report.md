# Occluded-PoseTrack-ReID — Dataset Audit Report

## 1. Overview

**Dataset:** Occluded-PoseTrack-ReID  
**Task:** Person Re-Identification (ReID)  
**Type:** Image-based ReID dataset derived from PoseTrack data, with strong pose, crowding, partial-body, and occlusion-related characteristics.

Audited package structure:

```text
occluded_posetrack_reid/
├── images/
│   ├── anns/
│   │   ├── train_reid_crops_anns.json
│   │   └── val_reid_crops_anns.json
│   ├── train/
│   │   └── <video_id>/*.jpg
│   └── val/
│       └── <video_id>/*.jpg
├── train_dataset_sampling.json
└── val_dataset_sampling.json
```

The downloaded archive size was:

```text
435.671 MiB
```

The extracted package contains:

```text
31,310 JPG images
4 JSON metadata files
31,314 physical files total
```

The raw package was preserved unchanged. All audit outputs were produced through manifests, reports, and diagnostic files.

---

## 2. Dataset Size and Official Sampling Roles

The final selected ReID crops are:

| Split | Images | Person IDs | Videos |
|---|---:|---:|---:|
| Train | 17,898 | 1,000 | 474 |
| Query | 2,581 | 1,379 | 163 |
| Gallery | 10,831 | 1,411 | 170 |
| **Total** | **31,310** | — | — |

The physical package has only two source partitions:

```text
images/train/
images/val/
```

However, `images/val/` is not a gallery-only folder. Query/gallery roles are defined by `val_dataset_sampling.json`.

Sampling metadata:

### Training annotations

```text
Total candidate annotation records = 118,611
train                            =  17,898
none                             = 100,713
```

Every record assigned `train` has a physical ReID crop, and every physical training crop is referenced exactly once.

### Validation/evaluation annotations

```text
Total candidate annotation records = 58,553
query                              =  2,581
gallery                            = 10,831
none                               = 45,141
```

The selected validation crops satisfy:

```text
2,581 query + 10,831 gallery = 13,412 physical val images
```

There are no unreferenced selected crops and no selected metadata records missing a physical image.

---

## 3. Filename and Metadata Schema

Confirmed filename format:

```text
PersonID_VideoID_PoseTrackImageID.jpg
```

Example:

```text
1659_000003_10000030011.jpg
```

Parsed as:

```text
1659         → Person ID
000003       → Video ID
10000030011  → PoseTrack Image ID
```

Important interpretation:

- `person_id` is the raw identity label encoded in the supplied filename.
- `video_id` identifies the PoseTrack video/sequence.
- `video_id` may be used as a logical ReID `camid` by the dataset implementation, but it is **not a physical surveillance camera ID**.
- `PoseTrack Image ID` is treated as an opaque identifier in this audit.
- No undocumented internal decomposition of the PoseTrack Image ID was assumed.

All 31,310 selected filenames conform to the expected schema.

The final audit manifest uses `person_id` rather than `pid` to avoid confusion with implementation-specific re-indexed IDs.

---

## 4. Annotation and Manifest Integrity

Final corrected manifest:

```text
occluded_posetrack_reid_manifest_v2.csv
```

Manifest rows:

```text
31,310
```

Integrity checks:

```text
Invalid filenames          = 0
Missing physical images    = 0
Folder/video mismatches    = 0
Unexpected sampling roles  = 0

Unique annotation IDs      = 31,310
Unique filenames           = 31,310
Duplicate annotation IDs   = 0
Duplicate filenames        = 0

Null crop widths           = 0
Null crop heights          = 0
Missing negative_kps       = 0
```

The folder containing each crop matches its `video_id` for every selected image.

The width and height stored in the annotation metadata also match the decoded physical image dimensions for every image.

---

## 5. Training Identity Distribution

Training set:

```text
Images  = 17,898
Persons = 1,000
Videos  = 474
```

Images per person:

```text
Min    = 4
Mean   = 17.90
Median = 20
Max    = 20
```

Exact high-level structure:

```text
751 / 1,000 training persons have exactly 20 images
249 / 1,000 have between 4 and 19 images
```

The identity-size imbalance is therefore moderate and strongly capped at 20 images per person.

Videos per training person:

```text
1 video  → 895 persons
2 videos →  79 persons
3 videos →  17 persons
4 videos →   6 persons
5 videos →   3 persons
```

Therefore:

```text
105 / 1,000 training persons (10.5%)
appear in more than one video.
```

Most training identities are single-video identities, but a smaller subset provides cross-video diversity.

---

## 6. Evaluation Identity Structure

Evaluation set:

```text
Query + Gallery images = 13,412
Evaluation person IDs   = 1,411
```

Images per evaluation person:

```text
Min    = 4
Mean   = 9.51
Median = 10
Max    = 10
```

Distribution:

```text
4 images  →   32 persons
5 images  →   23 persons
6 images  →   29 persons
7 images  →   46 persons
8 images  →   58 persons
9 images  →   21 persons
10 images → 1,202 persons
```

Videos per evaluation person:

```text
1 video  → 1,232 persons
2 videos →   112 persons
3 videos →    52 persons
4 videos →    11 persons
5 videos →     3 persons
6 videos →     1 person
```

Thus:

```text
179 / 1,411 evaluation persons
appear in more than one video.
```

---

## 7. Query / Gallery Protocol Integrity

Query set:

```text
Query images = 2,581
Query IDs    = 1,379
```

Gallery set:

```text
Gallery images = 10,831
Gallery IDs    = 1,411
```

Query-count distribution per person:

```text
0 query images →   32 persons
1 query image  →  177 persons
2 query images → 1,202 persons
```

All 32 gallery-only identities contain exactly four gallery images:

```text
32 identities × 4 images = 128 gallery-only images
```

These identities are not treated as distractors; they are labeled gallery identities with no selected query image.

Protocol validity:

```text
Total queries                   = 2,581
Queries with same-person gallery positive = 2,581
Queries without positive        = 0
```

Therefore, every official query has at least one valid gallery positive under the audited default evaluation setting.

Cross-video structure:

```text
Queries with cross-video positive     = 352
Queries with same-video-only positive = 2,229
```

This should be treated as a dataset characteristic rather than a protocol error. The evaluation set is predominantly same-video at the query-positive level.

---

## 8. Train / Evaluation Identity Separation

A raw numeric overlap was detected:

```text
Train person IDs ∩ Eval person IDs = {1601}
```

Observed samples:

```text
Train:
person_id = 1601
video_id  = 000048
20 images

Evaluation:
person_id = 1601
video_id  = 022842
2 query + 8 gallery images
```

Visual inspection shows clearly different visual subjects/scenes between the training and evaluation occurrences:

- the training samples show a person in an orange upper garment in video `000048`
- the evaluation samples show a cyclist in video `022842`

The visual evidence strongly indicates that the shared numeric label does **not** represent the same real identity across the two source partitions.

Therefore:

```text
Confirmed true identity leakage = none found
Raw numeric person_id overlap   = 1
```

The raw `person_id` namespace should not be assumed to be globally stable across train and validation partitions without additional source-level proof.

For all downstream work, training and evaluation identity namespaces should remain split-aware.

---

## 9. Image Integrity and Geometry

All selected images were successfully decoded:

```text
Manifest images       = 31,310
Readable images       = 31,310
Corrupted images      = 0
Dimension mismatches  = 0
JPEG images           = 31,310
RGB images            = 31,310
```

Overall geometry:

### Width

```text
Min    = 20
P01    = 33.09
P05    = 51
Median = 128
P95    = 128
P99    = 128
Max    = 128
```

### Height

```text
Min    = 21
P01    = 71
P05    = 115
Median = 384
P95    = 384
P99    = 384
Max    = 384
```

### Area

```text
Min    = 840
Median = 49,152
Max    = 49,152
```

### Aspect ratio (W/H)

```text
Min    = 0.0771
P01    = 0.2245
P05    = 0.3140
Median = 0.3333
P95    = 0.5962
P99    = 0.9352
Max    = 3.7619
```

The dominant crop geometry is:

```text
128 × 384
```

Horizontal images:

```text
Train   = 61
Query   = 32
Gallery = 163
Total   = 256
```

Total horizontal rate:

```text
256 / 31,310 ≈ 0.82%
```

Geometry extremes include:

- extremely narrow partial-body crops
- very small crops
- horizontal/reclining person crops
- crops containing only a small visible body region

These samples are valid physical images and are not corrupted.

No automatic geometry-based deletion is recommended.

---

## 10. Exact Duplicate Audit

SHA-256 hashing was computed for all 31,310 selected images.

Result:

```text
Exact duplicate groups = 1
Images in duplicate groups = 2
Extra duplicate copies = 1
```

The duplicate pair is:

```text
4624_016560_10165600065.jpg
4624_016560_10165600066.jpg
```

Both files are:

```text
split     = train
person_id = 4624
video_id  = 016560
```

Duplicate characteristics:

```text
Same-person groups      = 1
Cross-person groups     = 0
Same-video groups       = 1
Cross-video groups      = 0
Cross-split groups      = 0
Train↔Eval groups       = 0
Query↔Gallery groups    = 0
```

Visual inspection confirms that the two images are identical.

This is a small local redundancy issue, not a train/evaluation contamination issue.

The raw files should remain untouched. The pair may be flagged in the manifest if later custom training wishes to avoid redundant sampling.

---

## 11. Negative-Keypoint / Crowding Structure

The annotation package provides `negative_kps`, representing keypoint information associated with other people around the selected target crop.

Important limitation:

```text
negative_kps is NOT treated as the official target-person occlusion score.
```

It is used here only as a crowding / other-person interference signal.

All selected images have valid `negative_kps` metadata:

```text
Manifest images              = 31,310
Analyzed images              = 31,310
Missing negative_kps         = 0
Invalid non-empty shapes     = 0
```

Overall:

```text
No negative person      =  7,734 images (24.70%)
>=1 negative person     = 23,576 images (75.30%)
```

Negative-person count distribution:

```text
0  persons → 7,734 images
1  person  → 11,176 images
2  persons → 6,770 images
3  persons → 3,319 images
4  persons → 1,279 images
5  persons →   533 images
6  persons →   276 images
7  persons →    97 images
8  persons →    62 images
9  persons →    25 images
10 persons →    15 images
11 persons →     8 images
12 persons →     4 images
13 persons →     5 images
15 persons →     7 images
```

Visible negative-keypoint statistics:

```text
Min    = 0
Median = 4
Mean   = 5.71
P95    = 18
P99    = 29
Max    = 76
```

### Train

```text
Images                         = 17,898
No negative person             = 4,812 (26.89%)
With >=1 negative person       = 13,086 (73.11%)
Median negative people         = 1
Mean negative people           = 1.30
Median visible negative kps    = 4
Mean visible negative kps      = 5.12
```

### Query

```text
Images                         = 2,581
No negative person             = 220 (8.52%)
With >=1 negative person       = 2,361 (91.48%)
Median negative people         = 2
Mean negative people           = 2.10
Median visible negative kps    = 8
Mean visible negative kps      = 9.34
```

### Gallery

```text
Images                         = 10,831
No negative person             = 2,702 (24.95%)
With >=1 negative person       = 8,129 (75.05%)
Median negative people         = 1
Mean negative people           = 1.60
Median visible negative kps    = 4
Mean visible negative kps      = 5.81
```

The query subset contains substantially more negative-person context than train or gallery.

This supports the interpretation that query crops are more heavily affected by surrounding-person interference/crowding, while avoiding the unsupported claim that this measurement is itself the official occlusion level.

Maximum observed crowding signals:

```text
Maximum negative people in one crop = 15
Maximum visible negative keypoints  = 76
```

---

## 12. Visual Audit

Targeted contact sheets were inspected for:

- random train samples
- random query samples
- random gallery samples
- smallest crops
- narrowest crops
- widest/horizontal crops
- crops with no negative people
- most crowded crops
- the exact duplicate pair
- the raw `person_id=1601` train/eval overlap

Visual inspection showed broad variation in:

- full-body and partial-body crops
- very small person scale
- severe partial visibility
- unusual body orientations
- upright, bent, sitting, lying, athletic, and cycling poses
- indoor and outdoor scenes
- sports and crowd scenes
- background complexity
- blur and resolution
- strong surrounding-person interference

The smallest and narrowest examples include genuinely difficult low-information crops.

The widest examples include strong partial-body crops and horizontal/reclining poses.

The most crowded examples visually confirm that the high `negative_kps` counts correspond to real multi-person interference.

The exact duplicate pair is visually identical.

The numeric `person_id=1601` overlap appears visually inconsistent with a shared real-world identity across train and evaluation.

No automatic image deletion is justified from the visual audit alone.

---

## 13. Temporal Analysis Limitation

A temporal redundancy audit was **not** performed.

Reason:

- the package provides PoseTrack Image IDs
- the audit does not have an explicitly verified `frame_id` field for each selected crop
- no undocumented numeric decomposition of the PoseTrack Image ID was assumed

Therefore:

```text
Temporal frame-gap analysis = Not available from the verified package metadata
```

This is an intentional audit limitation rather than a missing processing step.

If verified source frame metadata becomes available later, temporal redundancy can be audited as an additional optional analysis.

---

## 14. Main Strengths

- 1,000 training identities and 17,898 selected training crops
- strong pose and body-orientation diversity
- substantial partial-body and difficult-scale variation
- real crowding and surrounding-person interference
- query set intentionally/structurally contains difficult crowded cases
- diverse indoor, outdoor, sports, and public-scene appearances
- clean physical image decoding: zero corrupted images
- perfect selected-metadata ↔ physical-image matching
- no cross-person exact duplicate conflicts
- no exact train↔evaluation duplicate contamination
- every official query has at least one gallery positive
- useful video metadata for logical sequence/domain-aware analysis
- a subset of identities provides cross-video observations

---

## 15. Main Risks and Caveats

- most identities appear in only one video
- the majority of queries have only same-video positives
- some crops are extremely small, narrow, horizontal, or partial-body
- image quality varies substantially
- one exact same-person/same-video duplicate pair exists in training
- raw numeric `person_id` values should not be assumed globally unique across train/eval partitions
- `video_id` is a logical sequence identifier, not a physical camera ID
- `negative_kps` is a crowding/interference signal, not a direct official target-person occlusion score
- explicit temporal/frame-gap redundancy cannot be measured from the currently verified metadata
- very hard crops may introduce training noise if sampled too aggressively

---

## 16. DG Fine-Tuning Decision

**Current role:** Strong source candidate for multi-source Domain-Generalizable Person Re-Identification.

The dataset's strongest contribution is not conventional multi-camera surveillance diversity. Its main value is:

```text
pose diversity
+ partial-body observations
+ crowding
+ occlusion-related interference
+ difficult person scale
+ unusual body orientation
```

This makes it complementary to cleaner conventional surveillance datasets.

Recommended later training strategy should consider:

```text
identity-balanced sampling
+ video diversity when available
+ avoid over-concentrating on extreme low-information crops
+ preserve crowding/pose diversity
```

For the 105 multi-video training identities, a video-diversity-aware sampler can prefer different videos when possible.

For single-video identities, sampling should still avoid unnecessary repeated highly similar observations when more diverse crops are available.

The single exact duplicate pair may be flagged through the manifest, but no raw-file deletion is required.

No automatic deletion of tiny, narrow, horizontal, partial-body, or crowded samples is recommended at this stage.

---

# Final Audit Summary

| Property | Result |
|---|---:|
| Total Images | 31,310 |
| Train Images | 17,898 |
| Train Person IDs | 1,000 |
| Train Videos | 474 |
| Query Images | 2,581 |
| Query Person IDs | 1,379 |
| Gallery Images | 10,831 |
| Gallery Person IDs | 1,411 |
| Evaluation Videos | 170 |
| Median Train Images / Person | 20 |
| Multi-Video Train Persons | 105 (10.5%) |
| Multi-Video Eval Persons | 179 |
| Queries Without Positive | 0 |
| Queries With Cross-Video Positive | 352 |
| Queries With Same-Video-Only Positive | 2,229 |
| Gallery-Only Person IDs | 32 |
| Gallery-Only Images | 128 |
| Corrupted Images | 0 |
| Dimension Mismatches | 0 |
| Horizontal Images | 256 (0.82%) |
| Exact Duplicate Groups | 1 |
| Cross-Person Exact Duplicate Groups | 0 |
| Train↔Eval Exact Duplicate Groups | 0 |
| Images With >=1 Negative Person | 23,576 (75.30%) |
| Query Images With >=1 Negative Person | 2,361 (91.48%) |
| Raw Train/Eval Numeric Person-ID Overlap | 1 (`1601`) |
| Confirmed True Identity Leakage | None found |
| Temporal Audit | N/A — no verified explicit frame metadata |
| DG Role | Strong Source Candidate |

## Final Decision

**Occluded-PoseTrack-ReID is retained as a strong source candidate for multi-source Domain-Generalizable Person Re-Identification.**

Its main value is the strong combination of pose variation, partial-body visibility, crowding, difficult person scale, and surrounding-person interference. These characteristics make it especially useful as a complementary source domain alongside cleaner surveillance datasets.

The dataset is physically and structurally clean: all selected images are readable, metadata mapping is complete, the official query/gallery selection is valid, and no exact train/evaluation image contamination was found.

The main controls required for later fine-tuning are identity-balanced sampling, video diversity when available, and awareness of extreme low-information crops. The raw dataset should remain unchanged; duplicate or geometry-related handling should be implemented only through reproducible manifest flags or sampler logic.
