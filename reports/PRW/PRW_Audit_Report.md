# PRW Dataset Audit Report

## 1. Audit Status

**Dataset:** PRW — Person Re-identification in the Wild  
**Audit status:** COMPLETE  
**Raw dataset:** Preserved unchanged  

### Final project decision

**PRW is excluded from the supervised DG source pool when Market1501 is used as an unseen target.**

The exclusion is not because PRW is unusable or low quality.

The reason is severe acquisition overlap between PRW and Market1501, including thousands of exact shared raw video frames.

---

## 2. Raw Dataset Structure

The audited PRW package contains:

- 11,816 full video frames
- 11,816 annotation MAT files
- 2,057 query crops
- 6 cameras
- official train/test frame split
- official train/test identity split

Main components:

- `frames/`
- `annotations/`
- `query_box/`
- `ID_train.mat`
- `ID_test.mat`
- `frame_train.mat`
- `frame_test.mat`
- `query_info.txt`
- `generate_query.m`
- `readme.txt`

Frame filenames encode:

`camera + segment + frame number`

Example:

`c1s1_000151`

means:

- Camera 1
- Segment 1
- Frame 151

Segments are collection/recording segments and should not be treated as physical camera IDs.

---

## 3. Official Protocol Audit

Frame split:

- Train frames: 5,704
- Test frames: 6,112
- Total: 11,816
- Train/test frame overlap: 0

Identity split in the distributed package:

- Train IDs: 483
- Test IDs: 450
- Total unique identified IDs: 933
- Train/test PID overlap: 0

The distributed package therefore contains one more identified PID than the commonly reported 932-person description.

The audit follows the actual distributed package for reproducibility rather than forcing published summary counts.

All 2,057 query images:

- map to official test identities
- map to official test frames
- have exact coordinate matches in the annotation files
- cover all 450 official test identities

---

## 4. Annotation Audit

Correct annotation parsing required three MAT keys:

- `box_new`: 11,792 files
- `anno_file`: 23 files
- `anno_previous`: 1 file

No annotation MAT file remained unsupported.

Total annotation boxes:

- Total boxes: 43,110
- Train-ID boxes: 17,690
- Test-ID boxes: 16,344
- Unknown PID `-2`: 9,076

The `-2` boxes are unknown identities and are not used as supervised ReID classes.

Important split behavior:

Some training identities appear inside official test frames.

Therefore the supervised source pool follows BOTH:

- official train frames
- official train identities

Train-ID boxes located in test frames are not included in source training.

---

## 5. Initial Supervised Source Pool

Using:

`train frame AND train identity`

produced:

- 14,907 supervised source boxes
- 483 identities
- 6 cameras

Identity distribution is strongly imbalanced:

- Mean images / PID: about 30.9
- Median: 23
- Minimum: 1
- Maximum before cleaning: 465

This supports using an identity-balanced sampler during training.

---

## 6. Bounding Box Geometry

Frame resolutions:

- 8,918 frames: 1920 × 1080
- 2,898 frames: 720 × 576

356 boxes in the complete annotation set extend slightly outside image boundaries.

Within the supervised source subset:

- 122 source boxes were initially out-of-bounds

These are mostly small boundary overflows rather than invalid pedestrians.

Policy:

- raw annotation unchanged
- do not exclude only because the box crosses the boundary
- clip the box to image limits when creating derived crops

After annotation-conflict cleaning, 121 retained training crops required clipping.

---

## 7. Same-PID / Same-Frame Conflict Audit

Ten suspicious same-PID/same-frame groups were found.

Classification:

- 1 exact annotation duplicate
- 1 overlapping duplicate of the same person
- 8 spatially distinct conflicts

Conservative cleaning policy:

- exact duplicate: keep one row, exclude one redundant row
- overlapping duplicate: keep one representative
- distinct conflicts: exclude both conflicting annotations
- no manual relabeling
- raw annotations remain unchanged

Hard exclusions:

- 16 rows from distinct conflicts
- 1 extra exact duplicate
- 1 extra overlapping duplicate

Total hard exclusions:

**18 rows**

---

## 8. Clean Derived Source Set

After audit exclusions:

- Clean source crops: 14,889
- Identities: 483
- Cameras: 6
- Boundary-clipped crops: 121

Crop width:

- Minimum: 23 px
- Median: 66 px
- Mean: 84.28 px
- Maximum: 626 px

Crop height:

- Minimum: 64 px
- Median: 176 px
- Mean: 224.25 px
- Maximum: 832 px

Rare identities remain:

- 2 identities have only 1 crop
- 18 identities have fewer than 4 crops

These identities are preserved.

Their presence is handled by sampling policy rather than raw-data deletion.

---

## 9. Exact Duplicate Audit

Exact decoded-pixel duplicate audit on all 14,889 derived source crops found:

- Duplicate groups: 0
- Images in duplicate groups: 0
- Cross-PID duplicate groups: 0
- Same-PID duplicate groups: 0
- Cross-camera duplicate groups: 0
- Cross-frame duplicate groups: 0

Derived JPEG byte-level duplicates:

- Duplicate groups: 0

Therefore no further duplicate exclusions are required.

---

## 10. Temporal Structure

Within the supervised source subset, same-PID / same-camera / same-segment adjacent observations show repeated temporal sampling.

Typical frame-number gaps are multiples of 25.

Observed adjacent-pair statistics included:

- gap <= 25: 38.70%
- gap <= 50: 53.95%
- gap <= 100: 69.73%
- gap <= 250: 84.07%
- gap <= 500: 89.30%
- gap <= 1000: 91.42%

This indicates substantial temporal redundancy.

If PRW is used in another experiment, temporal-diversity sampling is recommended.

---

## 11. Visual Audit

Visual inspection confirmed:

- outdoor surveillance environment
- six camera views
- substantial scale variation
- viewpoint variation
- illumination/background variation
- very small and distant pedestrians
- larger close-range pedestrians
- reasonable same-identity cross-camera appearance consistency

The dataset is suitable for person ReID by itself.

However, the scenes come from a limited acquisition environment rather than many unrelated physical domains.

---

# 12. Critical PRW ↔ Market1501 Leakage Audit

PRW and Market1501 encode compatible raw acquisition information:

`camera + segment + frame number`

This allowed direct comparison of their underlying video frames.

PRW clean source:

- Source crops: 14,889
- Unique source frames: 5,663
- Source identities: 483

Market1501 target consists of:

- Query
- Gallery (`bounding_box_test`)

## Exact raw-frame overlap

PRW source frames that are EXACTLY present in the Market1501 target:

**3,886 / 5,663 = 68.62%**

PRW source training crops located on these shared target frames:

**10,499 crops**

PRW source identities affected:

**475 / 483 identities**

This is severe source-target acquisition leakage.

---

## 13. Temporal Proximity to Market1501

Every PRW source frame has a Market1501 target frame in the same camera + segment acquisition space.

Nearest Market target-frame distances:

| Maximum frame gap | PRW source frames | Percentage |
|---|---:|---:|
| 0 | 3886 | 68.62% |
| 25 | 4271 | 75.42% |
| 50 | 4513 | 79.69% |
| 100 | 4888 | 86.31% |
| 250 | 5307 | 93.71% |
| 500 | 5537 | 97.78% |
| 1000 | 5638 | 99.56% |

This means simply deleting the exact shared frames would NOT create a clean DG source.

Large portions of the remaining PRW source would still come from the same cameras, same recording segments, and temporally nearby video content as Market1501.

---

# 14. DG Decision

## Dataset quality

PRW itself is a valid and useful ReID dataset.

After cleaning:

- protocol is reproducible
- source labels are usable
- crop generation is deterministic
- annotation conflicts are flagged
- no exact crop duplicates remain
- visual quality is suitable

## Role in the current DG experiment

### EXCLUDED AS SOURCE

Reason:

**Market1501 is one of the project's unseen evaluation targets, while PRW shares the same acquisition/video origin with Market1501.**

Using PRW during supervised training would violate the intended source-to-unseen-target separation.

The problem is not merely similar scenery.

The audit found:

- same cameras
- same acquisition segments
- temporally adjacent frames
- thousands of exact identical underlying raw frames

Therefore a high Market1501 result after PRW training could not be interpreted as clean domain generalization.

---

## 15. Final Status

| Item | Decision |
|---|---|
| Raw PRW preserved | Yes |
| Protocol understood | Yes |
| Annotation audit complete | Yes |
| Visual audit complete | Yes |
| Duplicate audit complete | Yes |
| Source crop manifest built | Yes |
| Audit flags built | Yes |
| Suitable ReID dataset | Yes |
| Suitable source if Market1501 is target | **No** |
| Final project role | **Excluded source / conflict with Market1501 target** |

---

# Final Conclusion

PRW passed its internal dataset audit after a small number of annotation conflicts were conservatively excluded.

A clean supervised subset of **14,889 crops from 483 identities** was produced.

However, a cross-dataset acquisition audit revealed severe overlap with Market1501, including **3,886 exact shared raw frames** and temporal/acquisition proximity across the complete PRW source recording space.

For the current domain-generalization experiment, where Market1501 must remain unseen, **PRW must not be used as a supervised training source**.

This is a dataset-role exclusion, not a claim that PRW is a poor-quality dataset.
