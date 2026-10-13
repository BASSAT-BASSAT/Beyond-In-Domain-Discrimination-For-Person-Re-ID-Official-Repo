# RPIfield Dataset Audit Report

## 1. Dataset

**Name:** RPIfield  
**Paper:** *RPIfield: A New Dataset for Temporally Evaluating Person Re-Identification*  
**Audit status:** Completed  
**Project context:** Candidate dataset for multi-source supervised training and Domain-Generalizable Person Re-Identification experiments.

---

## 2. Package Structure

```text
RPIfield_v0/
├── Data/
│   ├── Cam_1/
│   ├── Cam_2/
│   ├── ...
│   └── Cam_12/
│       └── <appearance_label>/
│           └── Cam_<camera>f_<frame_id>.png
│
├── Data_Info/
│   ├── Cam_1.mat
│   ├── ...
│   └── Cam_12.mat
│
├── load_data.m
└── Readme.txt
```

The package contains synchronized multi-camera person crops together with identity, timestamp/frame, and reappearance metadata.

---

## 3. Official Semantics

The dataset contains:

- **12 outdoor camera views**
- **112 known actors**
- **3,996 distractor identities**
- **4,108 total base person IDs**
- **6,577 appearance sequences**
- **601,581 PNG person crops**

Identity labels:

```text
1..112
    known actors

>= 10000
    distractors
```

Appearance labels may contain a suffix:

```text
2
2_1
2_2
```

where the suffix records a later appearance/reappearance label within a camera.

Image naming:

```text
Cam_10f_192406.png
```

means:

```text
camera = 10
frame_id = 192406
```

The frame/timestamp coordinate is synchronized across cameras according to the package documentation.

---

## 4. Manifest Verification

A manifest was built from the 12 `.mat` metadata files and reconciled with the filesystem.

Results:

- Manifest rows: **601,581**
- Unique image paths: **601,581**
- Existing paths: **601,581**
- Missing paths: **0**
- Folder-label mismatches: **0**
- `timestamp != frame_id`: **0**
- Unexpected identity IDs: **0**

All camera directories and metadata labels reconcile exactly.

Manifest:

```text
~/DG_ReID_Audit/manifests/RPIfield/rpifield_manifest.csv
```

---

## 5. Identity and Camera Coverage

### Global identities

- Total unique base person IDs: **4,108**
- Known actors: **112**
- Distractors: **3,996**

### Known actors

Known actors are strongly multi-camera overall:

- Mean cameras per actor: **6.99**
- Median: **7**
- Maximum: **12**

Camera-count distribution:

```text
1 camera  : 1 actor
3 cameras : 2
4 cameras : 11
5 cameras : 20
6 cameras : 16
7 cameras : 13
8 cameras : 19
9 cameras : 15
10 cameras: 9
11 cameras: 4
12 cameras: 2
```

One known actor, **PID 89**, appears only in **Cam_2** in the distributed package:

- Images: 289
- Appearance sequences: 1
- Frame range: 240282 → 243006

This is treated as a valid package characteristic, not an automatic annotation error.

### Distractors

Distractors are mostly camera-specific:

- Mean cameras per distractor: **1.25**
- Median: **1**
- Maximum: **5**

Distribution:

```text
1 camera : 3,237
2 cameras:   547
3 cameras:   192
4 cameras:    19
5 cameras:     1
```

---

## 6. Sequence / Appearance Structure

Total appearance sequences:

- **6,577**

Sequence size distribution:

- Minimum: **1 image**
- Median: **79**
- Mean: **91.47**
- 95th percentile: **167.2**
- 99th percentile: **324**
- Maximum: **16,694**

Known-actor sequences:

- Sequences: **1,585**
- Images: **211,644**
- Median images/sequence: **97**
- Mean: **133.53**

Distractor sequences:

- Sequences: **4,992**
- Images: **389,937**
- Median images/sequence: **75**
- Mean: **78.11**

---

## 7. Reappearance Semantics and Official Statistics

Raw suffix counting produced:

- Known labels with `_n` suffix: **803**
- Distractor labels with `_n` suffix: **4**

However, the official RPIfield statistic is:

- **802 known reappearances**

The audit explains the apparent difference exactly.

For every camera, the official `#Reapp` value matches:

```text
number of known appearance sequences
-
number of known base identities in that camera
```

This sums to **802**.

The only unusual person-camera case is:

```text
PID 14
Cam_12
labels = 14_1, 14_2
```

There is no base `14` appearance label in Cam_12.

Therefore:

```text
suffix labels = 2
true extra appearances = 1
```

This explains why Cam_12 has:

```text
80 suffix-labeled appearances
79 official reappearances
```

and why the global raw suffix count is 803 while the official reappearance total is 802.

The four distractor suffix labels are not part of the official known-actor `#Reapp` count.

---

## 8. Temporal / Frame Audit

Within appearance sequences:

- Adjacent image pairs: **595,004**
- Median frame gap: **2**
- 87.12% of adjacent gaps are `<= 2`
- 92.63% are `<= 5`
- 96.67% are `<= 10`
- 99.32% are `<= 50`

Therefore RPIfield has **very strong temporal redundancy**.

Large gaps also exist:

```text
Gap > 100    : 1,930 pairs
Gap > 500    :   420
Gap > 1,000  :   223
Gap > 5,000  :    59
Gap > 10,000 :    28
Gap > 50,000 :    10
```

Largest observed gap:

```text
233,742 frames
```

Because some appearance folders contain very large temporal discontinuities, the audit uses the safer term:

> **appearance sequence / appearance container**

rather than assuming every folder is one continuous physical tracklet.

No duplicate frame IDs were found inside a sequence.

No case was found where the same PID appeared twice at the same camera/frame coordinate.

---

## 9. Crowding

The synchronized frame structure shows substantial multi-person crowding.

- Camera-frame groups: **410,051**
- Camera-frame groups with more than one crop: **132,825**
- Maximum crops in one camera-frame: **10**
- Images belonging to multi-crop frames: **324,355**

Approximately **53.9% of all images** come from camera-frame positions containing multiple person crops.

This explains part of the difficulty of the dataset and also helps explain the observed crop-assignment conflicts.

---

## 10. Image Integrity

All **601,581** images were fully decoded.

Results:

- Readable: **601,581**
- Corrupted: **0**
- Format: **PNG for 100%**
- Mode: **RGB for 100%**

No corruption-based exclusion is needed.

---

## 11. Geometry

RPIfield uses variable-size person crops.

### Overall

Width:

- Minimum: 33
- Median: 75
- 99th percentile: 139
- Maximum: 344

Height:

- Minimum: 53
- Median: 184
- 99th percentile: 340
- Maximum: 675

Aspect ratio:

- Median: **0.408**
- 1st percentile: 0.403
- 99th percentile: 0.412

Unique resolutions:

- **1,347**

The wide range of resolutions represents genuine person-scale variation while the bounding-box shape is normally very stable.

### Orientation anomaly

- Portrait images: **601,580**
- Landscape images: **1**

The single landscape anomaly is:

```text
Cam_2
PID 66
appearance_label 66
frame 181850
129 × 53
aspect ratio 2.434
```

This row is flagged for future training exclusion.

Raw data remains unchanged.

---

## 12. Exact File-Byte Duplicate Audit

SHA256 over stored PNG bytes found:

- Duplicate groups: **14**
- Images involved: **28**
- Extra byte-identical copies: **14**

Group structure:

- Cross-person groups: **7**
- Same-person groups: **7**
- Cross-camera groups: **0**

The cross-person byte duplicates were visually inspected and found to be isolated crop/assignment conflicts rather than evidence that the identity labels should be merged.

---

## 13. Decoded-Pixel Duplicate Audit

A second audit hashed decoded RGB pixels, independent of PNG storage encoding.

Results:

- Pixel-duplicate groups: **644**
- Images involved: **1,290**
- Extra pixel copies: **646**

Group structure:

- Cross-person groups: **630**
- Same-person groups: **14**
- Cross-camera groups: **0**

Group sizes:

```text
642 groups × 2 images
2 groups × 3 images
```

This audit revealed many duplicate visual crops whose PNG byte encodings differ.

---

## 14. Cross-Person Pixel Conflicts

Cross-person conflicts:

- Groups: **630**
- Rows involved: **1,262**
- Unique affected persons: **77**
- Cameras involved: **6**

Cameras:

```text
Cam_1  :  38 groups
Cam_2  :   2
Cam_3  : 203
Cam_4  :   4
Cam_11 :  91
Cam_12 : 292
```

Cam_3, Cam_11 and Cam_12 account for the overwhelming majority of conflicts.

Conflict type:

```text
distractor ↔ distractor : 529 groups
known ↔ known           :  56
known ↔ distractor      :  45
```

Affected known actors:

```text
62, 63, 66, 67, 70, 73, 76, 78,
81, 82, 83, 88, 90, 92, 95, 96
```

Several conflict groups are concentrated in a small number of person pairs, indicating localized/systematic annotation or crop-assignment noise rather than dataset-wide corruption.

No cross-camera pixel duplicates were found.

---

## 15. Duplicate Handling Policy

The raw dataset remains immutable.

### Cross-person pixel conflicts

For future custom supervised training:

```text
exclude all rows belonging to cross-person
pixel-duplicate conflict groups
```

Rows recommended for exclusion from this issue:

- **1,262**

No automatic identity merging is performed.

### Same-person pixel duplicates

- 14 groups
- 28 rows

These are considered temporal redundancy rather than label corruption.

They are flagged, but are not mandatory hard exclusions.

If strict de-duplication is desired, one representative image may later be kept from each group.

---

## 16. Audit Flags

A reproducible flag file was created:

```text
~/DG_ReID_Audit/manifests/RPIfield/rpifield_audit_flags.csv
```

Summary:

- Flagged rows: **1,291**
- Cross-person conflict rows: **1,262**
- Same-person redundancy rows: **28**
- Geometry anomaly rows: **1**
- Recommended hard training exclusions: **1,263**

If only the hard-exclusion rule is applied:

```text
601,581
-
1,263
=
600,318 candidate rows
```

before any optional temporal de-redundancy or sampler subsampling.

---

## 17. Visual / Domain Audit

Visual inspection covered:

1. all 12 camera domains,
2. the same known identities across many cameras,
3. reappearance sequences,
4. crowded synchronized frames,
5. geometry outliers,
6. duplicate conflicts.

### Domain characteristics

RPIfield is primarily:

> **Outdoor multi-camera campus surveillance ReID**

Strong visual variation includes:

- camera viewpoint,
- person scale / camera distance,
- illumination and shadows,
- background,
- camera-specific appearance,
- pose and walking direction,
- temporal reappearances,
- crowding and partial occlusion.

The 12 cameras belong to the same broad campus environment, so RPIfield has weaker **scene diversity** than datasets such as LPW, but strong **camera-network diversity**.

### Reappearances

Reappearances visibly provide later observations of the same identities, but the inspected samples generally preserve the same clothing.

Therefore RPIfield should not be described as a strong clothing-change / long-term wardrobe-change dataset.

### Crowding

The visual sheets confirm that multiple people commonly appear at the same synchronized time and that crops may include neighboring people or partial occlusion.

---

## 18. DG Strengths

RPIfield offers several strong properties for domain-generalization research:

- 12-camera surveillance network,
- synchronized temporal coordinate,
- strong cross-camera appearance variation,
- large number of images,
- explicit reappearance information,
- real person-scale variation,
- outdoor illumination variation,
- crowding and partial occlusion,
- large distractor population,
- clean filesystem/metadata alignment,
- zero corrupted images.

It contributes a domain that is meaningfully different from clean two-camera datasets such as PRID2011.

---

## 19. DG Risks / Limitations

1. **Strong temporal redundancy**
   - Most neighboring crops are only a few frames apart.

2. **Appearance sequence is not always a continuous tracklet**
   - Some containers include very large frame gaps.

3. **Cross-person crop/annotation conflicts**
   - 1,262 rows should be excluded from custom supervised training.

4. **One geometry anomaly**
   - Landscape-oriented crop at Cam_2 / PID 66 / frame 181850.

5. **Same broad scene family**
   - Twelve cameras, but all belong to the same outdoor campus setting.

6. **Known/distractor imbalance**
   - 112 known actors versus 3,996 distractors.

7. **No standard fixed supervised train split is defined by the package**
   - The dataset was designed primarily around temporal ReID evaluation.

8. **Dynamic-gallery benchmark semantics**
   - Care is required if repurposing the dataset for conventional supervised source training.

---

## 20. Recommended Handling if Used as a DG Source

If RPIfield is selected as a source dataset:

- preserve the raw package unchanged,
- train from the manifest rather than raw folder assumptions,
- exclude the **1,262 cross-person pixel-conflict rows**,
- exclude the single landscape geometry anomaly,
- use identity-balanced sampling,
- include camera diversity in sampling,
- strongly control temporal redundancy,
- avoid drawing many neighboring frames from the same appearance sequence,
- consider derived temporal segmentation for sequences containing very large internal gaps,
- do not treat distractors as ordinary cross-camera positives without checking their camera coverage,
- decide explicitly whether all 3,996 distractor IDs participate as supervised classes,
- avoid using the same identities simultaneously for source training and unseen-target evaluation.

A useful future sampler would combine:

```text
identity balance
+
camera diversity
+
appearance/reappearance diversity
+
temporal spacing
```

rather than uniform image sampling.

---

## 21. Final Assessment

**RPIfield is a strong but non-trivial DG source candidate.**

Its major value is not merely its image count. Its value comes from:

- a large synchronized camera network,
- temporal/reappearance structure,
- strong viewpoint/scale/illumination changes,
- crowding,
- and a large distractor population.

However, direct naive training on all 601k images would be undesirable because of temporal redundancy and the discovered crop-assignment conflicts.

### Current project role

> **Strong source candidate, with mandatory audit flags and temporal-aware sampling.**

Final source/target assignment should still be frozen only after all candidate datasets have completed audit.

---

## 22. Final Status

**RPIfield Audit: COMPLETE**

Raw data: unchanged.  
Manifest: complete.  
Metadata reconciliation: complete.  
Temporal audit: complete.  
Integrity audit: complete.  
Geometry audit: complete.  
Byte-duplicate audit: complete.  
Pixel-duplicate audit: complete.  
Visual audit: complete.  
Official-statistics reconciliation: complete.  
Training flags: complete.
