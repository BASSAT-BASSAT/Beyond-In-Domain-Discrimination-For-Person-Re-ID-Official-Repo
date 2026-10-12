# PRID2011 Dataset Audit Report

## 1. Dataset
**Name:** Person Re-Identification 2011 (PRID2011)  
**Audit status:** Completed  
**Project context:** Candidate dataset for the multi-source Domain-Generalizable Person Re-Identification study.

---

## 2. Package Structure

The package contains two versions of the same underlying person trajectories:

```text
prid_2011/
├── single_shot/
│   ├── cam_a/
│   └── cam_b/
└── multi_shot/
    ├── cam_a/
    │   └── person_xxxx/
    │       ├── 0001.png
    │       ├── 0002.png
    │       └── ...
    └── cam_b/
        └── person_xxxx/
            ├── 0001.png
            ├── 0002.png
            └── ...
```

The dataset uses two fixed surveillance cameras with substantial viewpoint, illumination, background, and camera-characteristic differences.

---

## 3. Dataset Cardinality

### Single-shot
- cam_a: **385 images / 385 local persons**
- cam_b: **749 images / 749 local persons**
- Total: **1,134 images**

### Multi-shot
- cam_a: **43,477 images / 385 trajectories**
- cam_b: **51,510 images / 749 trajectories**
- Total: **94,987 images / 1,134 trajectories**

### Whole package
- Total PNG files: **96,121**

---

## 4. Identity Namespace

The README explicitly states that only persons **0001–0200** correspond across the two cameras.

Therefore:

- Cross-camera matched real identities: **200**
- A-only identities: **185**
- B-only identities: **549**
- Total distinct real identities: **934**

The same numeric local ID above 0200 must not be treated as the same person across cameras.

Audit global-ID rule:

```text
0001..0200:
    shared-0001 .. shared-0200

cam_a 0201..0385:
    cam_a-0201 .. cam_a-0385

cam_b 0201..0749:
    cam_b-0201 .. cam_b-0749
```

This prevents false cross-camera identity matches.

---

## 5. Official Evaluation Protocol

PRID2011 does not provide an official train split in the package README. It defines two evaluation directions.

### A → B
- Probe: first **200 persons of cam_a**
- Gallery: all **749 persons of cam_b**
- True cross-camera matches: 200
- B-only gallery distractors: **549**
- Probe IDs missing from gallery: **0**

### B → A
- Probe: first **200 persons of cam_b**
- Gallery: all **385 persons of cam_a**
- True cross-camera matches: 200
- A-only gallery distractors: **185**
- Probe IDs missing from gallery: **0**

The official evaluation logic is internally consistent.

---

## 6. Single-shot vs Multi-shot Relationship

The README describes the single-shot version as one randomly selected image per person trajectory.

This was verified computationally.

Across all **1,134 single-shot images**:

- **607** are byte-for-byte identical to exactly one image in the corresponding multi-shot trajectory.
- The remaining **527** are not byte-identical, but are **pixel-identical** to exactly one image in the corresponding multi-shot trajectory.
- Still unmatched after decoded-pixel comparison: **0**
- Single-shot images with more than one matching multi-shot image: **0**

Therefore:

> The single-shot set is a one-image-per-trajectory subset of the multi-shot visual content and should not be added on top of the multi-shot set as additional training data.

The byte-level differences in 527 files are storage/encoding differences, not different visual samples.

---

## 7. Image Integrity and Geometry

All **96,121** PNG files were fully decoded.

Results:

- Readable: **96,121**
- Corrupted: **0**
- Format: **PNG for 100%**
- Mode: **RGB for 100%**
- Width: **64 pixels for 100%**
- Height: **128 pixels for 100%**
- Aspect ratio W/H: **0.5 for 100%**
- Unique stored resolutions: **1**

No geometry-based exclusion rule is required.

---

## 8. Multi-shot Sequence Integrity

Multi-shot trajectories:

- Total trajectories: **1,134**
- Total images: **94,987**
- Minimum images/trajectory: **5**
- Median: **85**
- Mean: **83.76**
- Maximum: **675**

By camera:

### cam_a
- Trajectories: 385
- Images: 43,477
- Minimum: 5
- Median: 116
- Mean: 112.93
- Maximum: 675

### cam_b
- Trajectories: 749
- Images: 51,510
- Minimum: 5
- Median: 70
- Mean: 68.77
- Maximum: 290

Sequence-index integrity:

- Trajectories starting at index 1: **1,134 / 1,134**
- Duplicate sequence indices: **0**
- Non-contiguous trajectories: **0**
- Missing sequence indices: **0**
- Consecutive index pairs: **93,853**
- Gap = 1: **100%**
- Gap > 1: **0**

The numeric filenames are therefore clean contiguous **sequence indices**. They are not assumed to be original video-frame IDs.

---

## 9. Exact Duplicate Audit Within Multi-shot

SHA256 hashing was performed on the **94,987 multi-shot images**.

Results:

- Exact duplicate groups: **0**
- Images in duplicate groups: **0**
- Extra duplicate copies: **0**

No exact duplicate conflicts were found:
- no same-person duplicated files,
- no cross-person duplicated files,
- no cross-trajectory duplicated files,
- no cross-camera duplicated files.

---

## 10. Visual Audit

Three visual contact sheets were inspected:

1. matched cross-camera identities,
2. cam_a trajectory samples,
3. cam_b trajectory samples.

### Cross-camera matched identities

The sampled first-200 identity pairs show visually consistent clothing/body appearance across cam_a and cam_b while also showing a strong camera/domain shift.

The main cross-camera differences visible in the samples are:

- different color response / illumination,
- different background appearance,
- different viewing angle,
- different apparent scale and sharpness,
- camera-specific image characteristics.

The visual evidence is consistent with the README-defined cross-camera matching.

### cam_a trajectories

Within each sampled trajectory, the target remains visually consistent while pose changes along the walking sequence.

The trajectories show:
- repeated neighboring appearances,
- walking-cycle pose variation,
- mostly stable camera/background characteristics,
- some blur and low-detail frames,
- generally clean full-body crops.

### cam_b trajectories

cam_b shows the same trajectory behavior but with a strong camera-specific color/illumination appearance in many samples.

The sampled sequences show:
- substantial within-trajectory visual redundancy,
- gradual pose changes,
- stable camera/background appearance,
- occasional blur,
- generally usable pedestrian crops.

### Visual conclusion

PRID2011 is visually clean but strongly camera-specific. Its main diversity comes from **cross-camera appearance shift and walking-pose variation**, not from many different scenes or camera networks.

The 64×128 resolution is relatively low compared with several other project datasets.

No visual anomaly was found that justifies automatic deletion.

---

## 11. DG Strengths

PRID2011 offers useful properties for DG research:

- clean labels and directory structure,
- explicit two-camera cross-view correspondences,
- strong viewpoint and illumination shift,
- different camera response / background appearance,
- multiple poses per trajectory,
- no corrupted files,
- no exact duplicates inside multi-shot,
- completely clean sequence indexing.

It can contribute a compact but clean **cross-camera domain-shift source**.

---

## 12. DG Risks / Limitations

1. **Only 200 identities are matched across both cameras.**
   - The remaining 734 real identities are camera-specific.

2. **No official training split is provided.**
   - PRID2011 is primarily specified through A→B and B→A evaluation protocols.

3. **Strong within-trajectory redundancy.**
   - Neighboring sequence images are visually very similar.

4. **Only two cameras / one surveillance setup family.**
   - Domain diversity is narrower than large multi-scene datasets such as LPW.

5. **Low stored resolution.**
   - All crops are 64×128.

6. **Trajectory-size imbalance.**
   - Sequence lengths range from 5 to 675 images.

7. **Single-shot must not be counted as additional independent data.**
   - It is visually contained within multi-shot.

---

## 13. Recommended Handling for Future DG Experiments

Raw data should remain immutable.

If PRID2011 is later selected as a source dataset:

- use **multi-shot** as the authoritative image source,
- do not append single-shot images as additional samples,
- use the corrected global identity namespace,
- use identity-balanced sampling,
- use trajectory-aware subsampling to reduce sequence redundancy,
- treat the 200 shared IDs as the strongest cross-camera supervised component,
- carefully decide whether the 734 camera-only IDs should participate in training,
- do not use PRID2011 simultaneously as both source training data and an unseen target benchmark without an explicit leakage-safe split.

Because there is no official train split, the exact source-training policy should be frozen only during the final multi-source experiment design.

---

## 14. Final Decision

**PRID2011 is a clean but relatively small / narrow-domain dataset.**

Current project assessment:

> **Conditional source candidate / useful cross-camera diversity source.**

It is not as strong a source candidate as LPW by scale, but it provides a distinctive two-camera surveillance domain with a strong illumination/viewpoint shift and very clean annotations.

Its final role should be decided only after all candidate datasets have completed audit.

### Final status

**PRID2011 Audit: COMPLETE**

Raw data remains unchanged.  
No cleaning is required at this stage.  
Future handling decisions belong to the final multi-source training-manifest and sampler design.
