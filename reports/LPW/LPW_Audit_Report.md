# LPW Dataset Audit Report

## 1. Dataset
**Name:** Labeled Pedestrian in the Wild (LPW)  
**Audit status:** Completed  
**Project role:** Candidate source dataset for multi-source supervised DG Person Re-Identification.

## 2. Raw structure and cardinality
Observed hierarchy:

```text
pep_256x128/
├── scen1/{view1,view2,view3}/<person>/<image>.jpg
├── scen2/{view1,view2,view3,view4}/<person>/<image>.jpg
└── scen3/{view1,view2,view3,view4}/<person>/<image>.jpg
```

Audit keys:
- Global identity = `(scene, person_folder)`
- Tracklet = `(scene, view, person_folder)`
- JPG stem stored as `image_number`
- `image_number` is **not assumed to be an officially documented frame ID**

| Scene | Views | IDs | Tracklets | Images |
|---|---:|---:|---:|---:|
| scen1 | 3 | 756 | 1,828 | 171,808 |
| scen2 | 4 | 1,751 | 5,122 | 365,189 |
| scen3 | 4 | 224 | 744 | 53,550 |
| **Total** | — | **2,731** | **7,694** | **590,547** |

View coverage per identity:
- 2 views: 1,058 IDs
- 3 views: 1,114 IDs
- 4 views: 559 IDs

Images per identity: min 19, median 203, mean 216.24, max 805.

## 3. Official protocol representation
- Train: `scen2 + scen3`
- Eval: `scen1`
- Probe/query tracklets: `scen1/view2`
- Gallery tracklets: `scen1/view1 + scen1/view3`

Training:
- 418,739 images
- 1,975 IDs
- 5,866 tracklets

Evaluation:
- Probe/query: 63,118 images, 756 IDs, 756 tracklets
- Gallery: 108,690 images, 756 IDs, 1,072 tracklets

All 756 probe IDs have gallery coverage.
- View1 only: 346 IDs
- View3 only: 94 IDs
- Both view1 and view3: 316 IDs
- Missing gallery positive: 0
- Train/eval ID overlap: 0

## 4. Training distribution
Scene imbalance:
- scen2: 365,189 images (87.21%), 1,751 IDs (88.66%), 5,122 tracklets
- scen3: 53,550 images (12.79%), 224 IDs (11.34%), 744 tracklets

Training images per ID:
- min 19
- median 199
- mean 212.02
- p95 414.50
- max 805

Training tracklets per ID:
- 2 tracklets: 618 IDs
- 3 tracklets: 798 IDs
- 4 tracklets: 559 IDs

Training images per tracklet:
- min 8
- median 65
- mean 71.38
- p95 149
- max 298

## 5. Image integrity and geometry
All **590,547** images were fully decoded.

- Readable: 590,547
- Corrupted: 0
- JPEG: 100%
- RGB: 100%
- Resolution: 128×256 for 100%
- Aspect ratio W/H: 0.5 for 100%
- Unique stored resolutions: 1

No geometry-based exclusion rule is needed.

## 6. Numeric-sequence redundancy
The JPG stem is analyzed only as `image_number`, not as a confirmed frame index.

Across consecutive images within each tracklet:
- Consecutive pairs: 582,853
- Min gap: 1
- Median gap: 6
- Mean gap: 12.64
- P95: 11
- P99: 49
- Max: 198,740

Proximity:
- gap <= 5: 38.38%
- gap <= 10: 91.92%
- gap <= 20: 98.64%
- gap <= 50: 99.01%
- gap <= 100: 99.38%

Interpretation: LPW has strong within-tracklet numeric-sequence proximity and therefore substantial sequence redundancy. Because `image_number` is not verified as an official frame ID, this is not described as confirmed temporal/frame redundancy.

## 7. Exact duplicate audit
SHA256 hashing found:
- Duplicate groups: 472
- Images involved: 944
- Extra copies: 472

All duplicate groups were:
- Cross-person: 472/472
- Cross-tracklet: 472/472
- Cross-scene: 0
- Cross-view: 0
- Train↔Eval: 0
- Query↔Gallery: 0

The issue is highly concentrated:
- 8 affected IDs
- 4 affected identity pairs
- 8 affected tracklets

Pairs:
1. `scen1-670 ↔ scen1-672`: 169 groups
2. `scen2-819 ↔ scen2-826`: 146 groups
3. `scen1-633 ↔ scen1-644`: 127 groups
4. `scen1-1053 ↔ scen1-27`: 30 groups

Each affected tracklet is duplicated at 100% of its images.

### Training-side conflict
`scen2-819` and `scen2-826` share a complete duplicated `view2` tracklet of 146 exact images under two different IDs. Their non-duplicated `view3` samples show clearly different appearances, so this is treated as a real training label conflict / conflicting duplicated tracklet.

Future custom DG training policy:
- flag `scen2-view2-819`
- flag `scen2-view2-826`
- exclude both conflicting tracklets from the custom training manifest
- do not modify raw data

### Evaluation-side anomalies
The duplicated gallery pairs are:
- `scen1-670 ↔ scen1-672`
- `scen1-633 ↔ scen1-644`
- `scen1-1053 ↔ scen1-27`

The duplicated view3 tracklets plus visual comparison of other available views provide strong evidence consistent with identity fragmentation / duplicate identity labels. The official benchmark should remain unchanged; document the issue rather than silently correcting it.

## 8. General visual audit
Representative samples from all views in all three scenes were inspected.

### Scene 1
Predominantly outdoor/campus-like imagery. The sample shows varied illumination, front/back/side views, blur, partial occlusion, background pedestrians, bicycles, backpacks, and varied target scale.

### Scene 2
Predominantly indoor commercial/public-space imagery. The sample shows crowding, clutter, partial occlusion, shopping bags/luggage, front/back/side views, and noticeable variation in blur and image quality.

### Scene 3
Crowded public-space imagery with a different background/illumination profile from the other scenes. The sample shows front/back/side viewpoints, partial occlusion, blur, nearby pedestrians, and varied clothing/accessories.

### Visual conclusion
- Person crops are generally usable and centered.
- Stored geometry is uniform, but visual difficulty varies substantially.
- Blur, partial occlusion, crowd interference, and background variation are present.
- There is useful viewpoint, pose, and appearance diversity.
- No obvious malformed/corrupt crop pattern was observed in the general sample.
- No automatic visual-quality deletion is recommended.

## 9. DG strengths
LPW is attractive for source-domain training because of:
- very large scale
- 2,731 identities
- multi-view coverage for every identity
- multiple scenes/views
- 7,694 tracklets
- crowding and occlusion
- pose/viewpoint diversity
- blur and illumination variation
- substantial background/domain variation

## 10. DG risks
- scen2 dominates the training split
- identity-size imbalance
- tracklet-size imbalance
- strong within-tracklet sequence redundancy
- one known training label-conflict pair
- three evaluation-side duplicated identity anomalies
- image-number semantics are not fully documented

## 11. Recommended future handling
Keep the raw dataset immutable.

For the future custom multi-source DG training manifest:
- use identity-balanced sampling
- encourage view/tracklet diversity
- avoid repeatedly sampling near-neighbor images from one tracklet
- consider scene-aware balancing so scen2 does not dominate LPW
- exclude only the two known conflicting training tracklets
- leave the official evaluation protocol unchanged

No other deletion rule is justified by the current audit.

## 12. Final decision
**LPW is accepted as a strong source candidate for the multi-source DG ReID experiment.**

Main value:
- scale
- multi-view identities
- multiple scenes
- realistic crowding
- pose/viewpoint diversity
- blur/occlusion
- large tracklet coverage

Main risks are manageable through the future training manifest and sampler.

**LPW Audit: COMPLETE**

Raw data remains unchanged. Known anomalies are documented for later handling during final custom training-manifest construction.
