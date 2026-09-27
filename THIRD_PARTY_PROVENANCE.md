# Third-Party Provenance

## Code

| Component | Where | Source | License | Notes |
|---|---|---|---|---|
| FastReID | `fast-reid/` (cloned, gitignored) | https://github.com/JDAI-CV/fast-reid @ `c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d` | Apache-2.0 | Not vendored. `scripts/setup_fastreid.py` clones this exact commit. We only subclass/register into it (`reidfm/engine`, `reidfm/data/fastreid_datasets.py`) and apply one runtime patch (`reidfm/fastreid_compat.py`: `torch.load(weights_only=False)` for trusted checkpoints on torch >= 2.6). |
| FastReID base configs | referenced via `_BASE_: ../fast-reid/configs/Base-SBS.yml` | FastReID | Apache-2.0 | Our configs inherit and override; nothing copied. |
| OSNet-AIN network | `reidfm/models/backbones/osnet_ain.py` | torchreid (KaiyangZhou/deep-person-reid), via `Person-ReID-BenchMark/reid/models/osnet_ain.py` | MIT | Vendored verbatim network definition, plus a FastReID backbone wrapper. |
| Rank metrics (CMC/mAP/mINP) | `reidfm/evaluation/metrics.py` | Ported from `Person-ReID-BenchMark/reid/metrics.py` (itself following the FastReID/torchreid Market-1501 protocol) | Apache-2.0 / MIT lineage | Same numbers as the benchmark by construction. |
| Checkpoint key remap | `reidfm/models/init.py` | Ported from `Person-ReID-BenchMark/reid/models/fastreid.py` | internal | Handles old FastReID release key layouts (`classifier`, `bnneck`, MGN pools). |

## Pretrained weights

| Key(s) | Source | License / terms |
|---|---|---|
| `sbs_*`, `bot_*`, `agw_*`, `mgn_*` (Market/MSMT/Duke) | FastReID model zoo release v0.1.1 | Apache-2.0 (code); trained on the respective datasets, research use |
| `sbs_s50_mot17`, `sbs_s50_mot20`, `sbs_s50_dancetrack` | DiffMOT release v1.0 | see DiffMOT repo |
| `osnet_ain_msdc` | torchreid model zoo (Google Drive) | MIT (code); trained on MSMT17+DukeMTMC+CUHK03 |
| ImageNet / IBN-Net backbones | downloaded by FastReID on first use | see FastReID `resnet.py` model URLs |

Checksums of files we have verified are recorded on the `ModelSpec` entries in `reidfm/models/zoo.py`.

## Datasets

None are redistributed here. Converters read your local copies. Terms vary; most are
research / non-commercial only. Check each dataset's terms before building a Kaggle
dataset from its converted pool, and keep restricted ones (for example ORBench) in
**private** Kaggle datasets.

| Dataset | Registry key | Terms (summary) | Restricted? |
|---|---|---|---|
| Market-1501 | `market1501` | research use | no |
| MSMT17 | `msmt17` | research use, signed agreement | yes (agreement) |
| CUHK03-NP | `cuhk03np` | research use | no |
| LaST | `last` | research use, request access | yes (request) |
| Occluded-PoseTrack-ReID | `occposetrack` | PoseTrack terms | no |
| DeepChange | `deepchange` | research use, request access | yes (request) |
| PRW | `prw` | research use | no |
| MOT17 / MOT20 | `mot17`, `mot20` | CC BY-NC-SA 3.0 | no |
| MARS | `mars` | research use | no |
| LPW | `lpw` | research use | no |
| PRID2011 | `prid2011` | research use | no |
| IUST_PersonReID | `iust` | research use, request access | yes (request) |
| RPIfield | `rpifield` | research use | no |
| ORBench (RGB subset) | `orbenchrgb` | non-commercial, private sharing only | yes |
