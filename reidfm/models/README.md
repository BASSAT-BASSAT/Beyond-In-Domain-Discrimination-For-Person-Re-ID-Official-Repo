# `reidfm.models`

## Registry (`registry.py`, `zoo.py`)

`ModelSpec` records what a model is and how to start FastReID from it: key, family, backbone,
`trained_on`, `init` (`reid_pretrained` | `imagenet` | `trained`), backend (`fastreid` |
`osnet`), config yaml, weights URL / file / sha256, input size, feature dim.

Registered on import of `zoo.py`:

- FastReID v0.1.1 zoo: `{sbs,bot,agw}_{r50,r50_ibn,s50,r101_ibn}_{market,msmt,duke}`, `mgn_r50_ibn_{market,duke}`;
- DiffMOT SBS-S50: `sbs_s50_{mot17,mot20,dancetrack}`;
- torchreid `osnet_ain_msdc` (MSMT17+Duke+CUHK03);
- ImageNet controls: `sbs_r50_ibn_imagenet`, `sbs_r101_ibn_imagenet`;
- models trained here, from `configs/models/trained.yaml` (written by `scripts/export_weights.py --register`).

```bash
python scripts/list_models.py --family SBS --trained-on MSMT17
```

## Init weights (`init.py`)

`prepare_init_weights(spec, cfg)` downloads the checkpoint (sha256-checked when known),
remaps old zoo keys (`bnneck` -> `bottleneck`, MGN pools), drops the classifier (the pooled
id count differs) and any shape mismatches, and writes
`$REIDFM_WEIGHTS_ROOT/converted/<key>.pth` for `MODEL.WEIGHTS`. It warns when the run's
architecture differs from the checkpoint's (depth, IBN, non-local, pooling, ...).

## Backbones (`backbones/`)

`build_osnet_ain_backbone` (OSNet-AIN x1.0 trunk, 512-d) is registered into FastReID's
`BACKBONE_REGISTRY`, so the torchreid multi-source model can be fine-tuned with the same
recipe (`configs/models/osnet_ain_x1_0.yml`). Zero-shot evaluation of `osnet_ain_msdc` uses the
native torchreid head, like the benchmark.
