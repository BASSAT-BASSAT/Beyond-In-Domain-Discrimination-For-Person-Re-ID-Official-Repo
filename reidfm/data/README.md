# `reidfm.data`

## Manifest (`manifest.py`)

Every dataset is converted to one folder:

```text
<data_root>/<key>/manifest.csv   path,pid,camid,dataset,split,tracklet_id,frame
<data_root>/<key>/meta.json      converter options, counts, protocol, version
<data_root>/<key>/images/<split>/<pid>/...
```

- `split` is `train`, `query` or `gallery`; `tracklet_id` groups video frames.
- Test pids that could collide with train pids are namespaced by the converter (`test12`).
- Manifests are found at `<root>/<key>/manifest.csv` or one level deeper
  (`/kaggle/input/<slug>/<key>/manifest.csv`), for every root in `$REIDFM_DATA_ROOT`.
- Video test sets are stored pre-sampled (8 `linspace` frames per tracklet, the evaluation
  rule), so metrics are identical to using all frames while the pool is 10-100x smaller.

## Registry (`registry.py`)

`DatasetSpec` per key: FastReID name (`Market1501M`, ...), image/video, converter, raw folder
names, whether it has train / test splits, restricted license flag, protocol notes.

| key | FastReID name | kind | train | test |
|---|---|---|---|---|
| market1501 | Market1501M | image | yes | yes |
| msmt17 | MSMT17M | image | yes | yes |
| cuhk03np | CUHK03NPM | image | yes | yes |
| last | LaSTM | image | yes | yes |
| occposetrack | OccPoseTrackM | image | yes | yes |
| deepchange | DeepChangeM | image | yes | yes |
| prw | PRWM | image (crops) | yes | yes |
| mot17 / mot20 | MOT17M / MOT20M | video (GT crops) | yes | no |
| mars | MARSM | video | yes | yes |
| lpw | LPWM | video | yes | yes |
| prid2011 | PRID2011M | video | no | yes |
| iust | IUSTM | image | yes | yes |
| rpifield | RPIfieldM | image | yes | yes |
| orbenchrgb | ORBenchRGBM | image | yes | yes |

## Pools (`pool.py`, `configs/datasets/*.yaml`)

A pool lists `sources`, `ood`, `val_sets`, `test_sets`, sampler weights and converter options.
`validate()` hard-fails if an OOD dataset is a source, a source has no train split, or an
eval set has no query/gallery.

## FastReID integration

- `fastreid_datasets.py` registers one `ImageDataset` subclass per key. Train items are
  `(path, "<key>_<pid>", "<key>_<camid>")`, so ids never collide across datasets and the
  sampler reads the domain from the prefix.
- `samplers.py`: `DomainBalancedIdentitySampler` picks a domain per identity slot (weights
  `sqrt_ids` / `ids` / `uniform` x multipliers), then P identities x K images. DDP-safe.

## Converters and audit

See [`converters/README.md`](converters/README.md). `audit.py` checks manifests (schema, sampled
files exist, train/test id leakage, orphan queries), pool roles, and dHash near-duplicates
between known overlapping datasets (MARS / Market / PRW share a campus).
