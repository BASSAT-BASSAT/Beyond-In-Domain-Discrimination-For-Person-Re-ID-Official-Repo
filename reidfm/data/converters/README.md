# Converters

Each module turns one raw dataset layout into `SourceRow`s (`src, pid, camid, split,
tracklet_id, frame, box`). `build_dataset()` in `__init__.py` then applies the generic
options, writes images (copies, links, or crops) and the dataset's `manifest.csv` and `meta.json`.

| Module | Datasets | Notes |
|---|---|---|
| `market_style.py` | Market-1501, CUHK03-NP, IUST | `<pid>_c<cam>` names; pid -1 dropped, pid 0 gallery-only |
| `msmt17.py` | MSMT17 V1/V2 | official lists; train = train + val |
| `last.py` | LaST | unique camid per test image (official protocol) |
| `occposetrack.py` | Occluded-PoseTrack-ReID | reidfm query/gallery split unless `query/` + `gallery/` exist |
| `prw.py` | PRW | crops from frames using official boxes |
| `mot.py` | MOT17, MOT20 | GT crops, train only |
| `mars.py` | MARS | same protocol as the benchmark |
| `lpw.py` | LPW | scen2/3 train, scen1 test (view2 query) |
| `prid2011.py` | PRID2011 | OOD only: cam_a query vs cam_b gallery |
| `pattern_dirs.py` | DeepChange, RPIfield | folders + regex, configure in the pool yaml |
| `orbench_rgb.py` | ORBench RGB | official `train_annos.json` + `test_gallery_and_queries.json` |

Rules every converter follows:

- Identities that could collide between train and test are namespaced (`test<pid>`).
- Distractors keep pid `0` and appear only in the gallery.
- Video datasets set `tracklet_id`. Query/gallery tracklets are pre-sampled to the 8 frames
  the evaluator would pick, so the metrics match evaluating on the full tracklets.

Layouts marked "verify" in `reidfm/data/registry.py` (DeepChange, RPIfield, ORBench) could not
be checked against a real copy. Run `scripts/prepare_datasets.py --datasets <key> --dry-run`
first and adjust the options if it finds nothing.
