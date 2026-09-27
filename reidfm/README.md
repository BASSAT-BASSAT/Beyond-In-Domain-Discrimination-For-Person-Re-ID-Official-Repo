# `reidfm` package

| Module | What it does |
|---|---|
| `fastreid_compat.py` | `ensure_fastreid()`: puts `./fast-reid` on `sys.path`, makes checkpoint loading work on torch >= 2.6, registers our datasets, sampler and backbones into FastReID. Call it before importing `fastreid`. |
| `config.py` | FastReID cfg + a `REIDFM` node (`POOL`, `INIT_MODEL`, `TIME_BUDGET_H`, `ITERS_PER_EPOCH`, ...). `setup_cfg()` applies the pool (`DATASETS.NAMES/TESTS`), the init model (`MODEL.WEIGHTS` or ImageNet `PRETRAIN`), and resolves `OUTPUT_DIR`. |
| [`data/`](data/README.md) | manifest format, dataset registry, converters, pool configs, FastReID dataset classes, domain-balanced sampler, audit. |
| [`models/`](models/README.md) | model registry (`ModelSpec`), FastReID zoo / DiffMOT / torchreid / ImageNet entries, init-weight preparation, extra backbones. |
| [`engine/`](engine/README.md) | `ReidfmTrainer` (FastReID `DefaultTrainer` subclass), Kaggle hooks, run restore. |
| [`evaluation/`](evaluation/README.md) | benchmark-protocol evaluator (image + tracklet), extractors, metrics. |
| `utils/` | repo paths and env roots, `${VAR:-default}` yaml expansion, run provenance (git + FastReID commit). |

```python
from reidfm import ensure_fastreid, available_models, get_model_spec
ensure_fastreid()
available_models(init="reid_pretrained", family="SBS")
```
