# Tests

```bash
pytest -q
```

| File | Covers |
|---|---|
| `test_data.py` | env expansion, manifest round-trip/validation/discovery, Market-style converter on a synthetic layout, `build_dataset` (copy + eval-only link), subsampling rules, pool validation, audit |
| `test_sampler.py` | domain weights, rank interleaving, P x K batches, domain proportions, DDP rank streams |
| `test_evaluation.py` | metrics (junk handling, chunked == dense), mean-pool, video evaluator with a fake extractor, `evaluate_sets` mean row |
| `test_models.py` | model registry, checksums, checkpoint remap; FastReID dataset registration, `setup_cfg`, OSNet backbone (`fastreid` marker) |
| `test_engine.py` | checkpoint pruning, run restore, time guard decisions (`fastreid` marker) |

Tests marked `fastreid` are skipped automatically when `./fast-reid` is not set up.
Everything runs on CPU with synthetic data; no datasets or weights are needed.
