# `reidfm.evaluation`

Ported from `Person-ReID-BenchMark` so numbers are directly comparable to its tables.

- **Protocol:** cosine distance on L2-normalized features; gallery entries with the same pid
  **and** camera as the query are junk; CMC (R1/R5/R10), mAP, mINP. Distances are computed in
  chunks, so large galleries (MSMT17, LaST) fit in memory.
- **Video sets:** `linspace` sample of up to 8 frames per tracklet (`sampling.py`), features
  mean-pooled then L2-normalized (`aggregate.py`). FastReID frame features are not normalized
  before pooling; OSNet frame features are, both as in the benchmark.
- **Extractors (`extractors.py`):**
  - `FastReIDExtractor`: cv2 RGB, bicubic resize to `INPUT.SIZE_TEST`, 0-255 pixels (the model
    normalizes), optional fp16 autocast.
  - `OSNetExtractor`: torchreid OSNet-AIN head at 256x128 with ImageNet normalization.
  - `build_extractor(model_key=... | checkpoint=...)`: registry entry or a training checkpoint
    (uses `config.yaml` next to it).
- **Evaluator (`evaluator.py`):** `evaluate_sets(keys, extractor)` loads each manifest, runs the
  protocol, and adds a `mean` row; `results_markdown()` renders the table.

`scripts/evaluate.py` wraps it and also reports the in-domain and OOD means separately.
