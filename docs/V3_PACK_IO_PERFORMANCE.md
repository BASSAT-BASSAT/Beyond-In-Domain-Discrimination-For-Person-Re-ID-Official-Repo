# V3 PACK I/O diagnostics and shared SQLite index cache

## Why this exists

An observed two-GPU R0/Seed0 training log reported 1.4256 seconds per
iteration, of which 0.9868 seconds were `data_time` (~69%). The slowdown
cannot be attributed to DDP alone. The original `V3PackReader` also copied
and validated a new embedded SQLite index for every worker/domain pair.

The revised reader shares a **verified local SQLite index** across workers
using a Linux `flock` and atomic `os.replace`. Each reader validates the
local index SHA256 against the original PACK footer before opening it. Every
image payload is still read from its original PACK and SHA256 checked.

This is a **candidate fix for repeated index extraction**, not yet a measured
fix for sustained random-access PACK latency on Kaggle. This code must be
benchmarked there before a speedup is claimed.

## Settings

- Enabled by default on Linux.
- Optional `REIDFM_V3_INDEX_CACHE_DIR=/kaggle/working/reidfm_v3_indices`
  chooses a shared **local writable** cache location. By default the cache is
  `${TMPDIR:-/tmp}/reidfm_v3_indices`.
- `REIDFM_V3_INDEX_CACHE=0` disables caching, restoring the original
  private, temporary-index behavior (also used automatically on Windows).
- Cached indices take extra disk space equal to the embedded SQLite indices,
  **not the full 36.59-GiB PACK payloads**. Check available local disk.
- Do not place the cache inside read-only `/kaggle/input`.

## Diagnose without training

From the repository root, after attaching the PACKs and downloading the
exact manifest, run (replace both paths with real paths in that notebook):

```bash
python scripts/diagnose_v3_pack_io.py \
  --manifest /kaggle/working/REPO/unified/final_dataset/Global/train_manifest.csv \
  --pack-root /kaggle/input/DATASET/PACK_DIRECTORY \
  --samples-per-domain 30 \
  --output /kaggle/working/v3_pack_io_report.json
```

The report includes domain-level cold index open time, index size, read
latency (mean, p50, p95), payload-read-plus-hash timing, and SQLite query
plans. A `SCAN images` plan can explain sustained lookup latency and warrants
a separate, verified improvement to the lookup/index strategy.

For an apples-to-apples A/B comparison, run the diagnostic once with
`REIDFM_V3_INDEX_CACHE=0`, then with `REIDFM_V3_INDEX_CACHE=1` in equivalent
Kaggle conditions. The diagnostic processes a bounded reservoir of images
from the full training manifest; it does not train the model.

## Safety and checkpoint warning

1. The 2-GPU global batch (128), sampler, model, optimizer, loss, training
   transforms, freeze-counter, DDP and model checkpoint logic are unchanged.
2. All seven local synthetic PACK tests pass via
   `python -m unittest discover -s tests -p test_v3_pack_index_cache.py -v`.
3. Kaggle performance has **not** yet been measured with this code; no
   2.44-steps/second promise can be made yet.
4. The V3 research fingerprint explicitly SHA256-hashes
   `reidfm/data/v3_pack_reader.py`. Therefore this change alters the
   checkpoint fingerprint. **Do not resume old V3 checkpoints with this
   revision** until the research team explicitly validates or adopts a safe
   migration; do not bypass the fingerprint guard.
5. Run the I/O diagnostic first, then a short identical-configuration two-GPU
   training comparison. Publish to all 27 notebooks only after data parity,
   speed and checkpoint/resume semantics have been verified.
