# Training readiness checklist

Status after **pool v2** / paper protocol implementation. See [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md).

## 1. Dataset audits

| Layer | Scope | Location |
| --- | --- | --- |
| **DG-ReID unified audit (v1)** | 9 candidate sources (PRW excluded) | `reports/`, `manifests/`, `unified/`, [DG-ReID-AUDIT.md](DG-ReID-AUDIT.md) |
| **MOT audit** | MOT17 / MOT20 train GT stats | `scripts/MOT17/`, `scripts/MOT20/`, `reports/MOT17/`, `reports/MOT20/` |
| **reidfm pool** | `pool_v2.yaml` — 11 train domains, 4 held-out test domains | `configs/datasets/pool_v2.yaml` |

**MOT download:** `python scripts/download_mot.py` (license gate + layout check).

## 2. Manifest formats

| Schema | Role |
| --- | --- |
| Unified v1 (19 columns) | Audit / metadata |
| reidfm manifest (7 columns) | FastReID training |

```bash
python scripts/prepare_datasets.py --pool configs/datasets/pool_v2.yaml
python scripts/export_unified_to_manifest.py --unified <path.csv> --out pool/<key>
python scripts/build_train_val_split.py --pool configs/datasets/pool_v2.yaml
```

## 3. Pool v2 policy (frozen)

- **Train:** 9 audit sources + MOT17 + MOT20 (incl. IUST, PRID2011 train crops).
- **Never train:** Market, MSMT, MARS, Duke, **PRW**.
- **Val:** cluster holdout (`splits/pool_v2_train_ids.json`, `pool/poolv2val`).
- **Test:** Market, MSMT, MARS, Duke only (`evaluate.py --sets test`).

## 4. Before E2 training

1. Materialize all `pool_v2` sources under `$REIDFM_DATA_ROOT`.
2. `prepare_datasets.py --audit-only --dedup-check`.
3. `build_train_val_split.py` (creates train filter + val manifest).
4. `pytest -q` and smoke config on `pool_smoke` if needed.
5. Train `configs/finetune_sbs_r50ibn_msmt.yml` (defaults to `pool_v2`).

## 5. Legacy

`pool_v1.yaml` remains for historical comparisons (PRW + Market/MSMT in train).
