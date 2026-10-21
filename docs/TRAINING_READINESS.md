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
- **Test:** Market, MSMT, MARS, Duke, Occluded-Duke (`evaluate.py --sets test`).

## 4. Before training

1. Materialize all `pool_v2` sources under `$REIDFM_DATA_ROOT`.
2. `prepare_datasets.py --audit-only --dedup-check`.
3. `build_train_val_split.py` (creates train filter + val manifest).
4. `pytest -q` and `configs/kaggle/smoke_sbs_r50.yml` on `pool_smoke` if needed.
5. Train via `configs/kaggle/primary_sbs_r50_poolv2.yml` (Kaggle notebook or `scripts/train.py`).
