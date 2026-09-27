# Scripts

Thin CLIs over `reidfm`. Run from the repo root; `--help` on each for all flags.

| Script | Purpose |
|---|---|
| `setup_fastreid.py` / `.sh` | clone FastReID at the pinned commit into `./fast-reid` (`--from` copies a local clone, `--cython` builds the fast rank extension) |
| `prepare_datasets.py` | run converters for a pool, write manifests, audit (leakage, OOD roles, `--dedup-check`), stats table; `--dry-run`, `--audit-only`, `--skip-missing` |
| `list_models.py` | print the model registry (`--family`, `--trained-on`, `--init`, `--backend`, `--json`) |
| `train.py` | FastReID training on the pool; `--num-gpus`, `--resume`, `--restore-from`, `--model`, `--pool`, `--time-budget-h`, `--eval-only`, trailing cfg overrides |
| `evaluate.py` | benchmark-protocol evaluation of a registry model or checkpoint on `--sets val/test/ood/all/<keys>`; writes `results/<name>.json` + `.md` |
| `export_weights.py` | inference checkpoint (no classifier) + standalone FastReID yaml, loadable by the benchmark's `FastReIDEncoder`; `--register` adds it to the model registry |
| `extract_embeddings.py` | embed an image folder or a folder of tracklets to `.npz` |

Typical flow:

```bash
python scripts/prepare_datasets.py --pool configs/datasets/pool_v1.yaml --materialize copy --skip-missing
python scripts/train.py --config-file configs/finetune_sbs_r50ibn_msmt.yml --num-gpus 2 --resume
python scripts/evaluate.py --checkpoint outputs/sbs_r50ibn_msmt_v1/model_best.pth --sets all
python scripts/export_weights.py --run outputs/sbs_r50ibn_msmt_v1 --key reidfm_sbs_r50_ibn_v1 --register
```
