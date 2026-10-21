# Person-Re-ID Foundational Model (`reidfm`)

Fine-tune one **general** person-ReID encoder on a pool of datasets, instead of one model per
dataset. The benchmark (`Person-ReID-BenchMark`) showed why: FastReID SBS models reach ~96
Rank-1 in-domain but fall to 7-21 Rank-1 on MSMT17; only the multi-source OSNet-AIN model
transferred well.

- **Starting points:** FastReID `msmt_sbs_R50-ibn` / `msmt_sbs_R101-ibn`, and the same recipe
  from an ImageNet R50-IBN as a control.
- **Training:** FastReID's own `DefaultTrainer` + SBS recipe (cloned at a pinned commit, never
  edited). `reidfm` only plugs datasets, a domain-balanced sampler, and Kaggle-safe hooks into it.
- **Evaluation:** the benchmark protocol, unchanged, on every dataset's test split plus fully
  unseen OOD sets (PRID2011 and IUST by default).
- **Runtime:** Kaggle (2x T4, 12 h sessions, automatic stop + resume) or any local GPU.

```mermaid
flowchart LR
  raw["raw datasets"] --> conv["reidfm converters"] --> man["crops + manifest.csv"]
  man --> reg["FastReID DATASET_REGISTRY"]
  zoo["reidfm MODEL_REGISTRY"] --> cfg["FastReID cfg + init weights"]
  reg --> trainer["ReidfmTrainer (DefaultTrainer)"]
  cfg --> trainer
  trainer --> ckpt["model_best.pth"]
  ckpt --> eval["evaluator (benchmark protocol)"]
  ckpt --> export["export -> benchmark FastReIDEncoder"]
```

## Quick start

```bash
pip install -e ".[train,dev]"
python scripts/setup_fastreid.py            # clones JDAI-CV/fast-reid @ c9bc3ce into ./fast-reid
python scripts/list_models.py               # registered starting points

# 1. build the pool (raw data under $REIDFM_RAW_ROOT)
python scripts/prepare_datasets.py --pool configs/datasets/pool_v2.yaml --materialize copy --skip-missing
python scripts/build_train_val_split.py --pool configs/datasets/pool_v2.yaml

# 2. zero-shot baselines (E0)
python scripts/evaluate.py --model sbs_r50_ibn_msmt --sets all
python scripts/evaluate.py --model osnet_ain_msdc --sets all

# 3. train (E2 = main candidate), resumable
python scripts/train.py --config-file configs/finetune_sbs_r50ibn_msmt.yml --num-gpus 2 --resume

# 4. evaluate + export
python scripts/evaluate.py --checkpoint outputs/sbs_r50ibn_msmt_v1/model_best.pth --pool configs/datasets/pool_v2.yaml --sets test
python scripts/export_weights.py --run outputs/sbs_r50ibn_msmt_v1 --key reidfm_sbs_r50_ibn_v1 --register
```

On Kaggle, use the notebooks in [`notebooks/`](notebooks/README.md).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `REIDFM_RAW_ROOT` | `./raw` | raw dataset folders (converter input) |
| `REIDFM_DATA_ROOT` | `./pool` | converted pool; `os.pathsep`-separated list allowed (Kaggle: `/kaggle/input`) |
| `REIDFM_OUTPUT_ROOT` | `./outputs` | training runs (relative `OUTPUT_DIR`s land here) |
| `REIDFM_WEIGHTS_ROOT` | `./weights` | downloaded / converted checkpoints |
| `REIDFM_TIME_BUDGET_H` | `11` | stop cleanly after this many hours (0 = off) |
| `REIDFM_SESSION_START` | process start | unix time the session began (set by the Kaggle notebook) |
| `REIDFM_FASTREID_ROOT` | `./fast-reid` | FastReID checkout |

See `.env.example`.

## Datasets (pool v2 — paper protocol)

| Role | Datasets |
|---|---|
| Train sources | Nine unified-audit sources (incl. IUST, PRID2011) + MOT17 + MOT20 (**no PRW**, no Market/MSMT/MARS/Duke) |
| Validation during training | Cluster holdout (`splits/pool_v2_*`, manifest `pool/poolv2val`) |
| Test (reported) | Market1501, MSMT17, MARS, DukeMTMC, Occluded-Duke (held-out) |

See **[docs/PAPER_PROTOCOL.md](docs/PAPER_PROTOCOL.md)**. Caps and strides keep video/MOT sources from dominating (see `configs/datasets/pool_v2.yaml`);
the sampler weights domains by `sqrt(#ids)`. DukeMTMC is excluded (withdrawn). Licenses and
restricted datasets are listed in [`THIRD_PARTY_PROVENANCE.md`](THIRD_PARTY_PROVENANCE.md).

## Experiments

| Goal | Config / command |
|------|------------------|
| **Train (Kaggle)** | `configs/kaggle/primary_sbs_r50_poolv2.yml` via `notebooks/01_kaggle_train.ipynb` |
| Loss ablations | `configs/kaggle/loss_study_B0.yml` … `B7.yml` |
| Zero-shot zoo eval | `scripts/evaluate.py --model sbs_r50_ibn_msmt` (etc.) |
| Local train | `python scripts/train.py --config-file configs/kaggle/<preset>.yml` |

## Layout

```text
reidfm/            library (data, models, engine, evaluation, utils)   -> reidfm/README.md
configs/           FastReID training configs, pools, model registry     -> configs/README.md
scripts/           thin CLIs                                             -> scripts/README.md
notebooks/         Kaggle / data-prep notebooks                          -> notebooks/README.md
tests/             pytest suite                                          -> tests/README.md
fast-reid/         gitignored, created by scripts/setup_fastreid.py
```

## Tests

```bash
pytest -q            # FastReID-dependent tests are skipped if ./fast-reid is missing
```
## Dataset auditing (DG-ReID)

Ten source datasets were audited on `research/dg-reid-audit` (merged into main). Candidate
manifests, per-dataset reports, and the unified metadata pipeline live under `manifests/`,
`reports/`, `scripts/<dataset>/`, and `unified/`.

Full research snapshot, inventory tables, and workflow:
**[docs/DG-ReID-AUDIT.md](docs/DG-ReID-AUDIT.md)**.

**Note:** the audit excluded **PRW** from the nine-dataset unified pool (Market1501 overlap).
Training uses `configs/datasets/pool_v2.yaml` only.

**Before training:** audits, CSV formats, MOT17/20, and `clothes_id` rules are summarized in
**[docs/TRAINING_READINESS.md](docs/TRAINING_READINESS.md)**.
