# FARR: Forecast-Aware Residual Refinement

Source code for the FARR main experiment. FARR takes matched conditioned and null forecasts from a frozen backbone, predicts an anchor-relative residual, and gates the adjustment across the forecast horizon.

This repository contains the implementation, configurations, and run instructions. Benchmark data, frozen backbone weights, endpoint arrays, checkpoints, and paper result files are **not distributed** here.

## Repository contents

| Path | What it contains |
| --- | --- |
| [`src/farr/`](src/farr/) | Model, features, data loading, training, and metrics |
| [`configs/`](configs/) | Default and dataset-specific experiment settings |
| [`scripts/`](scripts/) | Endpoint preparation, training, evaluation, and batch runs |
| [`docs/`](docs/) | Data format, backbone adapter, and reproduction protocol |

## Install

Use Python 3.10 or newer and a PyTorch installation suitable for your CPU or GPU:

```bash
python -m pip install -r requirements.txt
```

## Prepare endpoint data

FARR expects a local bundle produced by your frozen forecasting backbone:

```text
<bundle>/
  train/
    history.npy  y_valid.npy  y_null.npy  y_true.npy
  val/
    history.npy  y_valid.npy  y_null.npy  y_true.npy
  test/
    history.npy  y_valid.npy  y_null.npy  y_true.npy
  scaler.npz  # optional; needed for raw-space metrics
```

`history` has shape `[N, L, C]`; forecasts and targets have shape `[N, H, C]`. Single-channel `[N, L]` and `[N, H]` arrays are also accepted. Valid and null forecasts must use the same samples, preprocessing, channel order, backbone checkpoint, and deterministic inference settings.

To connect a backbone, implement `predict(history, condition)` for `condition="valid"` and `condition="null"`, then use [`scripts/generate_endpoints.py`](scripts/generate_endpoints.py). See the [data guide](docs/data.md) and [model guide](docs/models.md) before preparing real data.

## Run an experiment

From the repository root, validate your local bundle and run one dataset/horizon:

```bash
python scripts/prepare_data.py --data-root /path/to/endpoint_bundle
python scripts/run_main.py \
  --config configs/etth1.yaml \
  --data-root /path/to/endpoint_bundle \
  --output-dir /path/to/local_outputs/ETTh1-H192
```

The run trains on `train/`, selects Stage 1 or Stage 2 using `val/`, and evaluates `test/` after selection. Outputs are written to the directory you specify.

For all four datasets and horizons, arrange bundles as `DATA_ROOT/ETTh1/H96`, `DATA_ROOT/Weather/H192`, and so on:

```bash
python scripts/run_suite.py \
  --config configs/suite.yaml \
  --data-root /path/to/DATA_ROOT \
  --output-dir /path/to/local_outputs/suite-main
```

To evaluate an existing checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/etth1.yaml \
  --data-root /path/to/endpoint_bundle \
  --checkpoint /path/to/local_outputs/farr.pt \
  --phase stage2
```

Metrics default to standardized-space MSE and MAE. Use `--metric-space raw` only if your bundle includes a training-fitted `scaler.npz`.

## Reproducibility and scope

See the [reproduction protocol](docs/reproduction.md) for the full workflow. Exact paper numbers require the permitted datasets, frozen backbone, and matched endpoints; these are not included in the repository.

License: [MIT](LICENSE). Third-party information: [notices](THIRD_PARTY_NOTICES.md).
