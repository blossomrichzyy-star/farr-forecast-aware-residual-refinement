# FARR open-source main experiment

This repository publishes only the clean FARR main-experiment source code,
configuration, and run instructions. It does **not** distribute data, model
weights, endpoint predictions, labels, logs, checkpoints, or paper result
files. All outputs are generated locally by the user and are ignored by git.

The public code implements the method described in the paper: matched valid
and null endpoint evidence, 18 deterministic features, a shared residual
refiner, horizon-wise conservative gating, Stage 1 training, and validation-
controlled Stage 2 forecast adjustment.

## Release layout

```text
src/       model, features, data loading, endpoint adapter, training, metrics
configs/   default settings and four dataset configurations
scripts/   data validation, endpoint generation, training, evaluation, batching
docs/      data, model, and reproduction instructions
```

The root contains only `README.md`, `requirements.txt`, `.gitignore`,
`LICENSE`, and `THIRD_PARTY_NOTICES.md`.

## Installation

Use a clean Python 3.10+ environment with a PyTorch installation suitable for
the target hardware:

```bash
python -m pip install -r requirements.txt
```

## Local endpoint data

FARR consumes a local endpoint bundle produced by the user's frozen backbone:

```text
<bundle>/
  train/  history.npy  y_valid.npy  y_null.npy  y_true.npy
  val/    history.npy  y_valid.npy  y_null.npy  y_true.npy
  test/   history.npy  y_valid.npy  y_null.npy  y_true.npy
  scaler.npz                         # optional for raw-space metrics
```

`history` is `[N, L, C]`, and endpoint/target arrays are `[N, H, C]`.
Single-channel `[N, L]` and `[N, H]` arrays are also accepted. Valid and null
forecasts must use the same samples, history, preprocessing, channel order,
checkpoint, and deterministic inference settings.

The endpoint generator is deliberately not tied to a private SE-LLM path. Use
`scripts/generate_endpoints.py` with a local adapter implementing
`predict(history, condition)` for `condition="valid"` and `condition="null"`.
See `docs/models.md` and `docs/data.md` before preparing real data.

## Run one main experiment

```bash
python scripts/prepare_data.py --data-root /path/to/endpoint_bundle
python scripts/run_main.py \
  --config configs/etth1.yaml \
  --data-root /path/to/endpoint_bundle \
  --output-dir /path/to/local_outputs/ETTh1-H192
```

The program fits FARR on `train/`, selects Stage 1/Stage 2 using `val/`, and
evaluates `test/` only after the configuration is fixed. Runtime files are
written under the output directory supplied by the user; none are stored in
this release.

## Run all four datasets and horizons

Arrange local bundles as `DATA_ROOT/ETTh1/H96`, `DATA_ROOT/Weather/H192`,
`DATA_ROOT/Traffic/H336`, and so on, then run:

```bash
python scripts/run_suite.py \
  --config configs/suite.yaml \
  --data-root /path/to/DATA_ROOT \
  --output-dir /path/to/local_outputs/suite-main
```

The shell wrappers are equivalent:

```bash
DATA_ROOT=/path/to/DATA_ROOT OUTPUT_DIR=/path/to/local_outputs/suite-main \
  bash scripts/run_main.sh
```

## Evaluate an existing local checkpoint

```bash
python scripts/evaluate.py \
  --config configs/etth1.yaml \
  --data-root /path/to/endpoint_bundle \
  --checkpoint /path/to/local_outputs/farr.pt \
  --phase stage2
```

Use `--metric-space raw` only when the bundle includes a training-fitted
`scaler.npz`. The default is standardized-space MSE/MAE.

## What is intentionally not promised

The repository contains the complete FARR refinement implementation, but it
does not include a frozen SE-LLM checkpoint or a dataset. Reproducing the
paper's exact numerical table therefore requires the user to obtain the
permitted data and backbone, then generate matched endpoints with the adapter.
This release makes that boundary explicit rather than embedding private
weights or prediction caches.

See `docs/reproduction.md` for the end-to-end protocol and
`docs/open_source_checklist.md` before publishing a website link.
