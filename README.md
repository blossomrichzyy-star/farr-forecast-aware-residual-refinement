# FARR: Forecast-Aware Residual Refinement

FARR refines a frozen forecaster using matched valid and null endpoints,
18 deterministic features, residual refinement, horizon-wise gating, and
forecast adjustment. Main experiments cover ETTh1, Weather, Traffic, and
ECL with prediction horizons 96, 192, 336, and 720.

Training uses strict joint validation selection and valid-endpoint fallback.
The method parameters are defined in `src/farr/protocol.py`.

## Repository structure

- `src/farr/`: model, features, data loading, training, and evaluation.
- `scripts/`: endpoint generation, main experiments, and evaluation.
- `docs/`: data preparation, backbone integration, and training protocol.

Data, weights, configurations, and experiment outputs are not distributed.

## Installation

Install the dependencies in `requirements.txt`. Use a PyTorch build
appropriate for the execution hardware when running the frozen forecaster
on a GPU.

```bash
python -m pip install -r requirements.txt
```

## Run with prepared endpoints

The local bundle needs `train/`, `val/`, and `test/` directories. Each contains
`history.npy`, `y_valid.npy`, `y_null.npy`, and `y_true.npy`, all in the same
training-standardized space and sample order. History shape is `[N,672,C]`;
forecast and target shapes are `[N,H,C]`. The loader expands channels into
single-channel complete windows so batch size has the paper's meaning.

```bash
python -B scripts/run_main.py --dataset ETTh1 --horizon 192 \
  --data-root /path/to/endpoints/ETTh1/H192 \
  --output-dir /path/to/local_outputs/ETTh1/H192
```

Model selection uses train and validation inputs only. Test files are opened
after selection is frozen. Standardized MSE/MAE and the selected stage are
generated in the output directory. Weights are saved only with
`--save-checkpoint`. Evaluation restores the selected stage automatically:

```bash
python -B scripts/evaluate.py --checkpoint /path/to/local_outputs/farr.pt \
  --data-root /path/to/endpoints/ETTh1/H192
```

## Run all main settings

Arrange local bundles as `DATA_ROOT/ETTh1/H96`, `DATA_ROOT/Weather/H192`,
etc. Then run:

```bash
python -B scripts/run_all.py --data-root /path/to/DATA_ROOT \
  --output-dir /path/to/local_outputs
python -B scripts/summarize.py --input /path/to/local_outputs/suite.json
```

## Generate matched endpoints

Raw CSV preparation includes chronological splits and training-fitted
standardization. Load a trained, frozen SE-LLM forecaster through a local
factory. See `docs/backbone.md` for the model interface and null condition.

```bash
python -B scripts/generate_endpoints.py --dataset ETTh1 --horizon 192 \
  --csv /path/to/ETTh1.csv --factory my_loader:load_model \
  --backbone-checkpoint /path/to/frozen_predictor.pt \
  --output-dir /path/to/endpoints/ETTh1/H192
```
