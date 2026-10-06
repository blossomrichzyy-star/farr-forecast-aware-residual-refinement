# Reproduction protocol

## Environment

Use Python 3.10 or newer, install `requirements.txt`, and use a PyTorch build
appropriate for the selected CPU/GPU. Record the package versions, GPU model,
CUDA version, and the commit of this source tree in your own experiment log.

## Data preparation

1. Download the permitted benchmark files from their official sources.
2. Fit the scaler on the training split only.
3. Create the train/validation/test history and target arrays.
4. Run the frozen backbone twice with identical settings, changing only the
   valid/null conditioning operation.
5. Save the endpoint bundle and run `scripts/prepare_data.py`.

## Main experiment

For one dataset and horizon:

```bash
python scripts/run_main.py \
  --config configs/etth1.yaml \
  --data-root /local/ETTh1/H192 \
  --output-dir /local/outputs/ETTh1-H192
```

The script trains Stage 1 on `train/`, selects its checkpoint on `val/`, and
starts Stage 2 only when Stage 1 improves both validation MSE and MAE over the
valid endpoint anchor. Stage 2 is retained only when it also improves both
validation metrics. The test split is evaluated once at the end.

For the complete four-dataset/four-horizon main suite, use
`scripts/run_suite.py` and `configs/suite.yaml`.

## Metrics and output policy

Metrics are computed by `src/farr/metrics.py` at runtime. Standardized-space
MSE/MAE is the default; raw-space metrics require the user-provided scaler.
The release tree contains no scores, predictions, labels, logs, checkpoints,
or result tables. The user's output directory may contain these runtime files.
