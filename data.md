# Data and endpoint format

The public repository does not contain any dataset or forecast cache. Obtain
each benchmark from its official source and follow its license.

## Bundle layout

Each fixed dataset/horizon bundle has `train/`, `val/`, and `test/` folders.
Each folder contains:

- `history.npy`: `[N, L, C]` historical context;
- `y_valid.npy`: `[N, H, C]` forecast under the standard condition;
- `y_null.npy`: `[N, H, C]` forecast under the fixed null condition;
- `y_true.npy`: `[N, H, C]` future target.

`scaler.npz` is optional and must contain `mean` and `scale` fitted on the
training split only. It is required only for raw-space reporting.

The four data configurations use input length 672 and support horizons 96,
192, 336, and 720. The loader also accepts single-channel arrays without the
final channel dimension and legacy names ending in `_normalized.npy`.

## Required endpoint invariants

The valid and null passes must share the checkpoint, historical input,
preprocessing, sample order, channel order, label length, token length, and
deterministic seed. Only the designated conditioning operation changes. The
null operation must be documented by the backbone adapter; it must not be
silently substituted by a second unrelated forecast.

Run `scripts/prepare_data.py --data-root ...` before training to validate
shapes and split availability without calculating or storing paper scores.
