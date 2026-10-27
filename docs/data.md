# Data preparation

The raw CSV has a timestamp first column and numerical channel columns in
their original order. `prepare_raw_splits` fits scikit-learn StandardScaler on
the training region only. ETTh1 uses fixed 12/4/4 blocks of 30*24 observations.
The other datasets use chronological 70% train, the remaining middle 10%
validation, and final 20% test. Validation and test history may use preceding
observations, but future targets stay in the assigned split. Missing and
non-finite values are rejected.

Each sample is a complete single-channel input of length 672 with the next H
values as target. Endpoint generation preserves channel-major order and uses
identical histories for the valid/null passes.

Already-generated endpoint bundles must use the same training-fitted scale
for histories, endpoints, and targets. Legacy endpoint names ending in
`_normalized.npy` are accepted. Model selection never opens test files.
