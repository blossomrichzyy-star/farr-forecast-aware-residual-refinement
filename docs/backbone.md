# Frozen forecaster and endpoint generation

Endpoint generation requires a trained SE-LLM forecasting checkpoint and
its matching architecture. A base Qwen language model must first be adapted
and trained for the forecasting task.

A local function named by `--factory module:function` receives keyword
arguments `checkpoint_path`, `dataset`, `horizon`, `device`, `backbone_name`,
`input_length`, `label_length`, and `token_length`, and returns the fully
loaded torch model. Model settings are passed directly from `PaperProtocol`:
Qwen2.5-0.5B-Instruct, input length 672, label length 576, and token length 96.
Load the checkpoint strictly with the matching architecture in the upstream
SE-LLM environment. Endpoint generation uses the protocol evaluation batch
size of 8192 complete single-channel windows.

The adapter uses the Qwen-based SE-LLM interface:

- `model.fuser.attention` produces the aligned semantic tensor;
- model forward accepts `(history, x_mark, None, y_mark)`;
- this interface ignores time-marker arguments;
- predictions use recursive 96-point output tokens.

For the null condition, a temporary forward hook replaces only the alignment
attention output with `zeros_like`. Temporal history, text embeddings, noise
decomposition, and both existing fusion branches remain unchanged. The hook
is removed after the pass. Seed 2021 is reset independently before each
complete ordered condition pass, with deterministic cuDNN settings and
`shuffle=False`, `drop_last=False`.

For a forecaster that uses time markers, adapt the rollout to pass its actual
timestamp features while preserving the endpoint order and null intervention.
