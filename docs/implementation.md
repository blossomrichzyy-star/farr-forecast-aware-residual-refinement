# Method and training protocol

The parameters in `src/farr/protocol.py` correspond to Table 3.
All main-experiment entry points use this protocol without separate
per-dataset training overrides.

| Paper item | Implementation |
| --- | --- |
| Eqs. (1)-(2), matched endpoints | Identical ordered histories and frozen model; seed reset before each condition pass |
| Historical context | Latest 168 points; population standard deviations floored at 1e-4; slope denominator 167 |
| Recurrence and ambiguity | Latest/preceding 24-point cosine, with 1e-5 denominator stabilization; ambiguity clipped to [0,10] |
| Eqs. (3)-(4), disagreement | First difference starts at zero; smoothness clipped to [0,10]; sign consistency is the absolute mean sign |
| Eq. (5), refiner | Shared 18-to-128-to-128-to-1 MLP, GELU, zero output initialization, tanh |
| Eq. (6), adjustment | Two zero-initialized horizon vectors, bounded scale and offset of 0.10 |
| Eqs. (7)-(8), gate and final forecast | 0.20 sigmoid(logit), initial logit -6; residual added to the effective valid anchor |
| Eqs. (9)-(10), objective | SmoothL1 with transition 1; residual weight 0.10; pre-gate smoothness and L2 weights 0.01 |
| Checkpoint updates | Both validation MSE and MAE must strictly decrease; Stage-2 candidates must first improve both over the valid anchor |
| Stage transition | Stage 2 runs only after Stage 1 improves both metrics over valid; Stage 2 must also pass that anchor comparison |
| Fallback | Retain eligible Stage 1 when Stage 2 fails; otherwise use original valid when Stage 1 is ineligible |
| Training budget | Stage 1/2 maximum epochs 10/5, patience 2/2, AdamW learning rates 1e-3/1e-4 |
| Batch size | 8192 complete single-channel windows for training, evaluation, and endpoint generation |
| Optimizer | AdamW weight decay 1e-4; global gradient clipping 1.0 |
| Frozen forecaster | Qwen2.5-0.5B-Instruct; input/label/token lengths 672/576/96 |
| Random seed | 2021 for training and each complete endpoint pass |
| Metrics | Standardized MSE/MAE; equal-horizon dataset means |

Stage-2 features and the residual target are recomputed from the adjusted
anchor, and remain differentiable through it. The correction bound is
relative to that effective anchor. Stage-2 promotion compares with the valid
anchor, not with Stage 1.

The feature tensor is exactly `[B,H,18]`. Data loading expands each channel
into an individual complete window; the shared refiner returns `[B,H,1]`.
