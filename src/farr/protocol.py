"""Public method constants from Table 3; no paths or external configuration."""

from dataclasses import dataclass

DATASETS = ("ETTh1", "Weather", "Traffic", "ECL")
HORIZONS = (96, 192, 336, 720)


@dataclass(frozen=True)
class PaperProtocol:
    horizon: int = 192
    seed: int = 2021
    input_length: int = 672
    label_length: int = 576
    token_length: int = 96
    history_stat_length: int = 168
    batch_size: int = 8192
    eval_batch_size: int = 8192
    hidden_width: int = 128
    lambda_max: float = 0.20
    initial_gate_logit: float = -6.0
    adjustment_scale: float = 0.10
    adjustment_offset: float = 0.10
    residual_weight: float = 0.10
    smoothness_weight: float = 0.01
    residual_l2_weight: float = 0.01
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    epochs_stage1: int = 10
    epochs_stage2: int = 5
    patience_stage1: int = 2
    patience_stage2: int = 2

    def validate(self):
        if self.horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS}")
        if self.input_length - self.label_length != self.token_length:
            raise ValueError("input/label/token lengths do not agree")
