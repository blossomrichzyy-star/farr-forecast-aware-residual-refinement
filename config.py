from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    dataset: str = "unspecified"
    data_root: str = ""
    backbone_path: str = ""
    output_dir: str = "outputs"
    seed: int = 2021
    input_length: int = 672
    history_stat_length: int = 168
    label_length: int = 576
    token_length: int = 96
    horizon: int = 192
    metric_space: str = "standardized"
    device: str = "auto"
    num_workers: int = 0
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
    epochs_stage2: int = 10
    patience_stage1: int = 2
    patience_stage2: int = 2
    run_stage2: bool = True

    def validate(self) -> None:
        if self.metric_space not in {"standardized", "raw"}:
            raise ValueError("metric_space must be 'standardized' or 'raw'")
        if self.lambda_max < 0:
            raise ValueError("lambda_max must be non-negative")
        if self.horizon <= 0 or self.history_stat_length <= 1:
            raise ValueError("horizon and history_stat_length must be positive")

    def with_updates(self, **updates: Any) -> "ExperimentConfig":
        result = replace(self, **updates)
        result.validate()
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return payload


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    payload = _read_yaml(path)
    base_ref = payload.pop("base", None)
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = _read_yaml(base_path)
        base.update(payload)
        payload = base
    allowed = set(ExperimentConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    config = ExperimentConfig(**payload)
    config.validate()
    return config
