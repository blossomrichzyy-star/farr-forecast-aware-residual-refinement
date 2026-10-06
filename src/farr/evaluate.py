from __future__ import annotations

from pathlib import Path

import torch

from .config import ExperimentConfig
from .data_loader import EndpointBundle
from .farr import FARRModel, make_model
from .metrics import metric_dict
from .train import TrainedRun, evaluate_test, resolve_device


def load_checkpoint(checkpoint: str | Path, config: ExperimentConfig,
                    device: str | None = None) -> FARRModel:
    target = resolve_device(device or config.device)
    model = make_model(config).to(target)
    state = torch.load(checkpoint, map_location=target, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_checkpoint(bundle: EndpointBundle, config: ExperimentConfig,
                        checkpoint: str | Path, phase: str = "stage1",
                        device: str | None = None) -> dict[str, float]:
    model = load_checkpoint(checkpoint, config, device)
    return evaluate_test(TrainedRun(model, phase, {}), bundle, config)


def evaluate_anchor(bundle: EndpointBundle, config: ExperimentConfig) -> dict[str, float]:
    return metric_dict(bundle.test.y_valid, bundle.test.y_true,
                       config.metric_space, bundle.scaler)
