from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data_loader import EndpointBundle, EndpointDataset
from .features import build_features
from .metrics import metric_dict, merge_metric_sums
from .farr import FARRModel, make_model


@dataclass
class TrainedRun:
    model: FARRModel
    phase: str
    validation: dict[str, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(dataset: EndpointDataset, batch_size: int, config: ExperimentConfig,
            shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=config.num_workers,
                      pin_memory=torch.cuda.is_available())


def _features(batch: dict[str, torch.Tensor], config: ExperimentConfig,
              device: torch.device, valid: torch.Tensor | None = None) -> torch.Tensor:
    history = batch["history"].to(device)
    y_null = batch["y_null"].to(device)
    y_valid = batch["y_valid"].to(device) if valid is None else valid
    features, _ = build_features(history, y_null, y_valid, config.history_stat_length)
    return features


def _loss(final: torch.Tensor, residual: torch.Tensor, effective: torch.Tensor,
          target: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    forecast = nn.functional.smooth_l1_loss(final, target)
    residual_fit = nn.functional.smooth_l1_loss(residual, target - effective)
    smooth = ((residual[:, 1:] - residual[:, :-1]).pow(2).mean()
              if residual.shape[1] > 1 else residual.new_zeros(()))
    return (forecast + config.residual_weight * residual_fit
            + config.smoothness_weight * smooth
            + config.residual_l2_weight * residual.pow(2).mean())


def _evaluate(model: FARRModel, dataset: EndpointDataset,
              config: ExperimentConfig, device: torch.device,
              scaler, use_adjustment: bool) -> dict[str, float]:
    model.eval()
    parts: list[tuple[dict[str, float], int]] = []
    with torch.no_grad():
        for batch in _loader(dataset, config.eval_batch_size, config, shuffle=False):
            target = batch["y_true"].to(device)
            y_valid = batch["y_valid"].to(device)
            y_null = batch["y_null"].to(device)
            effective = model.effective_valid(y_valid, y_null) if use_adjustment else y_valid
            features = _features(batch, config, device, effective if use_adjustment else None)
            final, _, _ = model(features, y_valid, y_null, use_adjustment=use_adjustment)
            values = metric_dict(final.cpu().numpy(), target.cpu().numpy(),
                                 config.metric_space, scaler)
            parts.append((values, target.shape[0]))
    return merge_metric_sums(parts)


def _anchor(dataset: EndpointDataset, config: ExperimentConfig, scaler) -> dict[str, float]:
    return metric_dict(dataset.y_valid, dataset.y_true, config.metric_space, scaler)


def _train_phase(model: FARRModel, train_set: EndpointDataset,
                 val_set: EndpointDataset, config: ExperimentConfig,
                 device: torch.device, scaler, use_adjustment: bool,
                 learning_rate: float, epochs: int,
                 patience: int) -> tuple[dict[str, Any], dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=config.weight_decay)
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        for batch in _loader(train_set, config.batch_size, config, shuffle=True):
            target = batch["y_true"].to(device)
            y_valid = batch["y_valid"].to(device)
            y_null = batch["y_null"].to(device)
            effective = model.effective_valid(y_valid, y_null) if use_adjustment else y_valid
            features = _features(batch, config, device, effective if use_adjustment else None)
            final, residual, effective = model(features, y_valid, y_null,
                                               use_adjustment=use_adjustment)
            loss = _loss(final, residual, effective, target, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        metrics = _evaluate(model, val_set, config, device, scaler, use_adjustment)
        score = metrics["mse"] + metrics["mae"]
        if score < best_score:
            best_score = score
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return best_state, best_metrics  # type: ignore[return-value]


def train_main(bundle: EndpointBundle, config: ExperimentConfig,
               output_dir: str | Path | None = None) -> TrainedRun:
    """Train the full FARR main experiment using train/validation only."""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if bundle.train.y_true.shape[1] != config.horizon:
        raise ValueError(f"config horizon={config.horizon} but data horizon={bundle.train.y_true.shape[1]}")
    model = make_model(config).to(device)
    state1, val1 = _train_phase(
        model, bundle.train, bundle.val, config, device, bundle.scaler,
        use_adjustment=False, learning_rate=config.learning_rate_stage1,
        epochs=config.epochs_stage1, patience=config.patience_stage1,
    )
    phase = "stage1"
    best_val = val1
    if config.run_stage2:
        anchor = _anchor(bundle.val, config, bundle.scaler)
        if val1["mse"] < anchor["mse"] and val1["mae"] < anchor["mae"]:
            _, val2 = _train_phase(
                model, bundle.train, bundle.val, config, device, bundle.scaler,
                use_adjustment=True, learning_rate=config.learning_rate_stage2,
                epochs=config.epochs_stage2, patience=config.patience_stage2,
            )
            if val2["mse"] < anchor["mse"] and val2["mae"] < anchor["mae"]:
                phase, best_val = "stage2", val2
            else:
                model.load_state_dict(state1)
        else:
            model.load_state_dict(state1)
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "farr.pt")
        (path / "validation.json").write_text(json.dumps({
            "phase": phase, "validation": best_val,
        }, indent=2), encoding="utf-8")
    return TrainedRun(model, phase, best_val)


def evaluate_test(run: TrainedRun, bundle: EndpointBundle,
                  config: ExperimentConfig) -> dict[str, float]:
    device = next(run.model.parameters()).device
    return _evaluate(run.model, bundle.test, config, device, bundle.scaler,
                     use_adjustment=run.phase == "stage2")
