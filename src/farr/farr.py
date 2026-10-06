from __future__ import annotations

import torch
from torch import nn


class ResidualRefiner(nn.Module):
    """Shared 18 -> 128 -> 128 -> 1 residual refiner."""

    def __init__(self, hidden_width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features)).squeeze(-1)


class HorizonGate(nn.Module):
    def __init__(self, horizon: int, lambda_max: float, initial_logit: float):
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.logits = nn.Parameter(torch.full((horizon,), float(initial_logit)))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        gate = self.lambda_max * torch.sigmoid(self.logits)
        return residual * gate.view(1, -1, 1)


class ForecastAdjustment(nn.Module):
    def __init__(self, horizon: int, scale: float = 0.10, offset: float = 0.10):
        super().__init__()
        self.scale = float(scale)
        self.offset = float(offset)
        self.s = nn.Parameter(torch.zeros(horizon))
        self.b = nn.Parameter(torch.zeros(horizon))

    def forward(self, y_valid: torch.Tensor, y_null: torch.Tensor) -> torch.Tensor:
        delta = y_valid - y_null
        scale = self.scale * torch.tanh(self.s).view(1, -1, 1)
        offset = self.offset * torch.tanh(self.b).view(1, -1, 1)
        return y_null + (1.0 + scale) * delta + offset


class FARRModel(nn.Module):
    """The main FARR model: refiner, conservative gate, and adjustment."""

    def __init__(self, horizon: int, hidden_width: int, lambda_max: float,
                 initial_gate_logit: float, adjustment_scale: float,
                 adjustment_offset: float):
        super().__init__()
        self.refiner = ResidualRefiner(hidden_width)
        self.gate = HorizonGate(horizon, lambda_max, initial_gate_logit)
        self.adjustment = ForecastAdjustment(horizon, adjustment_scale, adjustment_offset)

    def effective_valid(self, y_valid: torch.Tensor, y_null: torch.Tensor) -> torch.Tensor:
        return self.adjustment(y_valid, y_null)

    def forward(self, features: torch.Tensor, y_valid: torch.Tensor,
                y_null: torch.Tensor, use_adjustment: bool = True):
        batch, horizon, channels, width = features.shape
        residual = self.refiner(features.reshape(batch * horizon * channels, width))
        residual = residual.reshape(batch, horizon, channels)
        correction = self.gate(residual)
        effective = self.effective_valid(y_valid, y_null) if use_adjustment else y_valid
        return effective + correction, residual, effective


def make_model(config) -> FARRModel:
    return FARRModel(
        horizon=config.horizon,
        hidden_width=config.hidden_width,
        lambda_max=config.lambda_max,
        initial_gate_logit=config.initial_gate_logit,
        adjustment_scale=config.adjustment_scale,
        adjustment_offset=config.adjustment_offset,
    )
