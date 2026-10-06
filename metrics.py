from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data_loader import Scaler


def metric_dict(pred: np.ndarray, target: np.ndarray, metric_space: str = "standardized",
                scaler: Scaler | None = None) -> dict[str, float]:
    if metric_space == "raw":
        if scaler is None:
            raise ValueError("raw metrics require data_root/scaler.npz")
        pred = scaler.inverse(pred)
        target = scaler.inverse(target)
    error = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {"mse": float(np.mean(error ** 2)), "mae": float(np.mean(np.abs(error)))}


def merge_metric_sums(parts: list[tuple[dict[str, float], int]]) -> dict[str, float]:
    total = sum(count for _, count in parts)
    if total == 0:
        raise ValueError("cannot aggregate empty metrics")
    return {
        key: float(sum(values[key] * count for values, count in parts) / total)
        for key in ("mse", "mae")
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
