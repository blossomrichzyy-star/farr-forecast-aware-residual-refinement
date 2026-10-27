"""Streaming standardized-space MSE and MAE; no stored benchmark values."""

import torch


class ErrorAccumulator:
    def __init__(self):
        self.square_sum = self.absolute_sum = 0.0
        self.count = 0

    def update(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        error = prediction.detach().to(device="cpu", dtype=torch.float64)
        error = error - target.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(error).all():
            raise ValueError("prediction or target contains non-finite values")
        self.square_sum += error.square().sum().item()
        self.absolute_sum += error.abs().sum().item()
        self.count += error.numel()

    def result(self):
        if not self.count:
            raise ValueError("cannot evaluate an empty split")
        return {"mse": self.square_sum/self.count, "mae": self.absolute_sum/self.count}
