from __future__ import annotations

import torch


def _safe_std(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    return x.std(dim=dim, keepdim=keepdim, unbiased=False).clamp_min(1e-4)


def build_features(history: torch.Tensor, y_null: torch.Tensor, y_valid: torch.Tensor,
                   stat_length: int, recurrence_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the paper's 18 deterministic per-step features.

    The returned tensor has shape [B, H, 18] for single-channel windows.
    Multichannel series are split by the data loader before this function.
    The second return value is
    the raw endpoint disagreement, useful for residual diagnostics.
    """
    if history.ndim != 3 or y_null.ndim != 3 or y_valid.ndim != 3:
        raise ValueError("history and endpoint tensors must be [B, T, C]")
    if history.shape[2] != 1 or y_valid.shape[2] != 1 or y_null.shape[2] != 1:
        raise ValueError("features require complete single-channel windows")
    b, _, c = history.shape
    h = y_valid.shape[1]
    if y_null.shape != y_valid.shape:
        raise ValueError("valid and null endpoint shapes must match")
    if not 0 < 2*recurrence_window <= stat_length <= history.shape[1]:
        raise ValueError("history must contain the requested statistics and recurrence windows")
    window = history[:, -stat_length:]
    latest = window[:, -1:]
    mean = window.mean(dim=1, keepdim=True)
    scale = _safe_std(window, dim=1)
    slope = (window[:, -1:] - window[:, :1]) / max(window.shape[1] - 1, 1)
    diff = window[:, 1:] - window[:, :-1]
    diff_scale = _safe_std(diff, dim=1)
    volatility = diff.abs().mean(dim=1, keepdim=True)
    recent = window[:, -recurrence_window:]
    previous = window[:, -2*recurrence_window:-recurrence_window]
    # Stabilized cosine similarity between consecutive recurrence windows.
    cosine = ((recent * previous).sum(dim=1, keepdim=True)
              / (recent.norm(dim=1, keepdim=True)
                 * previous.norm(dim=1, keepdim=True) + 1e-5))
    ambiguity = (volatility / scale + 1.0 - cosine).clamp(0.0, 10.0)
    delta = y_valid - y_null
    d1 = torch.cat([torch.zeros_like(delta[:, :1]), delta[:, 1:] - delta[:, :-1]], dim=1)
    delta_scale = delta / scale
    smoothness = (d1.abs() / (delta.abs() + 1e-3)).clamp(0.0, 10.0)
    relative = delta.abs() / (y_null.abs() + 1e-3)
    sign_consistency = torch.sign(delta).mean(dim=1, keepdim=True).abs().expand(-1, h, -1)
    horizon = torch.linspace(0.0, 1.0, h, device=history.device, dtype=history.dtype).view(1, h, 1).expand(b, -1, c)

    repeat = lambda value: value.expand(-1, h, -1)
    features = torch.stack([
        repeat(latest), repeat(mean), repeat(scale), repeat(slope), repeat(diff_scale),
        repeat(volatility), repeat(cosine), repeat(ambiguity), y_null, y_valid,
        delta, delta.abs(), delta_scale, d1, smoothness, relative, sign_consistency, horizon,
    ], dim=-1)
    return features.squeeze(2), delta


FEATURE_NAMES = [
    "latest", "mean", "scale", "slope", "difference_scale", "local_volatility",
    "motif_similarity", "ambiguity", "null_endpoint", "valid_endpoint", "disagreement",
    "absolute_disagreement", "normalized_disagreement", "disagreement_difference",
    "disagreement_smoothness", "relative_displacement", "sign_consistency", "horizon",
]
