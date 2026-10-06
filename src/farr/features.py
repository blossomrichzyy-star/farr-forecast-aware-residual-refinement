from __future__ import annotations

import torch


def _safe_std(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    return x.std(dim=dim, keepdim=keepdim, unbiased=False).clamp_min(1e-4)


def build_features(history: torch.Tensor, y_null: torch.Tensor, y_valid: torch.Tensor,
                   stat_length: int = 168) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the paper's 18 deterministic per-step features.

    The returned tensor has shape [B, H, C, 18]. The second return value is
    the raw endpoint disagreement, useful for residual diagnostics.
    """
    if history.ndim != 3 or y_null.ndim != 3 or y_valid.ndim != 3:
        raise ValueError("history and endpoint tensors must be [B, T, C]")
    b, _, c = history.shape
    h = y_valid.shape[1]
    if y_null.shape != y_valid.shape:
        raise ValueError("valid and null endpoint shapes must match")
    window = history[:, -min(stat_length, history.shape[1]):]
    latest = window[:, -1:]
    mean = window.mean(dim=1, keepdim=True)
    scale = _safe_std(window, dim=1)
    slope = (window[:, -1:] - window[:, :1]) / max(window.shape[1] - 1, 1)
    diff = window[:, 1:] - window[:, :-1]
    diff_scale = _safe_std(diff, dim=1)
    volatility = diff.abs().mean(dim=1, keepdim=True)
    if window.shape[1] >= 48:
        recent = window[:, -24:]
        previous = window[:, -48:-24]
    else:
        half = max(window.shape[1] // 2, 1)
        recent = window[:, -half:]
        previous = window[:, -2 * half:-half]
    # Compare the two windows channel-wise, preserving the 24-point motif
    # rather than collapsing each window to one mean value.
    cosine = torch.nn.functional.cosine_similarity(
        recent.transpose(1, 2), previous.transpose(1, 2), dim=-1
    ).view(b, 1, c)
    ambiguity = (volatility / scale + 1.0 - cosine).clamp(0.0, 10.0)
    delta = y_valid - y_null
    d1 = torch.cat([torch.zeros_like(delta[:, :1]), delta[:, 1:] - delta[:, :-1]], dim=1)
    delta_scale = delta / scale
    smoothness = (d1.abs() / (delta.abs() + 1e-3)).clamp(0.0, 10.0)
    relative = delta.abs() / (y_null.abs() + 1e-3)
    sign_consistency = torch.sign(delta).mean(dim=1, keepdim=True).expand(-1, h, -1)
    horizon = torch.linspace(0.0, 1.0, h, device=history.device, dtype=history.dtype).view(1, h, 1).expand(b, -1, c)

    repeat = lambda value: value.expand(-1, h, -1)
    features = torch.stack([
        repeat(latest), repeat(mean), repeat(scale), repeat(slope), repeat(diff_scale),
        repeat(volatility), repeat(cosine), repeat(ambiguity), y_null, y_valid,
        delta, delta.abs(), delta_scale, d1, smoothness, relative, sign_consistency, horizon,
    ], dim=-1)
    return features, delta


FEATURE_NAMES = [
    "latest", "mean", "scale", "slope", "difference_scale", "local_volatility",
    "motif_similarity", "ambiguity", "null_endpoint", "valid_endpoint", "disagreement",
    "absolute_disagreement", "normalized_disagreement", "disagreement_difference",
    "disagreement_smoothness", "relative_displacement", "sign_consistency", "horizon",
]
