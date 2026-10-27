"""Optional user-local checkpoint I/O with automatic stage restoration."""

from pathlib import Path

import torch

from .farr import make_model
from .protocol import PaperProtocol
from .train import TrainingResult, resolve_device


def save_checkpoint(result, protocol, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No data, scaler, predictions, paths, scores, or private configuration.
    torch.save({"format_version": 1, "horizon": protocol.horizon,
                "selected_stage": result.stage,
                "state_dict": {key: value.detach().cpu()
                               for key, value in result.model.state_dict().items()}}, path)


def load_checkpoint(path, device="auto"):
    device = resolve_device(device)
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    stage = payload.get("selected_stage")
    if stage not in {"valid", "stage1", "stage2"}:
        raise ValueError("checkpoint does not identify its selected stage")
    protocol = PaperProtocol(horizon=int(payload["horizon"]))
    protocol.validate()
    model = make_model(protocol).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return TrainingResult(model, stage, {}), protocol, device
