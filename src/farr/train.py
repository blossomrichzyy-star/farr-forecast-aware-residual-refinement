"""Paper Section 2.5: strict joint validation selection and valid fallback."""

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .farr import make_model
from .features import build_features
from .metrics import ErrorAccumulator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def strictly_better(candidate, reference):
    return candidate["mse"] < reference["mse"] and candidate["mae"] < reference["mae"]


def loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


def forward(model, batch, protocol, stage):
    if stage == "valid":
        return batch["y_valid"], None, batch["y_valid"]
    effective = (model.effective_valid(batch["y_valid"], batch["y_null"])
                 if stage == "stage2" else batch["y_valid"])
    features, _ = build_features(batch["history"], batch["y_null"], effective,
                                 protocol.history_stat_length, protocol.recurrence_window)
    # Rebuild all endpoint-derived features from the Stage-2 effective anchor.
    return model(features, batch["y_valid"], batch["y_null"],
                 use_adjustment=stage == "stage2")


def loss_terms(prediction, residual, effective, target, protocol):
    # effective is intentionally not detached: Stage-2 residual supervision
    # must also differentiate through the adjusted-anchor target.
    forecast = nn.functional.smooth_l1_loss(prediction, target,
                                           beta=protocol.smooth_l1_transition)
    fit = nn.functional.smooth_l1_loss(residual, target-effective,
                                      beta=protocol.smooth_l1_transition)
    smooth = (residual[:, 1:]-residual[:, :-1]).square().mean()
    norm = residual.square().mean()
    return forecast + protocol.residual_weight*fit + protocol.smoothness_weight*smooth + protocol.residual_l2_weight*norm


def evaluate(model, dataset, protocol, stage, device):
    model.eval()
    accumulator = ErrorAccumulator()
    with torch.inference_mode():
        for batch in loader(dataset, protocol.eval_batch_size):
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction, _, _ = forward(model, batch, protocol, stage)
            accumulator.update(prediction, batch["y_true"])
    return accumulator.result()


def _train_phase(model, train_set, val_set, protocol, device, stage,
                 eligibility_reference=None):
    adjustment_enabled = stage == "stage2"
    for parameter in model.adjustment.parameters():
        parameter.requires_grad_(adjustment_enabled)
    parameters = [p for p in model.parameters() if p.requires_grad]
    learning_rate = (protocol.learning_rate_stage2 if adjustment_enabled
                     else protocol.learning_rate_stage1)
    epochs = protocol.epochs_stage2 if adjustment_enabled else protocol.epochs_stage1
    patience = protocol.patience_stage2 if adjustment_enabled else protocol.patience_stage1
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate,
                                  weight_decay=protocol.weight_decay)
    best_state = best_metrics = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for batch in loader(train_set, protocol.batch_size, shuffle=True):
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction, residual, effective = forward(model, batch, protocol, stage)
            loss = loss_terms(prediction, residual, effective, batch["y_true"], protocol)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, protocol.gradient_clip)
            optimizer.step()
        metrics = evaluate(model, val_set, protocol, stage, device)
        eligible = (eligibility_reference is None
                    or strictly_better(metrics, eligibility_reference))
        if eligible and (best_metrics is None or strictly_better(metrics, best_metrics)):
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        if eligibility_reference is not None:
            return None, None
        raise RuntimeError("no finite validation candidate was produced")
    model.load_state_dict(best_state)
    return best_state, best_metrics


@dataclass
class TrainingResult:
    model: nn.Module
    stage: str
    validation: dict


def fit(train_set, val_set, protocol, device="auto"):
    """This function has no test-dataset argument or test-file access."""
    protocol.validate()
    seed_everything(protocol.seed)
    device = resolve_device(device)
    model = make_model(protocol).to(device)
    anchor = evaluate(model, val_set, protocol, "valid", device)
    state1, val1 = _train_phase(model, train_set, val_set, protocol, device, "stage1")
    if not strictly_better(val1, anchor):
        return TrainingResult(model, "valid", anchor)
    # Stage 2 starts from the retained Stage-1 state and identity adjustment.
    _, val2 = _train_phase(model, train_set, val_set, protocol, device, "stage2",
                           eligibility_reference=anchor)
    if val2 is not None and strictly_better(val2, anchor):
        return TrainingResult(model, "stage2", val2)
    model.load_state_dict(state1)
    return TrainingResult(model, "stage1", val1)
