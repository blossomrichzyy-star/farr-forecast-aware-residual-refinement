"""Framework-neutral interface for a user's frozen endpoint forecaster."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

import numpy as np


class FrozenBackbone(Protocol):
    def predict(self, history: np.ndarray, condition: str) -> np.ndarray:
        """Return a [N, H, C] forecast for ``valid`` or ``null``."""


def load_backbone(module_name: str, class_name: str, **kwargs: Any) -> FrozenBackbone:
    """Load a user-provided backbone adapter without hard-coded private paths."""
    module = import_module(module_name)
    factory = getattr(module, class_name)
    provider = factory(**kwargs)
    if not hasattr(provider, "predict"):
        raise TypeError(f"{module_name}.{class_name} must implement predict(history, condition)")
    return provider
