from __future__ import annotations

import numpy as np

from .backbone import FrozenBackbone
from .endpoint_adapter import save_split


def generate_matched_endpoints(provider: FrozenBackbone,
                               history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run valid and null conditions on exactly the same history batch."""
    history = np.asarray(history, dtype=np.float32)
    valid = np.asarray(provider.predict(history, "valid"), dtype=np.float32)
    null = np.asarray(provider.predict(history, "null"), dtype=np.float32)
    if valid.shape != null.shape or valid.shape[0] != history.shape[0]:
        raise ValueError("valid/null endpoint outputs must match each other and history sample count")
    return valid, null


__all__ = ["generate_matched_endpoints", "save_split"]
