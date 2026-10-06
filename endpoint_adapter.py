"""Small adapter interface for a frozen endpoint forecaster.

The project deliberately does not vendor a backbone checkpoint. Implement
`FrozenBackbone.predict` in the repository that owns the compatible
forecaster, then call `save_split` to create the public bundle format.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def save_split(folder: str | Path, history: np.ndarray, y_valid: np.ndarray,
               y_null: np.ndarray, y_true: np.ndarray) -> None:
    """Save one matched endpoint split after validating its shapes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {"history": history, "y_valid": y_valid, "y_null": y_null, "y_true": y_true}
    n = {np.asarray(value).shape[0] for value in arrays.values()}
    if len(n) != 1:
        raise ValueError("history and endpoint arrays must contain the same number of samples")
    if np.asarray(y_valid).shape != np.asarray(y_null).shape or np.asarray(y_valid).shape != np.asarray(y_true).shape:
        raise ValueError("endpoint forecasts and target must have identical shapes")
    for name, value in arrays.items():
        np.save(folder / f"{name}.npy", np.asarray(value, dtype=np.float32))
