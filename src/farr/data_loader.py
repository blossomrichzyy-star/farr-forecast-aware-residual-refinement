from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    def inverse(self, values):
        return values * self.scale + self.mean


def _load_array(folder: Path, name: str) -> np.ndarray:
    for candidate in (folder / f"{name}.npy", folder / f"{name}_normalized.npy"):
        if candidate.exists():
            return np.load(candidate, allow_pickle=False)
    raise FileNotFoundError(f"Missing {name}.npy in {folder}")


def _as_float32(array: np.ndarray, name: str) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    return np.asarray(array, dtype=np.float32)


class EndpointDataset(Dataset):
    """One split of matched history, valid endpoint, null endpoint, and target."""

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Split folder does not exist: {folder}")
        self.history = _as_float32(_load_array(folder, "history"), "history")
        self.y_valid = _as_float32(_load_array(folder, "y_valid"), "y_valid")
        self.y_null = _as_float32(_load_array(folder, "y_null"), "y_null")
        self.y_true = _as_float32(_load_array(folder, "y_true"), "y_true")
        if self.history.ndim == 2:
            self.history = self.history[..., None]
        for name in ("y_valid", "y_null", "y_true"):
            value = getattr(self, name)
            if value.ndim == 2:
                setattr(self, name, value[..., None])
        if len({self.history.shape[0], self.y_valid.shape[0], self.y_null.shape[0], self.y_true.shape[0]}) != 1:
            raise ValueError("all endpoint arrays must have the same sample count")
        if self.y_valid.shape != self.y_null.shape or self.y_valid.shape != self.y_true.shape:
            raise ValueError("y_valid, y_null, and y_true must have the same shape")
        self.n = self.history.shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: torch.from_numpy(getattr(self, name)[index])
                for name in ("history", "y_valid", "y_null", "y_true")}


@dataclass
class EndpointBundle:
    train: EndpointDataset
    val: EndpointDataset
    test: EndpointDataset
    scaler: Scaler | None = None


def load_scaler(data_root: str | Path) -> Scaler | None:
    path = Path(data_root) / "scaler.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        if "mean" not in values or "scale" not in values:
            raise ValueError("scaler.npz must contain mean and scale")
        return Scaler(values["mean"].astype(np.float32), values["scale"].astype(np.float32))


def load_bundle(data_root: str | Path) -> EndpointBundle:
    root = Path(data_root)
    bundle = EndpointBundle(
        train=EndpointDataset(root / "train"),
        val=EndpointDataset(root / "val"),
        test=EndpointDataset(root / "test"),
        scaler=load_scaler(root),
    )
    if bundle.train.history.shape[1] != bundle.val.history.shape[1]:
        raise ValueError("train and validation input lengths differ")
    return bundle
