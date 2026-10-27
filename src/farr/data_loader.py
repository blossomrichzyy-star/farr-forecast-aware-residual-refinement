"""Memory-mapped endpoint input and train-fitted raw benchmark windows."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _array(root, name):
    for suffix in ("", "_normalized"):
        path = root / f"{name}{suffix}.npy"
        if path.is_file():
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            if array.ndim == 2:
                array = array[..., None]
            if array.ndim != 3 or not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"{name} must be a floating [N,T,C] array")
            return array
    raise FileNotFoundError(f"Missing endpoint input: {name}.npy")


class EndpointDataset(Dataset):
    """One sample is one complete single-channel window, as in Table 3.

    Multichannel bundles are indexed in channel-major order without copying
    or materializing their full arrays in RAM.
    """

    def __init__(self, root, protocol):
        root = Path(root)
        self.arrays = {name: _array(root, name) for name in
                       ("history", "y_valid", "y_null", "y_true")}
        history, target = self.arrays["history"], self.arrays["y_true"]
        if history.shape[1] != protocol.input_length:
            raise ValueError("history length must match the paper input length")
        if target.shape[1] != protocol.horizon:
            raise ValueError("endpoint horizon does not match the requested run")
        if any(self.arrays[name].shape != target.shape for name in ("y_valid", "y_null")):
            raise ValueError("target and endpoint shapes must be identical")
        if history.shape[0] != target.shape[0] or history.shape[2] != target.shape[2]:
            raise ValueError("history and forecasts must share sample/channel order")
        self.origins, self.channels = target.shape[0], target.shape[2]
        if self.origins == 0 or self.channels == 0:
            raise ValueError("endpoint split is empty")

    def __len__(self):
        return self.origins * self.channels

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        channel, origin = divmod(index, self.origins)
        return {name: torch.from_numpy(np.array(value[origin, :, channel:channel+1],
                                                dtype=np.float32, copy=True))
                for name, value in self.arrays.items()}


class RawWindows(Dataset):
    def __init__(self, standardized, start, stop, input_length, horizon):
        self.values = standardized[start:stop]
        self.input_length, self.horizon = input_length, horizon
        self.origins = len(self.values) - input_length - horizon + 1
        if self.origins <= 0:
            raise ValueError("split is too short for the input length and horizon")
        self.channels = self.values.shape[1]

    def __len__(self):
        return self.origins * self.channels

    def __getitem__(self, index):
        channel, origin = divmod(index, self.origins)
        end = origin + self.input_length
        return {
            "history": torch.from_numpy(self.values[origin:end, channel:channel+1].copy()),
            "y_true": torch.from_numpy(self.values[end:end+self.horizon, channel:channel+1].copy()),
        }


def prepare_raw_splits(csv_path, dataset, protocol):
    """Build chronological windows using a StandardScaler fitted on train only.

    The first CSV column is the timestamp; remaining columns are channels.
    No fitted scaler or sample is written by this function.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from .protocol import DATASETS

    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError("CSV needs a timestamp column and at least one channel")
    pd.to_datetime(frame.iloc[:, 0], errors="raise")
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("CSV contains missing or non-finite channel values")
    length = protocol.input_length
    if dataset == "ETTh1":
        train_end, val_end, test_end = 12*30*24, 16*30*24, 20*30*24
        if len(values) < test_end:
            raise ValueError("ETTh1 requires the complete fixed benchmark split")
    else:
        train_end = int(len(values) * 0.7)
        test_size = int(len(values) * 0.2)
        val_end, test_end = len(values) - test_size, len(values)
    standardized = StandardScaler().fit(values[:train_end]).transform(values).astype(np.float32)
    borders = {"train": (0, train_end), "val": (train_end-length, val_end),
               "test": (val_end-length, test_end)}
    return {split: RawWindows(standardized, start, stop, length, protocol.horizon)
            for split, (start, stop) in borders.items()}
