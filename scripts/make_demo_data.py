from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_split(folder: Path, n: int, length: int, horizon: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(n, length, 1)).astype(np.float32)
    trend = np.linspace(0.0, 0.3, horizon, dtype=np.float32)[None, :, None]
    noise = rng.normal(scale=0.15, size=(n, horizon, 1)).astype(np.float32)
    y_true = trend + noise
    y_null = np.repeat(history[:, -1:, :], horizon, axis=1)
    y_valid = y_null + 0.3 * trend
    for name, value in {"history": history, "y_valid": y_valid, "y_null": y_null, "y_true": y_true}.items():
        np.save(folder / f"{name}.npy", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic smoke-test endpoint data.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--input-length", type=int, default=672)
    args = parser.parse_args()
    root = Path(args.output)
    for index, (name, n) in enumerate((("train", 16), ("val", 8), ("test", 8))):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        make_split(folder, n, args.input_length, args.horizon, 2021 + index)
    np.savez(root / "scaler.npz", mean=np.zeros(1, dtype=np.float32), scale=np.ones(1, dtype=np.float32))
    print(f"Created synthetic endpoint bundle at {root}")


if __name__ == "__main__":
    main()
