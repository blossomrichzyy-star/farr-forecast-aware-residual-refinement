from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farr.data_loader import load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local endpoint bundle before training.")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.data_root)
    print(f"train: N={len(bundle.train)}, history={bundle.train.history.shape}, target={bundle.train.y_true.shape}")
    print(f"val:   N={len(bundle.val)}, history={bundle.val.history.shape}, target={bundle.val.y_true.shape}")
    print(f"test:  N={len(bundle.test)}, history={bundle.test.history.shape}, target={bundle.test.y_true.shape}")
    print(f"scaler: {'present' if bundle.scaler is not None else 'not provided (standardized metrics only)'}")


if __name__ == "__main__":
    main()
