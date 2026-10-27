from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from farr.data_loader import prepare_raw_splits
from farr.protocol import DATASETS, HORIZONS, PaperProtocol


def main():
    parser = argparse.ArgumentParser(description="Check a raw CSV and prepare chronological train-scaled windows in memory.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--horizon", required=True, type=int, choices=HORIZONS)
    args = parser.parse_args()
    splits = prepare_raw_splits(args.csv, args.dataset, PaperProtocol(horizon=args.horizon))
    for split, dataset in splits.items():
        print(f"{split}: {len(dataset)} single-channel windows")
    print("Input check passed; no data or scaler was exported.")


if __name__ == "__main__":
    main()
