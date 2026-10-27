from pathlib import Path
import argparse
import json

from run_main import DATASETS, HORIZONS, run_one


def main():
    parser = argparse.ArgumentParser(description="Run four datasets by four horizons, with fixed paper settings.")
    parser.add_argument("--data-root", required=True, help="Contains DATASET/H96, DATASET/H192, etc.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    records = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            records.append(run_one(Path(args.data_root)/dataset/f"H{horizon}",
                                    Path(args.output_dir)/dataset/f"H{horizon}",
                                    dataset, horizon, args.device))
    (Path(args.output_dir)/"suite.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Completed main suite; outputs are in the specified local directory.")


if __name__ == "__main__":
    main()
