from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from farr.config import load_config
from farr.data_loader import load_bundle
from farr.train import evaluate_test, train_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FARR main experiment over datasets and horizons.")
    parser.add_argument("--config", default="configs/suite.yaml")
    parser.add_argument("--data-root", required=True,
                        help="Contains DATASET/H96, DATASET/H192, ... endpoint bundles")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec_path = Path(args.config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    base_ref = spec.get("base", "default.yaml")
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = spec_path.parent / base_path
    base = load_config(base_path)
    root = Path(args.data_root)
    output = Path(args.output_dir)
    records = []
    for dataset in spec.get("datasets", []):
        for horizon in spec.get("horizons", []):
            data_path = root / dataset / f"H{horizon}"
            config = base.with_updates(horizon=int(horizon))
            bundle = load_bundle(data_path)
            run = train_main(bundle, config, output / dataset / f"H{horizon}")
            records.append({"dataset": dataset, "horizon": horizon, "phase": run.phase,
                            "validation": run.validation,
                            "test": evaluate_test(run, bundle, config)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "suite_main.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
