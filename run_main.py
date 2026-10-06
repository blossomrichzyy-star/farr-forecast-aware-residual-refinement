from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farr.config import load_config
from farr.data_loader import load_bundle
from farr.metrics import metric_dict, to_jsonable
from farr.train import evaluate_test, train_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FARR main experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-space", choices=["standardized", "raw"])
    parser.add_argument("--device")
    parser.add_argument("--horizon", type=int, choices=[96, 192, 336, 720])
    args = parser.parse_args()
    config = load_config(args.config)
    updates = {}
    if args.metric_space:
        updates["metric_space"] = args.metric_space
    if args.device:
        updates["device"] = args.device
    if args.horizon:
        updates["horizon"] = args.horizon
    if updates:
        config = config.with_updates(**updates)
    bundle = load_bundle(args.data_root)
    run = train_main(bundle, config, args.output_dir)
    test = evaluate_test(run, bundle, config)
    anchor = metric_dict(bundle.test.y_valid, bundle.test.y_true, config.metric_space, bundle.scaler)
    out = {"experiment": "full_farr", "phase": run.phase, "validation": run.validation,
           "local_valid_anchor": anchor, "test": test,
           "config": config.as_dict()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.json").write_text(json.dumps(out, indent=2, default=to_jsonable), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
