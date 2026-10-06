from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farr.config import load_config
from farr.data_loader import load_bundle
from farr.evaluate import evaluate_anchor, evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a user-provided FARR checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", choices=["stage1", "stage2"], default="stage1")
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
    output = {
        "experiment": "full_farr",
        "phase": args.phase,
        "local_valid_anchor": evaluate_anchor(bundle, config),
        "test": evaluate_checkpoint(bundle, config, args.checkpoint, args.phase, args.device),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
