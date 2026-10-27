from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from farr.data_loader import EndpointDataset
from farr.evaluate import load_checkpoint
from farr.train import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate user-generated weights; restore the selected stage automatically.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result, protocol, device = load_checkpoint(args.checkpoint, args.device)
    test = EndpointDataset(Path(args.data_root)/"test", protocol)
    print(json.dumps({"horizon": protocol.horizon, "selected_stage": result.stage,
                      "metric_space": "standardized",
                      "test": evaluate(result.model, test, protocol, result.stage, device)}, indent=2))


if __name__ == "__main__":
    main()
