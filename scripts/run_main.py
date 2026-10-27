from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from farr.data_loader import EndpointDataset
from farr.evaluate import save_checkpoint
from farr.protocol import DATASETS, HORIZONS, PaperProtocol
from farr.train import evaluate, fit, resolve_device


def run_one(data_root, output_dir, dataset, horizon, device="auto", save_weights=False):
    protocol = PaperProtocol(horizon=horizon)
    protocol.validate()
    root = Path(data_root)
    # No test endpoint is opened until validation-only selection is finished.
    training = EndpointDataset(root/"train", protocol)
    validation = EndpointDataset(root/"val", protocol)
    result = fit(training, validation, protocol, device)
    test = EndpointDataset(root/"test", protocol)
    device = resolve_device(device)
    final_metrics = evaluate(result.model, test, protocol, result.stage, device)
    anchor_metrics = evaluate(result.model, test, protocol, "valid", device)
    report = {"dataset": dataset, "horizon": horizon, "metric_space": "standardized",
              "selected_stage": result.stage, "validation": result.validation,
              "test": final_metrics, "local_valid_anchor": anchor_metrics}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if save_weights:
        save_checkpoint(result, protocol, output/"farr.pt")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run the paper-aligned FARR main method.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--horizon", type=int, required=True, choices=HORIZONS)
    parser.add_argument("--data-root", required=True, help="A user-local endpoint bundle")
    parser.add_argument("--output-dir", required=True, help="User-local runtime output directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-checkpoint", action="store_true", help="Opt in to saving user-generated weights")
    args = parser.parse_args()
    report = run_one(args.data_root, args.output_dir, args.dataset, args.horizon,
                     args.device, args.save_checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
