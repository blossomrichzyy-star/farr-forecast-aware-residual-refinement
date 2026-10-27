from pathlib import Path
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from farr.backbone import endpoint_pass, load_backbone
from farr.data_loader import prepare_raw_splits
from farr.protocol import DATASETS, HORIZONS, PaperProtocol
from farr.train import loader, resolve_device


def export_split(model, dataset, destination, protocol, device):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    count = len(dataset)
    arrays = {}
    for name, length in (("history", protocol.input_length), ("y_true", protocol.horizon),
                         ("y_valid", protocol.horizon), ("y_null", protocol.horizon)):
        arrays[name] = np.lib.format.open_memmap(destination/f"{name}.npy", mode="w+",
                                                dtype=np.float32, shape=(count, length, 1))
    offset = 0
    for batch in loader(dataset, protocol.eval_batch_size):
        size = batch["history"].shape[0]
        for name in ("history", "y_true"):
            arrays[name][offset:offset+size] = batch[name].numpy()
        offset += size
    for condition in ("valid", "null"):
        offset = 0
        batches = loader(dataset, protocol.eval_batch_size)
        for prediction in endpoint_pass(model, batches, condition, protocol, device):
            if prediction.shape[1:] != (protocol.horizon, 1) or not torch.isfinite(prediction).all():
                raise ValueError("invalid endpoint output shape or non-finite prediction")
            size = len(prediction)
            arrays[f"y_{condition}"][offset:offset+size] = prediction.numpy()
            offset += size
        if offset != count:
            raise RuntimeError("endpoint pass did not preserve sample count")
    for array in arrays.values():
        array.flush()


def main():
    parser = argparse.ArgumentParser(description="Generate matched endpoints on the user's machine.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--horizon", type=int, required=True, choices=HORIZONS)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--factory", required=True, help="Local upstream loader: module:function")
    parser.add_argument("--backbone-checkpoint", required=True, help="User-local frozen predictor weights")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    protocol = PaperProtocol(horizon=args.horizon)
    protocol.validate()
    device = resolve_device(args.device)
    model = load_backbone(args.factory, args.backbone_checkpoint, args.dataset,
                          protocol, device)
    splits = prepare_raw_splits(args.csv, args.dataset, protocol)
    for split in (("train", "val", "test") if args.split == "all" else (args.split,)):
        export_split(model, splits[split], Path(args.output_dir)/split,
                      protocol, device)
        print(f"Completed {split} endpoint generation.")


if __name__ == "__main__":
    main()
