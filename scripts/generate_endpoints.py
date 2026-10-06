from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farr.backbone import load_backbone
from farr.endpoint_adapter import save_split
from farr.endpoints import generate_matched_endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched valid/null endpoints with a user backbone adapter.")
    parser.add_argument("--provider-module", required=True)
    parser.add_argument("--provider-class", required=True)
    parser.add_argument("--provider-kwargs", default="{}", help="JSON object passed to the adapter constructor")
    parser.add_argument("--history", required=True, help="Local history .npy file")
    parser.add_argument("--target", required=True, help="Local target .npy file")
    parser.add_argument("--output", required=True, help="Output split directory")
    args = parser.parse_args()
    kwargs = json.loads(args.provider_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--provider-kwargs must be a JSON object")
    provider = load_backbone(args.provider_module, args.provider_class, **kwargs)
    history = np.load(args.history, allow_pickle=False)
    target = np.load(args.target, allow_pickle=False)
    valid, null = generate_matched_endpoints(provider, history)
    save_split(args.output, history, valid, null, target)
    print(f"Saved matched endpoints to {Path(args.output)}")


if __name__ == "__main__":
    main()
