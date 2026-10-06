from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize results produced locally by a FARR run.")
    parser.add_argument("--input", required=True, help="main.json or suite_main.json generated locally")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    metrics = [row.get("test", {}) for row in rows if row.get("test")]
    if not metrics:
        raise ValueError("No test metrics found in the supplied runtime output")
    summary = {key: sum(float(item[key]) for item in metrics) / len(metrics)
               for key in ("mse", "mae")}
    print(json.dumps({"count": len(metrics), "mean_test": summary}, indent=2))


if __name__ == "__main__":
    main()
