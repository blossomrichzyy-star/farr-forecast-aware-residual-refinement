from collections import defaultdict
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Compute equal-horizon dataset means from user-generated outputs.")
    parser.add_argument("--input", required=True, help="A suite.json produced locally")
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in records:
        groups[row["dataset"]].append(row)
    result = {}
    for dataset, rows in groups.items():
        if sorted(row["horizon"] for row in rows) != [96, 192, 336, 720]:
            raise ValueError("each dataset requires exactly four distinct horizons")
        result[dataset] = {key: sum(row["test"][key] for row in rows)/4
                           for key in ("mse", "mae")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
