#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to a local endpoint bundle}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/main}"

python scripts/run_main.py --config "$CONFIG" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" "$@"
