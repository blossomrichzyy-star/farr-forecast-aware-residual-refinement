#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/suite.yaml}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to a local dataset/horizon endpoint root}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/suite-main}"

python scripts/run_suite.py --config "$CONFIG" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" "$@"
