#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to a local endpoint bundle}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a local FARR checkpoint}"
PHASE="${PHASE:-stage1}"

python scripts/evaluate.py --config "$CONFIG" --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" --phase "$PHASE" "$@"
