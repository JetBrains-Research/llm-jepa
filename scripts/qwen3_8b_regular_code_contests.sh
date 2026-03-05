#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WANDB_ENABLED=${WANDB_ENABLED:-1}
TRAIN_FILE=${TRAIN_FILE:-datasets/code_contests_train.jsonl}
EVAL_FILE=${EVAL_FILE:-datasets/code_contests_test.jsonl}

if [[ "${DETACH:-0}" == "1" ]]; then
  LOG_DIR=${LOG_DIR:-logs}
  mkdir -p "$LOG_DIR"
  TS=$(date +%Y%m%d_%H%M%S)
  LOG_FILE=${LOG_FILE:-"$LOG_DIR/qwen3_8b_regular_code_contests_${TS}.log"}

  nohup env \
    WANDB_ENABLED="$WANDB_ENABLED" \
    TRAIN_FILE="$TRAIN_FILE" \
    EVAL_FILE="$EVAL_FILE" \
    uv run bash "$SCRIPT_DIR/qwen3_8b_regular_codeforces.sh" \
    >"$LOG_FILE" 2>&1 &

  PID=$!
  echo "Started detached training."
  echo "PID: $PID"
  echo "Log: $LOG_FILE"
else
  WANDB_ENABLED="$WANDB_ENABLED" \
  TRAIN_FILE="$TRAIN_FILE" \
  EVAL_FILE="$EVAL_FILE" \
  uv run bash "$SCRIPT_DIR/qwen3_8b_regular_codeforces.sh"
fi
