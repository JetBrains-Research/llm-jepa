#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE=${MODE:-generated}
MODEL_NAME=${MODEL_NAME:-./fine-tuned-codeforces-regular}
ORIGINAL_MODEL_NAME=${ORIGINAL_MODEL_NAME:-Qwen/Qwen3-8B}
INPUT_FILE=${INPUT_FILE:-datasets/code_contests_test.jsonl}
OUTPUT_FILE=${OUTPUT_FILE:-fine-tuned-codeforces-regular/code_contests_pass1_results.jsonl}
SUMMARY_FILE=${SUMMARY_FILE:-fine-tuned-codeforces-regular/code_contests_pass1_summary.json}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TP_SIZE=${TP_SIZE:-1}
TIMEOUT_SEC=${TIMEOUT_SEC:-5}
MEMORY_LIMIT_MB=${MEMORY_LIMIT_MB:-0}
COMPARE_MODE=${COMPARE_MODE:-tokens}
MAX_TESTS_PER_FIELD=${MAX_TESTS_PER_FIELD:-0}
TEST_FIELDS=${TEST_FIELDS:-"public_tests private_tests generated_tests"}
read -r -a TEST_FIELDS_ARR <<< "$TEST_FIELDS"

mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$SUMMARY_FILE")"

CMD=(uv run python "$REPO_ROOT/evaluate_code_contests_pass1.py"
  --mode "$MODE"
  --input_file "$INPUT_FILE"
  --output_file "$OUTPUT_FILE"
  --summary_file "$SUMMARY_FILE"
  --max_examples "$MAX_EXAMPLES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --tp_size "$TP_SIZE"
  --timeout_sec "$TIMEOUT_SEC"
  --memory_limit_mb "$MEMORY_LIMIT_MB"
  --compare_mode "$COMPARE_MODE"
  --max_tests_per_field "$MAX_TESTS_PER_FIELD"
  --test_fields "${TEST_FIELDS_ARR[@]}"
)

if [[ "$MODE" == "generated" ]]; then
  CMD+=(--model_name "$MODEL_NAME" --original_model_name "$ORIGINAL_MODEL_NAME")
fi

if [[ "${DETACH:-0}" == "1" ]]; then
  LOG_DIR=${LOG_DIR:-logs}
  mkdir -p "$LOG_DIR"
  TS=$(date +%Y%m%d_%H%M%S)
  LOG_FILE=${LOG_FILE:-"$LOG_DIR/code_contests_pass1_eval_${MODE}_${TS}.log"}
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "Started detached evaluation."
  echo "PID: $PID"
  echo "Log: $LOG_FILE"
else
  "${CMD[@]}"
fi
