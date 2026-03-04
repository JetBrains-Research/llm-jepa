#!/bin/bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-datasets/mbpp_train.jsonl}
TEST_FILE=${TEST_FILE:-datasets/mbpp_test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./fine-tuned-mbpp}
SEED=${SEED:-82}

source .venv/bin/activate

torchrun --nproc_per_node="$NGPUS" --master_port=29502 finetune.py \
  --train_file "$TRAIN_FILE" \
  --output_dir="$OUTPUT_DIR" \
  --num_epochs=5 --finetune_seed="$SEED" --regular \
  --model_name="$MODEL_NAME" --learning_rate=2e-5 \
  --lora --lora_rank=256 --batch_size=8 --grad_accum=8 \
  --eval_accuracy --max_eval_samples=50 --max_new_tokens_eval=512 \
  --wandb --no_save

python evaluate_mbpp_pass1.py \
  --model_name "$OUTPUT_DIR" \
  --original_model_name "$MODEL_NAME" \
  --input_file "$TEST_FILE" \
  --output_file "$OUTPUT_DIR/mbpp_pass1_results.jsonl" \
  --summary_file "$OUTPUT_DIR/mbpp_pass1_summary.json" \
  --max_new_tokens 512 \
  --timeout_sec 3

echo "MBPP pass@1 summary written to: $OUTPUT_DIR/mbpp_pass1_summary.json"
