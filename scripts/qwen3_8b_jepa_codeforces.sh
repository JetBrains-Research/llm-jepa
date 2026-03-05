#!/bin/bash
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  CUDA_VISIBLE_DEVICES="2,3"
elif [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="2,3"
fi
export CUDA_VISIBLE_DEVICES
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-datasets/codeforces_train.jsonl}
EVAL_FILE=${EVAL_FILE:-datasets/codeforces_test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./fine-tuned-codeforces-jepa}
SEED=${SEED:-82}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LORA_RANK=${LORA_RANK:-256}
EVAL_STEPS=${EVAL_STEPS:-50}
LOGGING_STEPS=${LOGGING_STEPS:-1}
LAST_TOKEN=${LAST_TOKEN:--3}
LBD=${LBD:-1.0}
PREDICTORS=${PREDICTORS:-1}

source .venv/bin/activate

WANDB_FLAGS=()
if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  WANDB_ENTITY=${WANDB_ENTITY:-machine-learning-methods-in-software-engineering}
  export WANDB_ENTITY
  WANDB_FLAGS=(--wandb --wandb_project "${WANDB_PROJECT:-llm-jepa}")
fi

EVAL_FLAGS=()
if [[ -n "${EVAL_FILE}" && "${EVAL_STEPS}" -gt 0 ]]; then
  EVAL_FLAGS=(
    --eval_file "$EVAL_FILE"
    --eval_strategy=steps
    --eval_steps="$EVAL_STEPS"
  )
fi

torchrun --nproc_per_node="$NGPUS" --master_port=29504 finetune.py \
  --train_file "$TRAIN_FILE" \
  --output_dir="$OUTPUT_DIR" \
  --num_epochs="$NUM_EPOCHS" --finetune_seed="$SEED" \
  --last_token="$LAST_TOKEN" --lbd="$LBD" --predictors="$PREDICTORS" \
  --model_name="$MODEL_NAME" --learning_rate="$LEARNING_RATE" \
  --lora --lora_rank="$LORA_RANK" --batch_size="$BATCH_SIZE" --grad_accum="$GRAD_ACCUM" \
  --logging_steps="$LOGGING_STEPS" \
  "${EVAL_FLAGS[@]}" \
  --no_save \
  "${WANDB_FLAGS[@]}"

echo "Training complete. Model saved to: $OUTPUT_DIR"
