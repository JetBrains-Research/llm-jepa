#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# 8 effective = 2*4*1

source .venv/bin/activate
torchrun --nproc_per_node=$NGPUS --master_port=29503 finetune.py \
  --train_file datasets/synth_train.jsonl \
  --output_dir=./results/fine-tuned \
  --num_epochs=4 --finetune_seed=82 --regular \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=2e-5 \
  --batch_size=2 --grad_accum=4 \
  --eval_accuracy --eval_vllm --wandb --wandb_project=llm-jepa --no_save --use_original_test \
  --temperature 1.0
