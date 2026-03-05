#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# 64 effective = 8*8

source .venv/bin/activate
torchrun --nproc_per_node=$NGPUS finetune.py \
  --train_file datasets/spider_train.jsonl \
  --output_dir=./results/fine-tuned \
  --num_epochs=4 --finetune_seed=82 --regular \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=4e-5 \
  --batch_size=8 --grad_accum=8 \
  --eval_accuracy --eval_vllm --wandb --wandb_project=llm-jepa --no_save \
  --spider_path=spider_data/database --use_original_test
