#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

source .venv/bin/activate
torchrun --nproc_per_node=$NGPUS finetune.py \
  --train_file datasets/gsm8k_train.jsonl \
  --output_dir=./results/fine-tuned \
  --num_epochs=5 --finetune_seed=82 --regular \
  --model_name=Qwen/Qwen2.5-7B-Instruct --learning_rate=2e-5 \
  --lora --lora_rank=256 --batch_size=8 --grad_accum=8 \
  --eval_accuracy --eval_vllm --wandb --wandb_project=llm-jepa --no_save
