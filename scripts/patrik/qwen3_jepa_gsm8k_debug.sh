#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

source .venv/bin/activate
torchrun --nproc_per_node=$NGPUS --master_port=29501 finetune.py \
  --train_file datasets/gsm8k_train.jsonl \
  --output_dir=./fine-tuned-jepa \
  --num_epochs=4 --finetune_seed=82 \
  --last_token=-3 --lbd=0.5 --predictors=1 \
  --model_name=Qwen/Qwen3-0.6B --learning_rate=4e-5 \
  --lora --lora_rank=256 --batch_size=8 --grad_accum=8 \
  --eval_accuracy --eval_vllm --wandb --wandb_project=llm-jepa --no_save
