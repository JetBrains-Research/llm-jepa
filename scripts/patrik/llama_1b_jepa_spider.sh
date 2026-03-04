#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

source .venv/bin/activate
torchrun --nproc_per_node=$NGPUS finetune.py \
  --train_file datasets/spider_train.jsonl \
  --output_dir=./fine-tuned-jepa \
  --num_epochs=4 --finetune_seed=82 \
  --last_token=-2 --lbd=0.5 --predictors=1 \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=1e-5 \
  --batch_size=4 --grad_accum=8 \
  --eval_accuracy --wandb --wandb_project=llm-jepa --no_save --max_eval_samples=50 \
  --spider_path=spider_data/database
