# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-JEPA adapts Joint Embedding Predictive Architecture (JEPA) from vision to language. Instead of predicting discrete tokens, the model learns to predict latent representations — aligning user-query and assistant-response embeddings in a shared space during fine-tuning. The companion **Semantic Tube Prediction (STP)** extension predicts masked random spans in embedding space rather than full sequence views.

Paper is in paper.html

## Commands

### Training

```bash
# Standard fine-tuning
torchrun --nproc_per_node=8 finetune.py \
  --train_file gsm8k_train.jsonl --output_dir=./fine-tuned \
  --num_epochs=4 --finetune_seed=82 --regular \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=1e-5

# LLM-JEPA fine-tuning
torchrun --nproc_per_node=8 finetune.py \
  --train_file gsm8k_train.jsonl --output_dir=./fine-tuned \
  --num_epochs=4 --finetune_seed=82 \
  --last_token=-2 --lbd=0.5 --predictors=1 \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=1e-5

# Semantic Tube Prediction
torchrun --nproc_per_node=8 stp.py \
  --train_file synth_train.jsonl --output_dir=./fine-tuned \
  --num_epochs=4 --finetune_seed=82 \
  --last_token=-2 --lbd=0.02 --predictors=0 \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=2e-5 \
  --linear=random_span
```

### Evaluation

```bash
python evaluate.py --model_name=./fine-tuned \
  --input_file=gsm8k_test.jsonl --output_file=eval.jsonl \
  --split_tune_untune --original_model_name=meta-llama/Llama-3.2-1B-Instruct \
  --nosplit_data --spider_path=spider_data/database
```

### Driver Scripts

`run.sh` defines `run_regular()` and `run_jepa()` for `finetune.py`. `run_stp.py` defines the same for `stp.py` with an extra `model_folder` arg. `run8bh200.sh` is for 8B+ models on H200 GPUs. `run_qwen3_8b.sh` runs Qwen3-8B with LoRA rank=256 on 2 GPUs.

## Architecture

### Entry Points

| File | Purpose |
|------|---------|
| `finetune.py` | LLM-JEPA fine-tuning (original two/three-pass or additive mask) |
| `stp.py` | Semantic Tube Prediction (extends finetune with span masking, linear predictor) |
| `finetune8bh200.py` | Large model variant for H200 GPUs |
| `evaluate.py` | Generation + accuracy evaluation |

### Core Abstractions

**`RepresentationTrainer`** (subclass of HuggingFace `Trainer`) — the central class in both `finetune.py` and `stp.py`. Overrides `compute_loss()` to combine:
- **LM loss**: standard cross-entropy on assistant tokens only
- **JEPA loss**: cosine similarity between user-view and assistant-view embeddings

`total_loss = gamma * lm_loss + lbd * jepa_loss`

**Multi-view forward pass**: the model runs on three views of each conversation:
1. Full conversation (system + user + assistant) — produces LM loss
2. User-only view (system + user + predictor tokens) — produces user embedding
3. Assistant-only view — produces assistant embedding

With `--additive_mask`, views 2 and 3 are combined into a single forward pass using a 4D causal attention mask.

**`LinearPredictor`** (in `stp.py`) — lightweight linear head for span-based prediction in STP mode.

### Model-Specific `last_token` Settings

The embedding is extracted from the last meaningful token, which varies by model family:

| Model | `last_token` |
|-------|-------------|
| Llama, Gemma, Phi | `-2` |
| OLMo, DeepSeek | `-1` |
| Qwen | `-3` |
| OpenELM | `-4` |

### Data Format

JSONL files with 3-message OpenAI format: `[system, user, assistant]`. Labels mask everything except assistant tokens (`-100`). Datasets live in `datasets/` directory.

### Key Flags

- `--additive_mask`: single forward pass for both views via 4D attention mask (recommended)
- `--lora / --lora_rank N`: LoRA fine-tuning
- `--pretrain`: train from random weights
- `--jepa_ratio`: random JEPA-loss dropout (e.g., 0.75 = 25% JEPA, 1.25x compute)
- `--track_flop / --same_flop`: FLOPs profiling and fair-compute comparison
- `--linear=random_span|e2e|curvature`: STP modes
- JEPA loss variants: `--jepa_l2`, `--jepa_mse`, `--infonce`, `--front_pred`, `--reverse_pred`

## Dependencies

Core: `torch` (CUDA 12.6+), `transformers`, `datasets`, `accelerate`, `peft`, `sentencepiece`. See `setup.sh` for exact versions — do not run it directly, pick commands for your environment.

Supported models: Llama-3.2, Gemma-2, Phi-1.5, OLMo-2, OpenELM, Qwen, DeepSeek.

Qwen3 models have "thinking mode" enabled by default — disable via `enable_thinking=False` in `apply_chat_template()` to prevent `<think>` token contamination during training/evaluation.
