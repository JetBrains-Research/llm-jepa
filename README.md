# LLM-JEPA

## Set Up

See `setup.sh`.

**NOTE**: Do NOT run `setup.sh` directly. Read the file, choose the configuration for your envirnoment, and execute the relevant commands manually.

<a id="stp"></a>
## Semantic Tube Prediction

The fine-tuning script is in `stp.py`. A convenient driver script, `run_stp.sh`, provides `run_regular()` for standard fine-tuning, and `run_stp_jepa()` for Semantic Tupe Prediction fine-tuning.

General flags:

*   `--linear=random_span` for Semantic Tube Prediction.
*   `--linear_predictor` for training a linear predictor.

Ablation study flags:

*   `--linear=e2e` for Two View in ablation study.
*   `--random_span_mask` and `--random_span_mask_recover` for Mask in ablation study.
*   `--linear=curvature` for Curvature in ablation study.

Other flags are documented in `stp.py`.

`run_stp_jepa()` will ignore `predictors`.

## LLM-JEPA Fine-tuning

The fine-tuning script is in `finetune.py`. A convenient driver script, `run.sh`, provides `run_regular()` for standard fine-tuning, and `run_jepa()` for LLM-JEPA fine-tuning.

For all experiments, we fix number of epochs to 4. The `last_token` setting depends on the model family; see the commented lines in `run.sh` for how to set it. Each configuration is run with 5 random seeds. We report mean accuracy and standard deviation.

The original implementation required two additional forward passes to encode `Text` and `Code` separately. The latest version combines them into a single forward pass using a 4D additive attention mask. Enable this feature with `--additive_mask`. **NOTE**: `--additive_mask` may not work if the tokenizer applies left-padding.

## Large models

Similarly, we provide `finetune8bh200.py` and `run8bh200.sh` for training modesl up to 8B parameters on NVIDIA H200 GPUs.

## LLM-JEPA with LoRA

Use `--lora` and `--lora_rank <N>` to enable LoRA fine-tuning for LLM-JEPA.

## Pretraining

Use `--pretrain` to start from randomly initialized weights.

For pretraining on the `paraphrase` dataset, pass `--plain --trainall` to disable the OpenAI message format, train next-token prediction, and jointly minimize distances between paraphrase variants.

After pretaining, fine-tune with `--plain` on `rotten_tomatoes` and `yelp`. For evaluation, run with `--plain --statswith` to bypass the OpenAI message format and score only the first token(the model isn't instruction-tuned, so it may not emit a clean stop).

## Ablation of JEPA-loss

We provide several options for ablating JEPA-loss in `finetune.py`:

*  L2 norm: pass `--jepa_l2`
*  Mean squred error: pass `--jepa_mse`
*  Prepend `[PRED]` token to `Text`: pass `--front_pred`
*  Let `Code` predict `Text`: pass `--reverse_pred`
*  Use InfoNCE loss, pass `--infonce`

## FLOPs

To track FLOPs per step, pass `--track_flop` to `finetune.py`. This prints the FLOPs for the first 10 steps. The total FLOPs can be estimated as `PER_STEP_FLOPS * NUMBER_OF_STEPS`. When `--jepa_ratio` is enabled (see [Random JEPA-loss Dropout](#random-jepa-loss-dropout) below), FLOPs may vary across steps; in this case, use the _average_ FLOPs per step instead.

For fair comparisons, we provide `--same_flop`, which computes the number of training steps required to match the total FLOPs of standard fine-tuning, taking into account `--additive_mask` and/or `--jepa_ratio`. Checkpoints are saved at those steps and can be used for evaluatioin. 

*  If `--additive_mask` is enabled, the same number of steps requires `2X` the compute.
*  If `--jepa_ratio` is set to `1 - alpha`, the same number of steps use `(2 - alpha)X` the compute.

## Random JEPA-loss Dropout

The fine-tuning script `finetune.py` supports `--jepa_ratio` to implement **random JEPA-loss dropout**. The idea is that randomly dorpping some JEPA-loss has little impact on performance, but can substaintially reduce compute cost.

When dropout is active, the extra forward pass for `Enc(Text)` and `Enc(Code)` is skipped. If the dropout rate `LD = alpha`, then correspondingly `--jepa_ratio` should be set to `1 - alpha`. On average, one training step costs `(2 - alpha)X` the compute of standard fine-tuning.

Empirical results show that LLM-JEPA can tolerate aggressive dropout rate (e.g., `LD = 0.75`), requiring `1.25X` the compute while maintaining fine-tuning performance.

## Datasets

Most datasets include `_train.jsonl` and `_test.jsonl` files for fine-tuning and evaluation, repsectively. The originals come from prior publications; we preprocessed them and include the results here for convenience.

*  `synth` and `turk`, from https://arxiv.org/abs/1608.03000
*  `gsm8k`, from https://arxiv.org/abs/2110.14168
*  `spider`, from https://arxiv.org/abs/1809.08887. You aslo need to unzip `spider_data.zip` which contains `sqlite` databases to execute and evaluate the generated queries.
*  `paraphrase`, from HuggingFace `cestwc/paraphrase` dataset. Only have `train` split, for pre-training only.
*  `rotten_tomatoes`, from HuggingFace `cornell-movie-review-data/rotten_tomatoes` dataset. Used for fine-tuning and evaluating models pretrained by `paraphrase` dataset.
*  `yelp`, from HuggingFace `Yelp/yelp_review_full` dataset. Used for fine-tuning and evaluating models pretrained by `paraphrase` dataset.
*  `nq_open`, from https://arxiv.org/abs/1906.00300.
*  `hellaswag`, from HuggingFace `hellaswag` dataset.

## MBPP Setup and Evaluation

This repo includes utilities to prepare MBPP and evaluate with execution-based `pass@1`.

### 1. Environment setup

1. Follow `setup.sh` instructions to install dependencies.
2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

### 2. Prepare MBPP dataset

Generate `datasets/mbpp_train.jsonl` and `datasets/mbpp_test.jsonl`:

```bash
python scripts/prepare_mbpp_dataset.py \
  --config sanitized \
  --output-dir datasets \
  --train-file mbpp_train.jsonl \
  --test-file mbpp_test.jsonl \
  --keep-metadata
```

For a custom random 90/10 split:

```bash
python scripts/prepare_mbpp_dataset.py \
  --config sanitized \
  --output-dir datasets \
  --train-file mbpp_train.jsonl \
  --test-file mbpp_test.jsonl \
  --keep-metadata \
  --resplit-test-size 0.1 \
  --split-seed 42
```

Notes:

* Keep `--keep-metadata` enabled for execution-based eval (`test_list`, `test_setup_code`, etc.).
* Do not use `--include-tests` for unbiased eval prompts, because it appends tests into the user prompt.

### 3. Train Qwen3-8B (regular) on MBPP

Use the new convenience script:

```bash
scripts/qwen3_8b_regular_mbpp.sh
```

Optional overrides:

```bash
MODEL_NAME=Qwen/Qwen3-8B \
TRAIN_FILE=datasets/mbpp_train.jsonl \
TEST_FILE=datasets/mbpp_test.jsonl \
OUTPUT_DIR=./fine-tuned-mbpp \
SEED=82 \
CUDA_VISIBLE_DEVICES=2,3 \
scripts/qwen3_8b_regular_mbpp.sh
```

### 4. Run MBPP execution-based pass@1 evaluation directly

You can also run evaluation separately:

```bash
python evaluate_mbpp_pass1.py \
  --model_name ./fine-tuned-mbpp \
  --original_model_name Qwen/Qwen3-8B \
  --input_file datasets/mbpp_test.jsonl \
  --output_file fine-tuned-mbpp/mbpp_pass1_results.jsonl \
  --summary_file fine-tuned-mbpp/mbpp_pass1_summary.json \
  --max_new_tokens 512 \
  --timeout_sec 3
```

Outputs:

* `mbpp_pass1_results.jsonl`: per-example pass/fail and error type.
* `mbpp_pass1_summary.json`: aggregate `pass@1` and error counts.

## Codeforces Setup and Training

This section prepares `open-r1/codeforces` for supervised fine-tuning and runs Qwen3-8B training.

Important: this dataset does not expose a canonical accepted solution code field in every subset.  
By default, the prep script uses `editorial` as the assistant target text.

### 1. Prepare Codeforces dataset

```bash
source .venv/bin/activate

python scripts/prepare_codeforces_dataset.py \
  --dataset open-r1/codeforces \
  --config verifiable \
  --assistant-fields editorial \
  --output-dir datasets \
  --train-file codeforces_train.jsonl \
  --test-file codeforces_test.jsonl
```

Optional 90/10 random split:

```bash
python scripts/prepare_codeforces_dataset.py \
  --dataset open-r1/codeforces \
  --config verifiable \
  --assistant-fields editorial \
  --output-dir datasets \
  --train-file codeforces_train.jsonl \
  --test-file codeforces_test.jsonl \
  --resplit-test-size 0.1 \
  --split-seed 42
```

`--config verifiable-problems` is also accepted as a legacy alias and maps to `verifiable`.

Default training behavior in both Codeforces scripts:

* Uses `datasets/codeforces_test.jsonl` as `--eval_file`
* Runs evaluation every `50` steps (`--eval_strategy=steps --eval_steps=50`)

### 2. Train Qwen3-8B regular

```bash
scripts/qwen3_8b_regular_codeforces.sh
```

Common overrides:

```bash
MODEL_NAME=Qwen/Qwen3-8B \
TRAIN_FILE=datasets/codeforces_train.jsonl \
OUTPUT_DIR=./fine-tuned-codeforces-regular \
CUDA_VISIBLE_DEVICES=2,3 \
WANDB_ENABLED=1 \
scripts/qwen3_8b_regular_codeforces.sh
```

### 3. Train Qwen3-8B JEPA

```bash
scripts/qwen3_8b_jepa_codeforces.sh
```

Common overrides:

```bash
MODEL_NAME=Qwen/Qwen3-8B \
TRAIN_FILE=datasets/codeforces_train.jsonl \
OUTPUT_DIR=./fine-tuned-codeforces-jepa \
LBD=1.0 \
PREDICTORS=1 \
LAST_TOKEN=-3 \
CUDA_VISIBLE_DEVICES=2,3 \
WANDB_ENABLED=1 \
scripts/qwen3_8b_jepa_codeforces.sh
```

To disable step-based eval, set:

```bash
EVAL_FILE="" EVAL_STEPS=0
```
