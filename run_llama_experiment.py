"""
Parallel LLM-JEPA vs Baseline experiment on GSM8K / Llama-3.2-1B-Instruct.

Training:
  - Baseline (--regular)  on GPUs 0-1
  - LLM-JEPA (--lbd/--predictors)  on GPUs 2-3
  Both run simultaneously.

Evaluation (after training):
  - Each model evaluated in parallel across 4 chunks on GPUs 0-3.

Usage:
    source .venv/bin/activate
    python run_llama_experiment.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
MODEL        = "meta-llama/Llama-3.2-1B-Instruct"
DATASET      = "mbpp"
SEED         = 82
LAST_TOKEN   = -2
LR           = 1e-5
EPOCHS       = 4
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
JEPA_LBD     = 0.5
JEPA_PREDICTORS = 4
MAX_NEW_TOKENS  = 512
WANDB_PROJECT   = "llm-jepa"

RESULTS_DIR  = f"results/{DATASET}_llama"
CHUNKS_DIR   = f"results/{DATASET}_llama/eval_chunks"
VENV         = ".venv/bin"

# Which GPU pairs to use for training
REGULAR_GPUS = "1,2"   # torchrun will use these
JEPA_GPUS    = "3,4"
EVAL_GPUS    = [0, 1, 2, 3]   # all 4 for evaluation
# ────────────────────────────────────────────────────────────────────────────


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def split_test_file(n_chunks: int) -> list[str]:
    pass
    # """Split gsm8k_test.jsonl into n_chunks, return list of chunk paths."""
    # os.makedirs(CHUNKS_DIR, exist_ok=True)
    # src = "gsm8k_test.jsonl"
    # with open(src) as f:
    #     lines = f.readlines()
    #
    # total = len(lines)
    # size  = (total + n_chunks - 1) // n_chunks
    # paths = []
    # for i in range(n_chunks):
    #     chunk_lines = lines[i * size : (i + 1) * size]
    #     # keep "gsm8k" in filename so evaluate.py detects the dataset
    #     path = os.path.join(CHUNKS_DIR, f"gsm8k_eval_chunk_{i}.jsonl")
    #     with open(path, "w") as f:
    #         f.writelines(chunk_lines)
    #     paths.append(path)
    # log(f"Split {total} examples into {n_chunks} chunks of ≤{size}")
    # return paths


def launch_training(method: str, gpu_ids: str, port: int) -> subprocess.Popen:
    """Start a training job (non-blocking). Returns the Popen handle."""
    n_gpus  = len(gpu_ids.split(","))
    out_dir = os.path.join(RESULTS_DIR, f"{method}_s{SEED}")
    log_path = os.path.join(RESULTS_DIR, f"{method}_s{SEED}_train.log")

    cmd = [
        f"{VENV}/torchrun",
        f"--nproc_per_node={n_gpus}",
        f"--master_port={port}",
        "finetune.py",
        f"--train_file=datasets/{DATASET}_train.jsonl",
        f"--model_name={MODEL}",
        f"--output_dir={out_dir}",
        f"--num_epochs={EPOCHS}",
        f"--learning_rate={LR}",
        f"--last_token={LAST_TOKEN}",
        f"--finetune_seed={SEED}",
        f"--batch_size={BATCH_SIZE}",
        f"--grad_accum={GRAD_ACCUM}",
        "--no_save",
        "--wandb",
        f"--wandb_project={WANDB_PROJECT}",
        f"--wandb_run_name={DATASET}_llama_{method}_s{SEED}",
    ]
    if method == "regular":
        cmd.append("--regular")
    else:
        cmd += [f"--lbd={JEPA_LBD}", f"--predictors={JEPA_PREDICTORS}"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    log(f"▶ Launching {method} training on GPUs [{gpu_ids}]  port={port}")
    log(f"  log: {log_path}")

    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=".", env=env)
    proc._log_file  = lf
    proc._log_path  = log_path
    proc._method    = method
    proc._output_dir = out_dir
    return proc


def run_parallel_eval(method: str, model_dir: str, chunk_paths: list[str]) -> float | None:
    """Evaluate model in parallel across chunks; return accuracy [0,100] or None."""
    n = len(chunk_paths)
    out_files  = [os.path.join(CHUNKS_DIR, f"{method}_out_{i}.jsonl") for i in range(n)]
    log_files  = [os.path.join(CHUNKS_DIR, f"{method}_log_{i}.log")   for i in range(n)]

    # Remove stale outputs so evaluate.py doesn't use cached zeros
    for p in out_files:
        if os.path.exists(p):
            os.remove(p)

    procs = []
    for i, (chunk, out, logf, gpu) in enumerate(zip(chunk_paths, out_files, log_files, EVAL_GPUS)):
        cmd = [
            f"{VENV}/python", "evaluate.py",
            f"--model_name={model_dir}",
            f"--input_file={chunk}",
            f"--output_file={out}",
            "--split_tune_untune",
            f"--original_model_name={MODEL}",
            "--nosplit_data",
            f"--max_new_tokens={MAX_NEW_TOKENS}",
            "--max_length=2048",
            "--device_map=cuda:0",   # CUDA_VISIBLE_DEVICES remaps physical GPU to index 0
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # isolate to this physical GPU
        lf   = open(logf, "w")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=".", env=env)
        proc._log_file = lf
        proc._chunk    = chunk
        proc._out      = out
        proc._logf     = logf
        procs.append(proc)

    log(f"  Evaluating {method} on {n} chunks across GPUs {EVAL_GPUS} …")

    for proc in procs:
        proc.wait()
        proc._log_file.close()

    # Tally correct across chunks
    correct = 0
    total   = 0
    pattern = re.compile(r"Success Rate: .+?,\s*([0-9.]+)")
    for logf in log_files:
        with open(logf) as f:
            content = f.read()
        m = pattern.search(content)
        if m:
            rate  = float(m.group(1))
            # count examples in this chunk
            chunk_idx = log_files.index(logf)
            with open(chunk_paths[chunk_idx]) as cf:
                n_ex = sum(1 for _ in cf)
            correct += round(rate * n_ex)
            total   += n_ex
        else:
            log(f"  ✗ Could not parse chunk {logf}")

    if total == 0:
        return None
    acc = correct / total * 100
    log(f"  ✓ {method} accuracy = {acc:.2f}%  ({correct}/{total})")
    return acc


def generate_report(reg_acc: float | None, jepa_acc: float | None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# LLM-JEPA vs Baseline — GSM8K / Llama-3.2-1B-Instruct",
        f"",
        f"> Experiment date: {now}  ",
        f"> Seed: {SEED}",
        f"",
        f"## Setup",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Model | `{MODEL}` |",
        f"| Dataset | GSM8K |",
        f"| Seed | {SEED} |",
        f"| Epochs | {EPOCHS} |",
        f"| Learning rate | `{LR}` |",
        f"| Batch size (per GPU) | {BATCH_SIZE} |",
        f"| Gradient accumulation | {GRAD_ACCUM} |",
        f"| Effective batch size | {2 * BATCH_SIZE * GRAD_ACCUM} (2 GPUs) |",
        f"| Training GPUs | 2 per run (0-1 regular, 2-3 JEPA) |",
        f"| JEPA λ | {JEPA_LBD} |",
        f"| JEPA predictors k | {JEPA_PREDICTORS} |",
        f"| `--last_token` | {LAST_TOKEN} |",
        f"",
        f"## Results",
        f"",
        f"| Method | GSM8K Accuracy | Paper (Llama-3.2-1B) |",
        f"|--------|---------------|----------------------|",
    ]

    reg_s  = f"{reg_acc:.2f}%"  if reg_acc  is not None else "—"
    jepa_s = f"{jepa_acc:.2f}%" if jepa_acc is not None else "—"
    lines += [
        f"| Baseline (ℒ_LLM) | **{reg_s}** | 32.36% |",
        f"| LLM-JEPA (ℒ_LLM−JEPA) | **{jepa_s}** | 36.36% |",
    ]

    if reg_acc is not None and jepa_acc is not None:
        delta = jepa_acc - reg_acc
        sign  = "+" if delta >= 0 else ""
        lines += [
            f"| Improvement | **{sign}{delta:.2f} pp** | +4.00 pp |",
            f"",
            f"## Analysis",
            f"",
        ]
        if delta > 0:
            lines.append(
                f"LLM-JEPA improves over the standard fine-tuning baseline by "
                f"**{delta:.2f} percentage points** on GSM8K (seed {SEED})."
            )
        else:
            lines.append(
                f"On this single seed, LLM-JEPA shows **{delta:.2f} pp** vs the baseline. "
                f"Multiple seeds are needed for a reliable comparison."
            )
        lines += [
            f"",
            f"The paper reported **+4.00 pp** averaged across 5 seeds with Llama-3.2-1B-Instruct.",
            f"This is a single-seed run for quick validation.",
        ]

    lines += [
        f"",
        f"## Raw Numbers",
        f"",
        f"```json",
        json.dumps({"seed": SEED, "regular_acc": reg_acc, "jepa_acc": jepa_acc}, indent=2),
        f"```",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Pre-split the test file (4 chunks for evaluation)
    chunk_paths = split_test_file(n_chunks=4)

    # ── 1. Launch both training jobs in parallel ────────────────────────────
    log("=" * 60)
    log("Phase 1: Parallel training")
    log("  Regular → GPUs 0-1 | JEPA → GPUs 2-3")
    log("=" * 60)

    reg_proc  = launch_training("regular", REGULAR_GPUS, port=29810)
    jepa_proc = launch_training("jepa",    JEPA_GPUS,    port=29812)

    # Wait for both
    reg_rc  = reg_proc.wait();  reg_proc._log_file.close()
    jepa_rc = jepa_proc.wait(); jepa_proc._log_file.close()

    log(f"Regular training exit code: {reg_rc}")
    log(f"JEPA training exit code:    {jepa_rc}")

    if reg_rc != 0:
        log("✗ Regular training FAILED — check results/gsm8k_llama/regular_s82_train.log")
    if jepa_rc != 0:
        log("✗ JEPA training FAILED — check results/gsm8k_llama/jepa_s82_train.log")

    # ── 2. Evaluate both models ─────────────────────────────────────────────
    # log("=" * 60)
    # log("Phase 2: Evaluation")
    # log("=" * 60)
    #
    # reg_acc  = None
    # jepa_acc = None
    #
    # if reg_rc == 0:
    #     log(f"▶ Evaluating regular model …")
    #     reg_acc = run_parallel_eval("regular", reg_proc._output_dir, chunk_paths)
    #
    # if jepa_rc == 0:
    #     log(f"▶ Evaluating JEPA model …")
    #     jepa_acc = run_parallel_eval("jepa", jepa_proc._output_dir, chunk_paths)
    #
    # # ── 3. Report ───────────────────────────────────────────────────────────
    # log("=" * 60)
    # log("Phase 3: Report")
    # log("=" * 60)
    #
    # report = generate_report(reg_acc, jepa_acc)
    # report_path = os.path.join(RESULTS_DIR, "report_llama.md")
    # with open(report_path, "w") as f:
    #     f.write(report)
    #
    # print("\n" + report)
    # log(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
