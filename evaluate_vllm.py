"""vLLM-accelerated evaluation for LLM-JEPA.

Drop-in replacement for evaluate.py that uses vLLM batch inference instead of
sequential HuggingFace generation.  Supports optional data parallelism
(--dp_size) for multi-GPU throughput scaling.

Embedding / similarity / t-SNE features are intentionally omitted; use the
original evaluate.py for those.
"""

from dotenv import load_dotenv
load_dotenv()

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import argparse
from multiprocessing import Process
from time import sleep

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ── Message helpers (from evaluate.py) ────────────────────────────────────────

def get_messages(model_name, messages):
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = (
            messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        )
        return full_messages
    return messages


def format_conversation(messages, tokenizer):
    """Format chat messages into a prompt string, excluding assistant turns."""
    messages = [msg for msg in messages if msg["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Task-specific evaluation (from evaluate.py) ──────────────────────────────

spider_pattern = re.compile(r"For db_id:\[(.+)\]")
gsm8k_pattern = re.compile(r"\n#### (.+)$")


def spider_eval(generated, ground_truth, spider_path, debug=0):
    db_id = re.search(spider_pattern, ground_truth[1]["content"])
    assert db_id
    db_id = db_id.group(1)
    dbfile = os.path.join(spider_path, db_id, db_id + ".sqlite")
    try:
        result = subprocess.run(
            ["sqlite3", dbfile, generated], capture_output=True, text=True
        )
        gen_result = result.stdout
        result = subprocess.run(
            ["sqlite3", dbfile, ground_truth[2]["content"]],
            capture_output=True,
            text=True,
        )
        gt_result = result.stdout
    except Exception:
        return False
    if debug == 1:
        print("[GEN]", gen_result)
        print("[GT:]", gt_result)
    return gen_result == gt_result


def eval_response(generated, ground_truth, input_file, spider_path,
                  startswith=False, debug=0):
    basename = os.path.basename(input_file)

    if startswith:
        if debug == 1:
            print("[GEN]", generated)
            print("[GT:]", ground_truth[2]["content"])
            print("-----startswith-----")
        return generated.startswith(ground_truth[2]["content"])

    if basename.startswith("gsm8k"):
        gt_match = re.search(gsm8k_pattern, ground_truth[2]["content"])
        gt_answer = None if not gt_match else gt_match.group(1)
        gen_match = re.search(gsm8k_pattern, generated)
        gen_answer = None if not gen_match else gen_match.group(1)
        if debug == 1:
            print("[RAW]", generated)
            print("[GEN]", gen_answer)
            print("[GT:]", gt_answer)
            print("-----GSM8K-----")
        return gt_answer == gen_answer

    if basename.startswith("spider"):
        return spider_eval(generated, ground_truth, spider_path, debug=debug)

    if basename.startswith("nq_open"):
        for answer in generated.split("; "):
            if answer in ground_truth[2]["content"]:
                return True
        return False

    if debug == 1:
        print("[GEN]", generated)
        print("[GT:]", ground_truth[2]["content"])
        print("-----")
    return generated == ground_truth[2]["content"]


# ── Dataset splitting (from evaluate.py) ─────────────────────────────────────

def split_dataset_and_save(input_file, train_file, test_file,
                           test_size=0.2, seed=42):
    print(f"\nSplitting dataset: {input_file}")
    print(f"Test size: {test_size}")
    print(f"Random seed: {seed}")

    if not input_file.endswith(".jsonl"):
        raise ValueError("Only JSONL files are supported")

    dataset = load_dataset("json", data_files=input_file)["train"]
    print(f"Total examples: {len(dataset)}")

    split_data = dataset.train_test_split(
        test_size=test_size, seed=seed, shuffle=True
    )
    train_dataset = split_data["train"]
    test_dataset = split_data["test"]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    for path, ds in [(train_file, train_dataset), (test_file, test_dataset)]:
        print(f"Saving to: {path}")
        with open(path, "w") as f:
            for example in ds:
                f.write(json.dumps(example) + "\n")

    print("Dataset splitting complete!")
    return train_file, test_file


# ── Tokenizer setup ──────────────────────────────────────────────────────────

def load_tokenizer(model_name):
    """Load tokenizer and add the special tokens used during fine-tuning."""
    if "apple/OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True
        )
        tokenizer.chat_template = (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n"
            "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'system' %}\n"
            "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'assistant' %}\n"
            "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
            "{% endif %}\n"
            "{% if loop.last and add_generation_prompt %}\n"
            "{{ '<|assistant|>' }}\n"
            "{% endif %}\n"
            "{% endfor %}"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = [
        "<|predictor_1|>", "<|predictor_2|>", "<|predictor_3|>",
        "<|predictor_4|>", "<|predictor_5|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "<|perception|>",
    ]
    new_tokens = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"Added {len(new_tokens)} new special tokens")

    return tokenizer


# ── vLLM generation ──────────────────────────────────────────────────────────

def _dp_worker(model_name, prompts, indices, dp_size, global_rank, local_rank,
               tp_size, max_tokens, result_dir):
    """Worker process for one data-parallel rank."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = visible.split(",") if visible else []
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            gpus[local_rank * tp_size:(local_rank + 1) * tp_size]
        )

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

    worker_prompts = prompts if prompts else ["Placeholder"]
    outputs = llm.generate(worker_prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        if i < len(indices):
            text = output.outputs[0].text.strip()
            if text.endswith("<|end|>"):
                text = text[:-7].strip()
            results.append((indices[i], text))

    with open(os.path.join(result_dir, f"rank_{global_rank}.json"), "w") as f:
        json.dump(results, f)

    sleep(1)


def _shard_range(rank, total, dp_size):
    """Return (start, end) indices for a given DP rank."""
    floor_val = total // dp_size
    remainder = total % dp_size
    start = rank * floor_val + min(rank, remainder)
    end = (rank + 1) * floor_val + min(rank + 1, remainder)
    return start, end


def generate_responses(model_name, prompts, max_tokens,
                       dp_size=1, tp_size=1, llm=None):
    """Batch-generate responses, optionally with data parallelism."""
    if dp_size <= 1:
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        outputs = llm.generate(prompts, sampling_params)
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            if text.endswith("<|end|>"):
                text = text[:-7].strip()
            results.append(text)
        return results

    result_dir = tempfile.mkdtemp(prefix="vllm_eval_")

    procs = []
    for rank in range(dp_size):
        s, e = _shard_range(rank, len(prompts), dp_size)
        my_indices = list(range(s, e))
        my_prompts = [prompts[i] for i in my_indices]
        proc = Process(
            target=_dp_worker,
            args=(
                model_name, my_prompts, my_indices,
                dp_size, rank, rank,
                tp_size, max_tokens, result_dir,
            ),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join(timeout=600)
        if proc.exitcode is None:
            print(f"Killing hung worker {proc.pid}")
            proc.kill()

    responses = [None] * len(prompts)
    for rank in range(dp_size):
        path = os.path.join(result_dir, f"rank_{rank}.json")
        with open(path) as f:
            for idx, text in json.load(f):
                responses[idx] = text

    shutil.rmtree(result_dir, ignore_errors=True)
    return responses


# ── Dataset processing & evaluation ──────────────────────────────────────────

def process_dataset(input_file, output_file, original_model_name, model_name,
                    tokenizer, spider_path, max_examples=None,
                    split_tune_untune=False, debug=0, startswith=False,
                    max_new_tokens=128, dp_size=1, tp_size=1, llm=None):
    """Format prompts, batch-generate with vLLM, evaluate results."""

    if not input_file.endswith(".jsonl"):
        raise ValueError("Only JSONL files are supported")

    dataset = load_dataset("json", data_files=input_file)["train"]
    print(f"Loaded {len(dataset)} examples from {input_file}")

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        print(f"Processing {len(dataset)} examples (limited by max_examples)")

    print("Formatting prompts...")
    prompts = []
    all_messages = []
    for example in dataset:
        messages = example["messages"]
        all_messages.append(messages)
        full_messages = get_messages(original_model_name, messages)
        prompts.append(format_conversation(full_messages, tokenizer))

    print(f"Generating {len(prompts)} responses with vLLM...")
    responses = generate_responses(
        model_name, prompts, max_new_tokens,
        dp_size=dp_size, tp_size=tp_size, llm=llm,
    )

    print("Evaluating responses...")
    n_correct = 0
    n_incorrect = 0
    n_startswith = 0

    for response, messages in zip(responses, all_messages):
        if response is None:
            response = ""

        if split_tune_untune:
            equal = eval_response(
                response, messages, input_file, spider_path,
                startswith=False, debug=debug,
            )
            if startswith:
                is_sw = eval_response(
                    response, messages, input_file, spider_path,
                    startswith=True, debug=debug,
                )
                if is_sw:
                    n_startswith += 1
            if debug == 2:
                print(
                    f"gt_vs_gen-{input_file}, "
                    f"{repr(messages[2]['content'])}, "
                    f"{repr(response)}, {equal}"
                )
            if equal:
                n_correct += 1
            else:
                n_incorrect += 1
        else:
            n_correct += 1

    total = n_correct + n_incorrect
    rate = n_correct / total if total > 0 else 0.0
    print(f"Success Rate: {model_name}, {rate}", end="")
    if startswith:
        print(f", {n_startswith / total if total > 0 else 0.0}")
    else:
        print()
    print(n_correct)
    if split_tune_untune:
        print(n_incorrect)

    if output_file:
        with open(output_file, "w") as f:
            for idx, (response, messages) in enumerate(
                zip(responses, all_messages)
            ):
                result = {
                    "index": idx,
                    "generated": response,
                    "ground_truth": (
                        messages[2]["content"] if len(messages) > 2 else None
                    ),
                }
                f.write(json.dumps(result) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses using vLLM batch inference"
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--original_model_name", type=str, required=True,
                        help="Original (base) model name")

    # Data
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSONL file")
    parser.add_argument("--output_file", type=str,
                        help="Output JSONL file for generated responses")
    parser.add_argument("--max_examples", type=int,
                        help="Maximum examples to process")

    # Splitting
    parser.add_argument("--nosplit_data", action="store_true",
                        help="Do not split input data")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set proportion (default: 0.2)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Split random seed (default: 42)")
    parser.add_argument("--train_file", type=str,
                        help="Train set output file")
    parser.add_argument("--test_file", type=str,
                        help="Test set output file")
    parser.add_argument("--process_split", type=str, default="test",
                        choices=["train", "test", "both"],
                        help="Which split to evaluate (default: test)")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate (-1 = unlimited)")

    # Evaluation
    parser.add_argument("--split_tune_untune", action="store_true",
                        help="Report correct/incorrect counts separately")
    parser.add_argument("--debug", type=int, default=0,
                        help="Debug level")
    parser.add_argument("--startswith", action="store_true",
                        help="Also report startswith matches")
    parser.add_argument("--spider_path", type=str, default="",
                        help="Path to Spider database directory")

    # vLLM
    parser.add_argument("--dp_size", type=int, default=1,
                        help="Data-parallel size (model replicas)")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="Tensor-parallel size (GPUs per replica)")

    args = parser.parse_args()

    if args.nosplit_data and not args.input_file:
        parser.error("--input_file required with --nosplit_data")

    max_new_tokens = None if args.max_new_tokens == -1 else args.max_new_tokens

    print("=== vLLM Evaluation ===")
    print(f"Model: {args.model_name}")
    print(f"Input: {args.input_file}")
    print(f"DP size: {args.dp_size}, TP size: {args.tp_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Split tune/untune: {args.split_tune_untune}")

    # ── Resolve files to process ──
    if not args.nosplit_data:
        base_name = os.path.splitext(args.input_file)[0]
        train_file = args.train_file or f"{base_name}_train.jsonl"
        test_file = args.test_file or f"{base_name}_test.jsonl"
        train_file, test_file = split_dataset_and_save(
            args.input_file, train_file, test_file,
            test_size=args.test_size, seed=args.split_seed,
        )
        files_to_process = []
        if args.process_split in ("train", "both"):
            out = args.output_file or f"{base_name}_train_responses.jsonl"
            files_to_process.append(("train", train_file, out))
        if args.process_split in ("test", "both"):
            out = args.output_file or f"{base_name}_test_responses.jsonl"
            files_to_process.append(("test", test_file, out))
    else:
        files_to_process = [("full", args.input_file, args.output_file)]

    # ── Load tokenizer (for prompt formatting) ──
    print("\n1. Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    # ── Create vLLM engine (single-process path only) ──
    llm = None
    if args.dp_size <= 1:
        print("\n2. Loading vLLM model...")
        llm = LLM(
            model=args.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=args.tp_size,
        )

    # ── Process each file ──
    print(f"\n3. Processing {len(files_to_process)} file(s)...")
    for split_name, input_file, output_file in files_to_process:
        print(f"\n--- Processing {split_name}: {input_file} ---")
        process_dataset(
            input_file=input_file,
            output_file=output_file,
            original_model_name=args.original_model_name,
            model_name=args.model_name,
            tokenizer=tokenizer,
            spider_path=args.spider_path,
            max_examples=args.max_examples,
            split_tune_untune=args.split_tune_untune,
            debug=args.debug,
            startswith=args.startswith,
            max_new_tokens=max_new_tokens,
            dp_size=args.dp_size,
            tp_size=args.tp_size,
            llm=llm,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
