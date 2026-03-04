"""Execution-based MBPP pass@1 evaluation using vLLM generation."""

from dotenv import load_dotenv

load_dotenv()

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import re
from typing import Any

from datasets import load_dataset
from vllm import LLM, SamplingParams

from evaluate_vllm import format_conversation, get_messages, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MBPP pass@1 via execution-based unit tests."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model path/name")
    parser.add_argument(
        "--original_model_name",
        type=str,
        required=True,
        help="Original base model name (chat-template handling)",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="MBPP JSONL file with metadata"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="mbpp_pass1_results.jsonl",
        help="Per-example output JSONL",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="mbpp_pass1_summary.json",
        help="Summary output JSON",
    )
    parser.add_argument(
        "--max_examples", type=int, default=0, help="Limit number of examples (0=all)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max generated tokens"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument("--tp_size", type=int, default=1, help="vLLM tensor-parallel size")
    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=3.0,
        help="Per-problem execution timeout in seconds",
    )
    parser.add_argument(
        "--memory_limit_mb",
        type=int,
        default=0,
        help="Per-problem memory limit in MB (0=unlimited, POSIX only)",
    )
    parser.add_argument(
        "--use_challenge_tests",
        action="store_true",
        help="Use challenge_test_list when available",
    )
    parser.add_argument(
        "--default_system_prompt",
        type=str,
        default="Solve the Python task. Return only Python code.",
        help="Fallback system prompt when dataset row has no system message",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    return parser.parse_args()


def _extract_first_by_role(messages: list[dict[str, Any]], role: str) -> str | None:
    for message in messages:
        if message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None


def get_system_prompt(example: dict[str, Any], fallback: str) -> str:
    messages = example.get("messages")
    if isinstance(messages, list):
        system_prompt = _extract_first_by_role(messages, "system")
        if system_prompt and system_prompt.strip():
            return system_prompt.strip()
    return fallback


def get_user_prompt(example: dict[str, Any]) -> str:
    text = example.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    messages = example.get("messages")
    if isinstance(messages, list):
        user_prompt = _extract_first_by_role(messages, "user")
        if user_prompt and user_prompt.strip():
            return user_prompt.strip()

    raise ValueError("Missing MBPP prompt. Expected 'text' or user message content.")


def get_tests(example: dict[str, Any], use_challenge_tests: bool) -> list[str]:
    primary = "challenge_test_list" if use_challenge_tests else "test_list"
    secondary = "test_list" if use_challenge_tests else "challenge_test_list"

    tests = example.get(primary)
    if not isinstance(tests, list) or not tests:
        tests = example.get(secondary)

    if not isinstance(tests, list) or not tests:
        raise ValueError(
            "Missing tests. Rebuild dataset with "
            "scripts/prepare_mbpp_dataset.py --keep-metadata."
        )

    normalized = []
    for test in tests:
        if isinstance(test, str) and test.strip():
            normalized.append(test.strip())
    if not normalized:
        raise ValueError("Tests are present but empty after normalization.")
    return normalized


def sanitize_generated_code(text: str) -> str:
    text = text.strip()
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return text


def _set_memory_limit(memory_limit_mb: int) -> None:
    if memory_limit_mb <= 0:
        return
    try:
        import resource
    except Exception:
        return
    limit_bytes = memory_limit_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        return


def _exec_worker(payload: dict[str, Any], queue: mp.Queue) -> None:
    code = payload["code"]
    setup_code = payload["setup_code"]
    tests = payload["tests"]
    memory_limit_mb = payload["memory_limit_mb"]
    _set_memory_limit(memory_limit_mb)

    namespace: dict[str, Any] = {}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            compiled_code = compile(code, "<generated>", "exec")
            exec(compiled_code, namespace, namespace)
            if setup_code:
                compiled_setup = compile(setup_code, "<setup>", "exec")
                exec(compiled_setup, namespace, namespace)
            for idx, test in enumerate(tests):
                compiled_test = compile(test, f"<test_{idx}>", "exec")
                exec(compiled_test, namespace, namespace)
        queue.put({"passed": True, "error_type": None, "error_message": None})
    except SyntaxError as exc:
        queue.put(
            {
                "passed": False,
                "error_type": "syntax",
                "error_message": f"{exc.msg} (line {exc.lineno})",
            }
        )
    except AssertionError as exc:
        queue.put(
            {
                "passed": False,
                "error_type": "assertion",
                "error_message": str(exc) or "Assertion failed",
            }
        )
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "passed": False,
                "error_type": "runtime",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        )


def run_code_with_timeout(
    code: str,
    setup_code: str,
    tests: list[str],
    timeout_sec: float,
    memory_limit_mb: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    payload = {
        "code": code,
        "setup_code": setup_code,
        "tests": tests,
        "memory_limit_mb": memory_limit_mb,
    }
    proc = ctx.Process(target=_exec_worker, args=(payload, queue))
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {
            "passed": False,
            "error_type": "timeout",
            "error_message": f"Timed out after {timeout_sec:.2f}s",
        }

    if not queue.empty():
        return queue.get()

    return {
        "passed": False,
        "error_type": "runtime",
        "error_message": "No result returned from execution worker",
    }


def evaluate_mbpp(
    dataset: Any,
    responses: list[str],
    output_file: str,
    summary_file: str,
    timeout_sec: float,
    memory_limit_mb: int,
    use_challenge_tests: bool,
    debug: bool,
) -> dict[str, Any]:
    passed = 0
    total = len(dataset)
    error_counts: dict[str, int] = {
        "assertion": 0,
        "runtime": 0,
        "syntax": 0,
        "timeout": 0,
        "none": 0,
    }

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, (example, generated) in enumerate(zip(dataset, responses)):
            task_id = example.get("task_id", idx)
            tests = get_tests(example, use_challenge_tests)
            setup_code = example.get("test_setup_code", "")
            if not isinstance(setup_code, str):
                setup_code = ""

            generated = generated or ""
            generated_code = sanitize_generated_code(generated)
            result = run_code_with_timeout(
                code=generated_code,
                setup_code=setup_code,
                tests=tests,
                timeout_sec=timeout_sec,
                memory_limit_mb=memory_limit_mb,
            )
            if result["passed"]:
                passed += 1
                error_counts["none"] += 1
            else:
                error_counts[result["error_type"]] = error_counts.get(
                    result["error_type"], 0
                ) + 1

            row = {
                "index": idx,
                "task_id": task_id,
                "passed": result["passed"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "generated": generated,
                "generated_code": generated_code,
                "num_tests": len(tests),
                "used_challenge_tests": use_challenge_tests,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if debug:
                print(
                    f"[{idx + 1}/{total}] task_id={task_id} "
                    f"passed={result['passed']} error={result['error_type']}"
                )

    pass_at_1 = passed / total if total else 0.0
    summary = {
        "metric": "pass@1",
        "total": total,
        "passed": passed,
        "pass@1": pass_at_1,
        "error_counts": error_counts,
        "timeout_sec": timeout_sec,
        "memory_limit_mb": memory_limit_mb,
        "used_challenge_tests": use_challenge_tests,
        "output_file": output_file,
    }
    with open(summary_file, "w", encoding="utf-8") as summary_f:
        json.dump(summary, summary_f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    if not args.input_file.endswith(".jsonl"):
        raise ValueError("Only JSONL files are supported.")

    dataset = load_dataset("json", data_files=args.input_file)["train"]
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print("=== MBPP pass@1 Evaluation ===")
    print(f"Model: {args.model_name}")
    print(f"Input file: {args.input_file}")
    print(f"Examples: {len(dataset)}")
    print(f"Timeout: {args.timeout_sec}s")
    print(f"Challenge tests: {args.use_challenge_tests}")

    tokenizer = load_tokenizer(args.model_name)
    prompts = []
    for example in dataset:
        system_prompt = get_system_prompt(example, args.default_system_prompt)
        user_prompt = get_user_prompt(example)
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_messages = get_messages(args.original_model_name, prompt_messages)
        prompt = format_conversation(full_messages, tokenizer)
        prompts.append(prompt)

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    responses = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        if text.endswith("<|end|>"):
            text = text[:-7].strip()
        responses.append(text)

    summary = evaluate_mbpp(
        dataset=dataset,
        responses=responses,
        output_file=args.output_file,
        summary_file=args.summary_file,
        timeout_sec=args.timeout_sec,
        memory_limit_mb=args.memory_limit_mb,
        use_challenge_tests=args.use_challenge_tests,
        debug=args.debug,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
