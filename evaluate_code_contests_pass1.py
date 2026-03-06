"""Execution-based CodeContests pass@1 evaluation using vLLM generation."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import re
import sys
import copy
from typing import Any

def get_messages(model_name: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = (
            messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        )
        return full_messages
    return messages


def format_conversation(
    messages: list[dict[str, str]],
    tokenizer: Any,
    model_name: str,
    enable_thinking: bool,
) -> str:
    filtered = [message for message in messages if message["role"] != "assistant"]
    chat_template_kwargs = {}
    if "qwen3" in model_name.lower() and not enable_thinking:
        chat_template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        filtered,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CodeContests pass@1 via execution-based tests. "
            "Supports model-generated code or reference solution self-check mode."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["generated", "reference"],
        default="generated",
        help=(
            "generated: run vLLM generation and evaluate outputs; "
            "reference: evaluate assistant/reference solutions from input JSONL."
        ),
    )
    parser.add_argument("--model_name", type=str, help="Model path/name")
    parser.add_argument(
        "--original_model_name",
        type=str,
        help="Original base model name (chat-template handling)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="CodeContests JSONL file with tests metadata.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="code_contests_pass1_results.jsonl",
        help="Per-example output JSONL",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="code_contests_pass1_summary.json",
        help="Summary output JSON",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Limit number of examples (0=all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max generated tokens (generated mode)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument("--tp_size", type=int, default=1, help="vLLM tensor-parallel size")
    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=5.0,
        help="Per-problem execution timeout in seconds",
    )
    parser.add_argument(
        "--memory_limit_mb",
        type=int,
        default=0,
        help="Per-problem memory limit in MB (0=unlimited, POSIX only)",
    )
    parser.add_argument(
        "--test_fields",
        nargs="+",
        default=["public_tests", "private_tests", "generated_tests"],
        help=(
            "Test-case fields to evaluate (in order). "
            "Default: public_tests private_tests generated_tests."
        ),
    )
    parser.add_argument(
        "--max_tests_per_field",
        type=int,
        default=0,
        help="Optional cap of tests per field (0=all).",
    )
    parser.add_argument(
        "--default_system_prompt",
        type=str,
        default="Solve the competitive programming problem in Python 3. Return only Python code.",
        help="Fallback system prompt when dataset row has no system message.",
    )
    parser.add_argument(
        "--compare_mode",
        choices=["tokens", "lines"],
        default="tokens",
        help="Output comparison mode. tokens is whitespace-insensitive.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode during generation (disabled by default).",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()
    if args.mode == "generated":
        if not args.model_name:
            parser.error("--model_name is required when --mode=generated")
        if not args.original_model_name:
            parser.error("--original_model_name is required when --mode=generated")
    return args


def _extract_first_by_role(messages: list[dict[str, Any]], role: str) -> str | None:
    for message in messages:
        if message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def get_system_prompt(example: dict[str, Any], fallback: str) -> str:
    messages = example.get("messages")
    if isinstance(messages, list):
        system_prompt = _extract_first_by_role(messages, "system")
        if system_prompt and system_prompt.strip():
            return system_prompt.strip()
    return fallback


def get_user_prompt(example: dict[str, Any]) -> str:
    messages = example.get("messages")
    if isinstance(messages, list):
        user_prompt = _extract_first_by_role(messages, "user")
        if user_prompt and user_prompt.strip():
            return user_prompt.strip()
    description = _normalize_text(example.get("description"))
    if description:
        return description
    raise ValueError("Missing prompt. Expected user message or 'description'.")


def get_reference_solution(example: dict[str, Any]) -> str:
    messages = example.get("messages")
    if isinstance(messages, list):
        assistant = _extract_first_by_role(messages, "assistant")
        if assistant and assistant.strip():
            return assistant.strip()

    for key in ("reference_solution", "solution", "code"):
        value = _normalize_text(example.get(key))
        if value:
            return value
    raise ValueError("Missing reference solution code in input row.")


def _iter_cases_from_field(field_value: Any) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    if isinstance(field_value, dict):
        inputs = field_value.get("input")
        outputs = field_value.get("output")
        if isinstance(inputs, list) and isinstance(outputs, list):
            for inp, out in zip(inputs, outputs):
                inp_text = _normalize_text(inp)
                out_text = _normalize_text(out)
                if inp_text and out_text:
                    cases.append((inp_text, out_text))
        return cases

    if isinstance(field_value, list):
        for item in field_value:
            if not isinstance(item, dict):
                continue
            inp_text = _normalize_text(item.get("input"))
            out_text = _normalize_text(item.get("output"))
            if inp_text and out_text:
                cases.append((inp_text, out_text))
    return cases


def get_test_cases(
    example: dict[str, Any], test_fields: list[str], max_tests_per_field: int
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    all_cases: list[tuple[str, str]] = []
    per_field_counts: dict[str, int] = {}
    for field in test_fields:
        cases = _iter_cases_from_field(example.get(field))
        if max_tests_per_field > 0:
            cases = cases[:max_tests_per_field]
        if cases:
            per_field_counts[field] = len(cases)
            all_cases.extend(cases)

    if not all_cases:
        raise ValueError(
            "Missing tests. Rebuild dataset with "
            "scripts/prepare_code_contests_dataset.py --keep-metadata."
        )
    return all_cases, per_field_counts


def sanitize_generated_code(text: str) -> str:
    code = text.strip()
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        code = fenced[0].strip()

    cleanup_patterns = [
        # Qwen/DeepSeek style thinking blocks.
        (r"<think>.*?</think>", " "),
        # Generic analysis/reasoning tags.
        (r"<analysis>.*?</analysis>", " "),
        (r"<reasoning>.*?</reasoning>", " "),
        # Common chat wrapper tokens.
        (r"<\|im_start\|>assistant", " "),
        (r"<\|im_end\|>", " "),
        (r"<\|assistant\|>", " "),
        (r"<\|end\|>", " "),
    ]
    for pattern, replacement in cleanup_patterns:
        code = re.sub(pattern, replacement, code, flags=re.DOTALL | re.IGNORECASE)

    code = code.strip()
    if not code:
        return code

    # If full text is still not valid Python, try dropping non-code prefixes.
    try:
        compile(code, "<generated>", "exec")
        return code
    except SyntaxError:
        pass

    lines = code.splitlines()
    start_idx = 0
    code_start = re.compile(
        r"^\s*(?:"
        r"from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|@|if\s+__name__\s*==|"
        r"for\s+|while\s+|try:|with\s+|return\b|print\(|[A-Za-z_]\w*\s*=|#)"
    )
    for idx, line in enumerate(lines):
        if code_start.search(line):
            start_idx = idx
            break
    trimmed = "\n".join(lines[start_idx:]).strip()
    return trimmed or code


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


@contextlib.contextmanager
def _redirect_stdio(
    new_stdin: io.TextIOWrapper, new_stdout: io.TextIOWrapper, new_stderr: io.TextIOWrapper
):
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdin = new_stdin
        sys.stdout = new_stdout
        sys.stderr = new_stderr
        yield
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _normalize_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines).strip()


def _outputs_match(actual: str, expected: str, compare_mode: str) -> bool:
    if compare_mode == "tokens":
        return actual.split() == expected.split()
    return _normalize_lines(actual) == _normalize_lines(expected)


def _exec_worker(payload: dict[str, Any], queue: mp.Queue) -> None:
    code = payload["code"]
    test_cases = payload["test_cases"]
    memory_limit_mb = payload["memory_limit_mb"]
    compare_mode = payload["compare_mode"]
    _set_memory_limit(memory_limit_mb)

    try:
        compiled_code = compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        queue.put(
            {
                "passed": False,
                "error_type": "syntax",
                "error_message": f"{exc.msg} (line {exc.lineno})",
                "failed_test_index": 0,
            }
        )
        return

    try:
        for idx, (stdin_text, expected_output) in enumerate(test_cases):
            namespace: dict[str, Any] = {"__name__": "__main__"}
            stdin_bytes = io.BytesIO(stdin_text.encode("utf-8"))
            stdout_bytes = io.BytesIO()
            stderr_bytes = io.BytesIO()
            stdin_capture = io.TextIOWrapper(stdin_bytes, encoding="utf-8")
            stdout_capture = io.TextIOWrapper(stdout_bytes, encoding="utf-8")
            stderr_capture = io.TextIOWrapper(stderr_bytes, encoding="utf-8")

            with _redirect_stdio(stdin_capture, stdout_capture, stderr_capture):
                exec(compiled_code, namespace, namespace)

            stdout_capture.flush()
            actual_output = stdout_bytes.getvalue().decode("utf-8", errors="ignore")
            if not _outputs_match(actual_output, expected_output, compare_mode):
                queue.put(
                    {
                        "passed": False,
                        "error_type": "wrong_answer",
                        "error_message": "Output mismatch",
                        "failed_test_index": idx,
                        "actual_output": actual_output[:1000],
                        "expected_output": expected_output[:1000],
                    }
                )
                return

        queue.put(
            {
                "passed": True,
                "error_type": None,
                "error_message": None,
                "failed_test_index": None,
            }
        )
    except SystemExit as exc:
        queue.put(
            {
                "passed": False,
                "error_type": "runtime",
                "error_message": f"SystemExit: {exc}",
                "failed_test_index": None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "passed": False,
                "error_type": "runtime",
                "error_message": f"{type(exc).__name__}: {exc}",
                "failed_test_index": None,
            }
        )


def run_code_with_timeout(
    code: str,
    test_cases: list[tuple[str, str]],
    timeout_sec: float,
    memory_limit_mb: int,
    compare_mode: str,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    payload = {
        "code": code,
        "test_cases": test_cases,
        "memory_limit_mb": memory_limit_mb,
        "compare_mode": compare_mode,
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
            "failed_test_index": None,
        }

    if not queue.empty():
        return queue.get()

    return {
        "passed": False,
        "error_type": "runtime",
        "error_message": "No result returned from execution worker",
        "failed_test_index": None,
    }


def evaluate_code_contests(
    dataset: Any,
    responses: list[str],
    output_file: str,
    summary_file: str,
    timeout_sec: float,
    memory_limit_mb: int,
    test_fields: list[str],
    max_tests_per_field: int,
    compare_mode: str,
    debug: bool,
) -> dict[str, Any]:
    passed = 0
    total = len(dataset)
    error_counts: dict[str, int] = {
        "wrong_answer": 0,
        "runtime": 0,
        "syntax": 0,
        "timeout": 0,
        "none": 0,
    }

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, (example, generated) in enumerate(zip(dataset, responses)):
            task_name = example.get("name", f"row_{idx}")
            test_cases, per_field_counts = get_test_cases(
                example, test_fields, max_tests_per_field
            )

            generated = generated or ""
            generated_code = sanitize_generated_code(generated)
            result = run_code_with_timeout(
                code=generated_code,
                test_cases=test_cases,
                timeout_sec=timeout_sec,
                memory_limit_mb=memory_limit_mb,
                compare_mode=compare_mode,
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
                "task_name": task_name,
                "passed": result["passed"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "failed_test_index": result["failed_test_index"],
                "generated": generated,
                "generated_code": generated_code,
                "num_tests": len(test_cases),
                "test_counts": per_field_counts,
                "compare_mode": compare_mode,
            }
            if "actual_output" in result:
                row["actual_output"] = result["actual_output"]
            if "expected_output" in result:
                row["expected_output"] = result["expected_output"]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if debug:
                print(
                    f"[{idx + 1}/{total}] name={task_name} "
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
        "test_fields": test_fields,
        "max_tests_per_field": max_tests_per_field,
        "compare_mode": compare_mode,
        "output_file": output_file,
    }
    with open(summary_file, "w", encoding="utf-8") as summary_f:
        json.dump(summary, summary_f, indent=2)
    return summary


def generate_responses(dataset: Any, args: argparse.Namespace) -> list[str]:
    from vllm import LLM, SamplingParams

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
        prompts.append(
            format_conversation(
                full_messages,
                tokenizer,
                model_name=args.original_model_name,
                enable_thinking=args.enable_thinking,
            )
        )

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
    return responses


def reference_responses(dataset: Any) -> list[str]:
    responses = []
    for example in dataset:
        responses.append(get_reference_solution(example))
    return responses


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    if not args.input_file.endswith(".jsonl"):
        raise ValueError("Only JSONL files are supported.")

    dataset = load_dataset("json", data_files=args.input_file)["train"]
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print("=== CodeContests pass@1 Evaluation ===")
    print(f"Mode: {args.mode}")
    print(f"Input file: {args.input_file}")
    print(f"Examples: {len(dataset)}")
    print(f"Timeout: {args.timeout_sec}s")
    print(f"Test fields: {args.test_fields}")

    if args.mode == "generated":
        print(f"Model: {args.model_name}")
        if "qwen3" in args.original_model_name.lower():
            print(f"Qwen3 thinking enabled: {args.enable_thinking}")
        responses = generate_responses(dataset, args)
    else:
        print("Using reference solutions from input dataset rows.")
        responses = reference_responses(dataset)

    summary = evaluate_code_contests(
        dataset=dataset,
        responses=responses,
        output_file=args.output_file,
        summary_file=args.summary_file,
        timeout_sec=args.timeout_sec,
        memory_limit_mb=args.memory_limit_mb,
        test_fields=args.test_fields,
        max_tests_per_field=args.max_tests_per_field,
        compare_mode=args.compare_mode,
        debug=args.debug,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
