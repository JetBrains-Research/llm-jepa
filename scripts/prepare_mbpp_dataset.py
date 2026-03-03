#!/usr/bin/env python3
"""Convert MBPP from Hugging Face into this repo's JSONL chat format."""

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MBPP train/test JSONL files in gsm8k-style messages format."
    )
    parser.add_argument(
        "--dataset",
        default="google-research-datasets/mbpp",
        help="Hugging Face dataset path (default: google-research-datasets/mbpp).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config name (for example: sanitized, full).",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train"],
        help="One or more MBPP splits to merge into mbpp_train.jsonl (default: train).",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="MBPP split to use for mbpp_test.jsonl (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Directory where mbpp_train.jsonl and mbpp_test.jsonl are written.",
    )
    parser.add_argument(
        "--train-file",
        default="mbpp_train.jsonl",
        help="Output train JSONL filename (default: mbpp_train.jsonl).",
    )
    parser.add_argument(
        "--test-file",
        default="mbpp_test.jsonl",
        help="Output test JSONL filename (default: mbpp_test.jsonl).",
    )
    parser.add_argument(
        "--system-prompt",
        default="Solve the Python task. Return only Python code.",
        help="System message content used for every sample.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Append test setup and tests to the user prompt.",
    )
    return parser.parse_args()


def load_mbpp_dataset(dataset_name: str, config: str | None) -> DatasetDict:
    if config:
        return load_dataset(dataset_name, config)

    try:
        return load_dataset(dataset_name)
    except Exception:
        for fallback in ("sanitized", "full"):
            try:
                return load_dataset(dataset_name, fallback)
            except Exception:
                continue
        raise


def require_field(example: dict, field_name: str) -> str:
    value = example.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty '{field_name}' field in MBPP example.")
    return value


def build_user_prompt(example: dict, include_tests: bool) -> str:
    prompt = require_field(example, "text")
    if not include_tests:
        return prompt

    parts = [prompt]
    setup = example.get("test_setup_code")
    tests = example.get("test_list")

    if isinstance(setup, str) and setup.strip():
        parts.append("Test setup:")
        parts.append(setup.strip())

    if isinstance(tests, list) and tests:
        parts.append("Tests:")
        parts.extend(f"- {t}" for t in tests if isinstance(t, str) and t.strip())

    return "\n\n".join(parts)


def to_messages(example: dict, system_prompt: str, include_tests: bool) -> dict:
    user_prompt = build_user_prompt(example, include_tests)
    assistant_code = require_field(example, "code")
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_code},
        ]
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def merge_splits(dataset: DatasetDict, split_names: list[str]) -> Dataset:
    merged = []
    for split_name in split_names:
        if split_name not in dataset:
            raise ValueError(
                f"Split '{split_name}' not found. Available splits: {sorted(dataset.keys())}"
            )
        merged.extend(dataset[split_name])
    return Dataset.from_list(merged)


def main() -> None:
    args = parse_args()
    dataset = load_mbpp_dataset(args.dataset, args.config)

    train_split = merge_splits(dataset, args.train_splits)
    if args.test_split not in dataset:
        raise ValueError(
            f"Split '{args.test_split}' not found. Available splits: {sorted(dataset.keys())}"
        )
    test_split = dataset[args.test_split]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / args.train_file
    test_path = output_dir / args.test_file

    train_rows = (
        to_messages(example, args.system_prompt, args.include_tests)
        for example in train_split
    )
    test_rows = (
        to_messages(example, args.system_prompt, args.include_tests)
        for example in test_split
    )

    train_count = write_jsonl(train_path, train_rows)
    test_count = write_jsonl(test_path, test_rows)

    print(f"Wrote {train_count} rows to {train_path}")
    print(f"Wrote {test_count} rows to {test_path}")


if __name__ == "__main__":
    main()
