#!/usr/bin/env python3
"""Convert open-r1/codeforces into this repo's JSONL chat format."""

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Codeforces train/test JSONL files in chat messages format."
    )
    parser.add_argument(
        "--dataset",
        default="open-r1/codeforces",
        help="Hugging Face dataset path (default: open-r1/codeforces).",
    )
    parser.add_argument(
        "--config",
        default="verifiable",
        help=(
            "Dataset config/subset name "
            "(default: verifiable; alternatives include default, "
            "verifiable-prompts)."
        ),
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train"],
        help="One or more source splits to merge into train output (default: train).",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Source split to write as test output (default: test).",
    )
    parser.add_argument(
        "--resplit-test-size",
        type=float,
        default=None,
        help=(
            "Optional random test ratio (e.g. 0.1 for 90/10). "
            "When set, re-splits from --resplit-source-splits "
            "(or train+test by default)."
        ),
    )
    parser.add_argument(
        "--resplit-source-splits",
        nargs="+",
        default=None,
        help="Source splits used by --resplit-test-size (default: train splits + test split).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for --resplit-test-size.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Directory where output JSONL files are written.",
    )
    parser.add_argument(
        "--train-file",
        default="codeforces_train.jsonl",
        help="Output train JSONL filename (default: codeforces_train.jsonl).",
    )
    parser.add_argument(
        "--test-file",
        default="codeforces_test.jsonl",
        help="Output test JSONL filename (default: codeforces_test.jsonl).",
    )
    parser.add_argument(
        "--system-prompt",
        default="Solve the competitive programming problem in Python. Return only Python code.",
        help="System message content used for every sample.",
    )
    parser.add_argument(
        "--assistant-fields",
        nargs="+",
        default=["editorial"],
        help=(
            "Candidate fields used as assistant target, in priority order "
            "(default: editorial)."
        ),
    )
    parser.add_argument(
        "--min-assistant-chars",
        type=int,
        default=1,
        help="Minimum assistant target length to keep a sample (default: 1).",
    )
    parser.add_argument(
        "--include-examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include sample examples in user prompt when present (default: true).",
    )
    parser.add_argument(
        "--keep-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep selected raw dataset fields in each output row (default: true).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Limit output train rows (0 = all).",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Limit output test rows (0 = all).",
    )
    return parser.parse_args()


def load_codeforces_dataset(dataset_name: str, config: str | None) -> DatasetDict:
    alias_map = {
        "verifiable-problems": "verifiable",
    }

    if config:
        normalized = alias_map.get(config, config)
        tried = []
        for cfg in [normalized, config]:
            if cfg in tried:
                continue
            tried.append(cfg)
            try:
                return load_dataset(dataset_name, cfg)
            except Exception:
                continue

    try:
        return load_dataset(dataset_name)
    except Exception:
        for fallback in ("verifiable", "verifiable-prompts", "default"):
            try:
                return load_dataset(dataset_name, fallback)
            except Exception:
                continue
        raise


def merge_splits(dataset: DatasetDict, split_names: list[str]) -> Dataset:
    merged = []
    for split_name in split_names:
        if split_name not in dataset:
            raise ValueError(
                f"Split '{split_name}' not found. Available splits: {sorted(dataset.keys())}"
            )
        merged.extend(dataset[split_name])
    return Dataset.from_list(merged)


def first_nonempty_str(example: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_tags(value: object) -> str | None:
    if isinstance(value, list):
        tags = [str(tag).strip() for tag in value if str(tag).strip()]
        if tags:
            return ", ".join(tags)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def normalize_examples(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()

    if not isinstance(value, list) or not value:
        return None

    rows = []
    for i, item in enumerate(value, 1):
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            parts = []
            if isinstance(item.get("input"), str) and item["input"].strip():
                parts.append(f"input:\n{item['input'].strip()}")
            if isinstance(item.get("output"), str) and item["output"].strip():
                parts.append(f"output:\n{item['output'].strip()}")
            if isinstance(item.get("explanation"), str) and item["explanation"].strip():
                parts.append(f"explanation:\n{item['explanation'].strip()}")
            text = "\n".join(parts).strip()
        else:
            text = str(item).strip()
        if text:
            rows.append(f"Example {i}:\n{text}")
    if not rows:
        return None
    return "\n\n".join(rows)


def build_user_prompt(example: dict, include_examples: bool) -> str:
    prompt = first_nonempty_str(example, ("prompt",))
    if prompt:
        return prompt

    title = first_nonempty_str(example, ("title", "name"))
    statement = first_nonempty_str(
        example,
        (
            "description",
            "problem_description",
            "statement",
            "question",
        ),
    )
    input_format = first_nonempty_str(
        example, ("input_format", "input_spec", "input_description")
    )
    output_format = first_nonempty_str(
        example, ("output_format", "output_spec", "output_description")
    )
    constraints = first_nonempty_str(example, ("constraints",))
    tags = normalize_tags(example.get("tags"))
    examples = (
        normalize_examples(example.get("examples"))
        or normalize_examples(example.get("sample_tests"))
        or normalize_examples(example.get("test_cases"))
    )

    sections = []
    if title:
        sections.append(f"Problem: {title}")
    if statement:
        sections.append(f"Statement:\n{statement}")
    if input_format:
        sections.append(f"Input Format:\n{input_format}")
    if output_format:
        sections.append(f"Output Format:\n{output_format}")
    if constraints:
        sections.append(f"Constraints:\n{constraints}")
    if tags:
        sections.append(f"Tags: {tags}")
    if include_examples and examples:
        sections.append(f"Examples:\n{examples}")

    if not sections:
        raise ValueError(
            "Could not build user prompt: expected 'prompt' or problem statement fields."
        )
    return "\n\n".join(sections)


def get_assistant_target(example: dict, fields: list[str], min_chars: int) -> str | None:
    target = first_nonempty_str(example, fields)
    if target is None:
        return None
    if len(target) < min_chars:
        return None
    return target


def to_messages(
    example: dict,
    system_prompt: str,
    assistant_fields: list[str],
    min_assistant_chars: int,
    include_examples: bool,
    keep_metadata: bool,
) -> dict | None:
    assistant_text = get_assistant_target(example, assistant_fields, min_assistant_chars)
    if assistant_text is None:
        return None

    try:
        user_prompt = build_user_prompt(example, include_examples=include_examples)
    except ValueError:
        return None
    row = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]
    }

    if keep_metadata:
        for key in (
            "id",
            "problem_id",
            "contest_id",
            "index",
            "name",
            "title",
            "rating",
            "tags",
            "prompt",
            "description",
            "input_format",
            "output_format",
            "constraints",
            "editorial",
        ):
            if key in example and example[key] is not None:
                row[key] = example[key]

    return row


def write_transformed_jsonl(
    path: Path,
    rows: Dataset,
    transform_fn,
) -> tuple[int, int]:
    kept = 0
    skipped = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            converted = transform_fn(row)
            if converted is None:
                skipped += 1
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1
    return kept, skipped


def maybe_limit(dataset: Dataset, max_rows: int) -> Dataset:
    if max_rows <= 0:
        return dataset
    return dataset.select(range(min(max_rows, len(dataset))))


def main() -> None:
    args = parse_args()
    dataset = load_codeforces_dataset(args.dataset, args.config)

    if args.resplit_test_size is not None:
        if not (0.0 < args.resplit_test_size < 1.0):
            raise ValueError("--resplit-test-size must be in the open interval (0, 1).")

        source_splits = args.resplit_source_splits
        if source_splits is None:
            source_splits = list(dict.fromkeys([*args.train_splits, args.test_split]))
        merged = merge_splits(dataset, source_splits)
        split = merged.train_test_split(
            test_size=args.resplit_test_size, seed=args.split_seed, shuffle=True
        )
        train_split = split["train"]
        test_split = split["test"]
        print(
            f"Random split from {source_splits}: "
            f"{len(train_split)} train / {len(test_split)} test "
            f"(test_size={args.resplit_test_size}, seed={args.split_seed})"
        )
    else:
        train_split = merge_splits(dataset, args.train_splits)
        if args.test_split not in dataset:
            raise ValueError(
                f"Split '{args.test_split}' not found. Available splits: {sorted(dataset.keys())}"
            )
        test_split = dataset[args.test_split]

    train_split = maybe_limit(train_split, args.max_train_rows)
    test_split = maybe_limit(test_split, args.max_test_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / args.train_file
    test_path = output_dir / args.test_file

    transform_fn = lambda ex: to_messages(
        ex,
        system_prompt=args.system_prompt,
        assistant_fields=args.assistant_fields,
        min_assistant_chars=args.min_assistant_chars,
        include_examples=args.include_examples,
        keep_metadata=args.keep_metadata,
    )
    train_kept, train_skipped = write_transformed_jsonl(train_path, train_split, transform_fn)
    test_kept, test_skipped = write_transformed_jsonl(test_path, test_split, transform_fn)

    print(f"Wrote {train_kept} rows to {train_path} (skipped {train_skipped})")
    print(f"Wrote {test_kept} rows to {test_path} (skipped {test_skipped})")


if __name__ == "__main__":
    main()
