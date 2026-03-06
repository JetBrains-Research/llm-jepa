#!/usr/bin/env python3
"""Convert deepmind/code_contests into this repo's JSONL chat format."""

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CodeContests train/test JSONL files in chat messages format. "
            "Assistant targets are filtered to Python3 solutions only."
        )
    )
    parser.add_argument(
        "--dataset",
        default="deepmind/code_contests",
        help="Hugging Face dataset path (default: deepmind/code_contests).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config/subset name.",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train"],
        help="One or more source splits to merge into train output (default: train).",
    )
    parser.add_argument(
        "--test-split",
        default="valid",
        help="Source split to write as test output (default: valid).",
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
        default="code_contests_train.jsonl",
        help="Output train JSONL filename (default: code_contests_train.jsonl).",
    )
    parser.add_argument(
        "--test-file",
        default="code_contests_test.jsonl",
        help="Output test JSONL filename (default: code_contests_test.jsonl).",
    )
    parser.add_argument(
        "--system-prompt",
        default="Solve the competitive programming problem in Python 3. Return only Python code.",
        help="System message content used for every sample.",
    )
    parser.add_argument(
        "--python-language-id",
        type=int,
        default=3,
        help=(
            "Language id used by deepmind/code_contests for Python3 solutions "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--min-solution-chars",
        type=int,
        default=1,
        help="Minimum solution length to keep a sample (default: 1).",
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


def load_code_contests_dataset(dataset_name: str, config: str | None) -> DatasetDict:
    if config:
        return load_dataset(dataset_name, config)
    return load_dataset(dataset_name)


def merge_splits(dataset: DatasetDict, split_names: list[str]) -> Dataset:
    merged = []
    for split_name in split_names:
        if split_name not in dataset:
            raise ValueError(
                f"Split '{split_name}' not found. Available splits: {sorted(dataset.keys())}"
            )
        merged.extend(dataset[split_name])
    return Dataset.from_list(merged)


def maybe_limit(dataset: Dataset, max_rows: int) -> Dataset:
    if max_rows <= 0:
        return dataset
    return dataset.select(range(min(max_rows, len(dataset))))


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def build_user_prompt(example: dict) -> str | None:
    description = normalize_text(example.get("description"))
    if not description:
        return None
    return description


def pick_python3_solution(
    example: dict, python_language_id: int, min_solution_chars: int
) -> tuple[str | None, int | None, int]:
    solutions = example.get("solutions")
    if not isinstance(solutions, dict):
        return None, None, 0

    languages = solutions.get("language")
    codes = solutions.get("solution")
    if not isinstance(languages, list) or not isinstance(codes, list):
        return None, None, 0

    python_match_count = 0
    first_valid_code = None
    first_valid_idx = None
    for idx, (lang, code) in enumerate(zip(languages, codes)):
        try:
            lang_id = int(lang)
        except (TypeError, ValueError):
            continue
        if lang_id != python_language_id:
            continue

        python_match_count += 1
        code_text = normalize_text(code)
        if first_valid_code is None and len(code_text) >= min_solution_chars:
            first_valid_code = code_text
            first_valid_idx = idx

    if first_valid_code is None:
        return None, None, python_match_count
    return first_valid_code, first_valid_idx, python_match_count


def to_messages(
    example: dict,
    system_prompt: str,
    python_language_id: int,
    min_solution_chars: int,
    keep_metadata: bool,
) -> tuple[dict | None, str | None]:
    user_prompt = build_user_prompt(example)
    if user_prompt is None:
        return None, "missing_description"

    assistant_code, selected_idx, python_match_count = pick_python3_solution(
        example=example,
        python_language_id=python_language_id,
        min_solution_chars=min_solution_chars,
    )
    if assistant_code is None:
        return None, "missing_python3_solution"

    row = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_code},
        ]
    }

    if keep_metadata:
        for key in (
            "name",
            "description",
            "difficulty",
            "cf_rating",
            "cf_tags",
            "public_tests",
            "private_tests",
            "generated_tests",
            "time_limit",
            "memory_limit_bytes",
            "input_file",
            "output_file",
        ):
            if key in example and example[key] is not None:
                row[key] = example[key]
        row["python_language_id"] = python_language_id
        row["selected_solution_index"] = selected_idx
        row["python_solution_candidates"] = python_match_count

    return row, None


def write_transformed_jsonl(
    path: Path,
    rows: Iterable[dict],
    transform_fn,
) -> tuple[int, dict[str, int]]:
    kept = 0
    skipped_reasons: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            converted, reason = transform_fn(row)
            if converted is None:
                if reason is None:
                    reason = "unknown"
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1
    return kept, skipped_reasons


def main() -> None:
    args = parse_args()
    dataset = load_code_contests_dataset(args.dataset, args.config)

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
        example=ex,
        system_prompt=args.system_prompt,
        python_language_id=args.python_language_id,
        min_solution_chars=args.min_solution_chars,
        keep_metadata=args.keep_metadata,
    )
    train_kept, train_skipped = write_transformed_jsonl(train_path, train_split, transform_fn)
    test_kept, test_skipped = write_transformed_jsonl(test_path, test_split, transform_fn)

    print(f"Wrote {train_kept} rows to {train_path}")
    print(f"Wrote {test_kept} rows to {test_path}")
    print(f"Train skipped: {train_skipped}")
    print(f"Test skipped: {test_skipped}")


if __name__ == "__main__":
    main()
