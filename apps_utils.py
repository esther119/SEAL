"""APPS dataset and evaluation support for the SEAL code runners.

The loader follows the maintained parquet conversion because the legacy
``codeparrot/apps`` dataset script no longer works with recent ``datasets``.
Generated programs are graded by SEAL's existing APPS-compatible evaluator.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from typing import Any


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


PARQUET_BASE = (
    "https://huggingface.co/datasets/codeparrot/apps/resolve/"
    "refs%2Fconvert%2Fparquet/all"
)
SHARDS = {"train": ["0000"], "test": ["0000", "0001"]}


def parse_apps_tests(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return parsed tests, or None for APPS rows without usable tests."""
    raw = row.get("input_output") or ""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        tests = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(tests, dict):
        return None
    inputs = tests.get("inputs")
    outputs = tests.get("outputs")
    if not inputs or not outputs:
        return None
    n_tests = min(len(inputs), len(outputs))
    if n_tests == 0:
        return None
    return {
        "fn_name": tests.get("fn_name") or None,
        "inputs": inputs[:n_tests],
        "outputs": outputs[:n_tests],
        "n_tests": n_tests,
    }


def load_apps(
    split: str = "test",
    start: int | None = None,
    max_examples: int | None = None,
    random_sample: bool = False,
    sample_seed: int = 42,
    task_file: str | None = None,
    stratify_by_difficulty: bool = False,
) -> list[dict[str, Any]]:
    """Load usable APPS rows in deterministic order.

    ``task_file`` may point at a JSONL task list from ``v_code-SEAL``. Only its
    ``problem_id`` values are needed; tests are always reloaded from APPS.
    """
    if split not in SHARDS:
        raise ValueError(f"Unsupported APPS split: {split!r}")

    from datasets import load_dataset

    files = [f"{PARQUET_BASE}/{split}/{shard}.parquet" for shard in SHARDS[split]]
    dataset = load_dataset("parquet", data_files=files, split="train")

    requested_ids: list[Any] | None = None
    if task_file:
        with open(task_file) as f:
            requested_ids = [json.loads(line)["problem_id"] for line in f if line.strip()]
        wanted = set(requested_ids)
        by_id = {
            row["problem_id"]: dict(row)
            for row in dataset
            if row["problem_id"] in wanted and parse_apps_tests(row) is not None
        }
        missing = [problem_id for problem_id in requested_ids if problem_id not in by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} APPS task ids are missing or have unusable tests; "
                f"first missing id: {missing[0]}"
            )
        rows = [by_id[problem_id] for problem_id in requested_ids]
    else:
        rows = [dict(row) for row in dataset if parse_apps_tests(row) is not None]

    if random_sample and max_examples is not None and len(rows) > max_examples:
        rng = random.Random(sample_seed)
        if stratify_by_difficulty:
            levels = ("introductory", "interview", "competition")
            buckets = {
                level: [row for row in rows if row.get("difficulty") == level]
                for level in levels
            }
            base, remainder = divmod(max_examples, len(levels))
            selected = []
            for i, level in enumerate(levels):
                count = base + (1 if i < remainder else 0)
                if len(buckets[level]) < count:
                    raise ValueError(
                        f"APPS {split} has only {len(buckets[level])} usable "
                        f"{level} rows; cannot sample {count}"
                    )
                selected.extend(rng.sample(buckets[level], count))
            rows = selected
        else:
            rows = rng.sample(rows, max_examples)
        rows.sort(key=lambda row: row["problem_id"])
    else:
        if start:
            rows = rows[start:]
        if max_examples is not None:
            rows = rows[:max_examples]
    return rows


def build_apps_prompt(row: dict[str, Any]) -> str:
    """Build a reasoning-plus-code instruction for either APPS problem kind."""
    tests = parse_apps_tests(row)
    if tests is None:
        raise ValueError(f"APPS problem {row.get('problem_id')} has no usable tests")

    question = (row.get("question") or "").strip()
    starter = (row.get("starter_code") or "").strip()
    if tests["fn_name"]:
        prompt = (
            "Please solve the following coding problem. You should think "
            "step-by-step, then give the final Python solution.\n"
            f"Complete the function `{tests['fn_name']}`. "
            "Put your final solution in a ```python code block.\n"
            f"Problem:\n{question}"
        )
        if starter:
            prompt += f"\nStarter code:\n```python\n{starter}\n```"
        return prompt

    return (
        "Please solve the following coding problem. You should think "
        "step-by-step, then give the final Python program.\n"
        "Your program must read from standard input and print the answer to "
        "standard output. Put your final program in a ```python code block.\n"
        f"Problem:\n{question}"
    )


def extract_apps_code(generation: str) -> str:
    """Extract the final code, preferring the last fence after ``</think>``."""
    answer = generation.split("</think>", 1)[1] if "</think>" in generation else generation
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", answer, re.DOTALL)
    return blocks[-1].strip() if blocks else answer.strip()


def _attempt_passed(test_results: Any) -> bool:
    return bool(test_results) and all(result > 0 for result in test_results)


def _as_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(item) for item in value)
    return str(value)


def _evaluation_sample(row: dict[str, Any]) -> dict[str, str]:
    """Adapt parquet APPS values to SEAL's legacy APPS evaluator schema."""
    tests = parse_apps_tests(row)
    if tests is None:
        raise ValueError(f"APPS problem {row.get('problem_id')} has no usable tests")

    if tests["fn_name"]:
        inputs = []
        for value in tests["inputs"]:
            arguments = value if isinstance(value, list) else [value]
            inputs.append("\n".join(json.dumps(argument) for argument in arguments))
        outputs = [json.dumps(value) for value in tests["outputs"]]
    else:
        inputs = [_as_text(value) for value in tests["inputs"]]
        outputs = [_as_text(value) for value in tests["outputs"]]

    return {
        "input_output": json.dumps(
            {
                "fn_name": tests["fn_name"],
                "inputs": inputs,
                "outputs": outputs,
            }
        )
    }


def save_apps_results(
    results: list[list[str]],
    rows: list[dict[str, Any]],
    save_dir: str,
    timeout: int = 10,
    workers: int = 12,
) -> float:
    """Grade APPS generations and write predictions, per-task grades and metrics."""
    from code_evaluation import codegen_metrics

    extracted = [
        [extract_apps_code(generation) for generation in attempts]
        for attempts in results
    ]
    samples = [_evaluation_sample(row) for row in rows]
    metrics, raw_results, metadata = codegen_metrics(
        samples,
        extracted,
        num_process_evaluate=workers,
        timeout=timeout,
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "predictions.jsonl"), "w") as f:
        for row, attempts, codes in zip(rows, results, extracted):
            f.write(
                json.dumps(
                    {
                        "problem_id": row["problem_id"],
                        "difficulty": row.get("difficulty"),
                        "problem": row["question"],
                        "model_generation": attempts,
                        "extracted_code": codes,
                    }
                )
                + "\n"
            )

    passed = []
    graded_rows = []
    with open(os.path.join(save_dir, "code_eval.jsonl"), "w") as f:
        for i, (row, attempts, codes, task_metadata) in enumerate(
            zip(rows, results, extracted, metadata)
        ):
            attempt_results = raw_results[i]
            attempt_passed = [_attempt_passed(test_results) for test_results in attempt_results]
            first_passed = attempt_passed[0] if attempt_passed else False
            passed.append(first_passed)
            tests = parse_apps_tests(row)
            kind = "call" if tests and tests["fn_name"] else "stdio"
            graded_rows.append(
                {
                    "passed": first_passed,
                    "difficulty": row.get("difficulty") or "unknown",
                    "kind": kind,
                }
            )
            f.write(
                json.dumps(
                    {
                        "problem_id": row["problem_id"],
                        "difficulty": row.get("difficulty"),
                        "problem": row["question"],
                        "kind": kind,
                        "model_generation": attempts,
                        "extracted_code": codes,
                        "all_eval": attempt_passed,
                        "test_results": attempt_results,
                        "metadata": task_metadata,
                    }
                )
                + "\n"
            )

    pass_at_1 = sum(passed) / len(passed) if passed else 0.0
    breakdown = {}
    for field in ("difficulty", "kind"):
        breakdown[field] = {}
        for value in sorted({row[field] for row in graded_rows}):
            group = [row["passed"] for row in graded_rows if row[field] == value]
            breakdown[field][value] = {
                "pass@1": sum(group) / len(group),
                "n": len(group),
                "num_correct": sum(group),
            }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "pass@1": pass_at_1,
                "n": len(passed),
                "num_correct": sum(passed),
                "breakdown": breakdown,
                "evaluator_metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"pass@1: {pass_at_1}")
    return pass_at_1
