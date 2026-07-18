"""MBPP support for the SEAL code-eval scripts (Harness A).

Lets `eval_code_vllm.py` (baseline) and `eval_code_steering.py` (steered) target
MBPP with the *same* generation + steering core they already use for
LiveCodeBench. Only the data / prompt / extraction / scoring differ:

  - load:    google-research-datasets/mbpp  (config "full", split "test")
  - prompt:  a chat instruction that elicits a <think> block + a final code block
             (so the SEAL steering trigger on `\\n\\n` inside <think> still fires)
  - extract: take the last ```python``` block after </think>
  - score:   pass@1 — run the function against MBPP's `test_list` in an isolated
             subprocess (hard timeout + process-group kill; never exec in-process)
"""
import json
import os
import re
import signal
import subprocess
import sys


def load_mbpp(split="test", config="full", start=None, max_examples=None):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", config, split=split)
    data = [
        {
            "task_id": r["task_id"],
            "text": r["text"],
            "test_list": list(r["test_list"]),
            "test_setup_code": (r.get("test_setup_code") or ""),
        }
        for r in ds
    ]
    if start:
        data = data[start:]
    if max_examples and len(data) > max_examples:
        data = data[:max_examples]
    return data


MBPP_INSTRUCTION = (
    "You are an expert Python programmer. Solve the following task. "
    "Think step-by-step, then give your final answer as a single self-contained "
    "Python code block that defines the required function.\n\n"
    "Task: {problem}\n\n"
    "Your code must pass these tests:\n{tests}"
)


def build_mbpp_prompt(ex):
    return MBPP_INSTRUCTION.format(
        problem=ex["text"].strip(),
        tests="\n".join(ex["test_list"]),
    )


def extract_mbpp_code(gen):
    # Scope the fenced-block search to after </think> so a scratch snippet inside
    # the reasoning isn't picked over the real final function.
    ans = gen.split("</think>", 1)[1] if "</think>" in gen else gen
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", ans, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return ans.strip()


def _run_one(code, ex, timeout):
    if not code:
        return False
    # Solution BEFORE setup + tests (the MBPP contract).
    script = code + "\n" + ex["test_setup_code"] + "\n" + "\n".join(ex["test_list"])
    try:
        p = subprocess.Popen(
            [sys.executable, "-I", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return False
    try:
        return p.wait(timeout=timeout) == 0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait(timeout=5)
        except Exception:
            pass
        return False


def score_mbpp(codes, data, timeout=6):
    """codes: list[str] extracted solutions; data: list[dict].
    Returns (pass@1: float, per_instance: list[bool])."""
    per = [_run_one(c, ex, timeout) for c, ex in zip(codes, data)]
    passk = sum(per) / len(per) if per else 0.0
    return passk, per


def save_mbpp_results(results, data, save_dir, timeout=6):
    """Score MBPP and write predictions / graded / metrics files.

    results: list of attempts-lists (each is list[str] generations); attempt[0] is used.
    Mirrors the code-eval scripts' output layout so downstream tooling is consistent.
    """
    gens = [attempts[0] if attempts else "" for attempts in results]
    codes = [extract_mbpp_code(g) for g in gens]
    passk, per = score_mbpp(codes, data, timeout=timeout)
    print(f"pass@1: {passk}")

    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "predictions.jsonl"), "w") as f:
        for ex, attempts, code in zip(data, results, codes):
            f.write(json.dumps({
                "task_id": ex["task_id"],
                "problem": ex["text"],
                "tests": ex["test_list"],
                "model_generation": attempts,
                "extracted_code": code,
            }) + "\n")

    with open(os.path.join(save_dir, "code_eval.jsonl"), "w") as f:
        for ex, attempts, code, ok in zip(data, results, codes, per):
            f.write(json.dumps({
                "task_id": ex["task_id"],
                "problem": ex["text"],
                "tests": ex["test_list"],
                "model_generation": attempts,
                "extracted_code": code,
                "all_eval": [bool(ok)],
                "correct": bool(ok),
            }) + "\n")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({"pass@1": passk, "n": len(per), "num_correct": int(sum(per))}, f, indent=2)

    return passk
