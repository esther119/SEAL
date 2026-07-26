"""APPS dataset loading (parquet branch) + test-case filtering.

The `codeparrot/apps` script loader is dead on modern `datasets`, so we load the
auto-converted parquet branch directly (see PLAN.md). File order is preserved —
SEAL iterates its train file in order with no shuffling, and we replicate that.

Row fields (codeparrot/apps): problem_id, question, solutions (json str),
input_output (json str: {inputs, outputs[, fn_name]}), difficulty, url,
starter_code.
"""
import json
import sys

# Some APPS test cases hold 9000+ digit integers; Python >=3.11 caps int<->str
# conversion at 4300 digits by default, which breaks json parsing of those.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)  # 0 = unlimited

PARQUET_BASE = ("https://huggingface.co/datasets/codeparrot/apps/resolve/"
                "refs%2Fconvert%2Fparquet/all")
SHARDS = {"train": ["0000"], "test": ["0000", "0001"]}


def load_apps(split="train"):
    """APPS split in file order. Returns a `datasets.Dataset`."""
    from datasets import load_dataset
    files = [f"{PARQUET_BASE}/{split}/{s}.parquet" for s in SHARDS[split]]
    # data_files parquet loads always expose a single "train" split
    return load_dataset("parquet", data_files=files, split="train")


def parse_tests(row):
    """Parse a row's input_output into {fn_name, inputs, outputs, n_tests}.

    Returns None for missing/empty/broken test cases (a known APPS defect —
    the caller must filter these out and log them, per PLAN.md Stage 2).
    """
    raw = row.get("input_output") or ""
    if not raw.strip():
        return None
    try:
        io = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(io, dict):
        return None
    ins, outs = io.get("inputs"), io.get("outputs")
    if not ins or not outs:
        return None
    n = min(len(ins), len(outs))
    if n == 0:
        return None
    return {"fn_name": io.get("fn_name") or None,
            "inputs": ins[:n], "outputs": outs[:n], "n_tests": n}


def build_instruction(row, tests):
    """User-turn instruction for one APPS problem (mirrors v1 MBPP phrasing).

    stdin/stdout problems ask for a full program; call-based (fn_name) problems
    ask to complete the function, including the starter code when present.
    """
    question = (row.get("question") or "").strip()
    starter = (row.get("starter_code") or "").strip()
    if tests and tests["fn_name"]:
        instr = ("Please solve the following coding problem. You should think "
                 "step-by-step, then give the final Python solution.\n"
                 f"Complete the function `{tests['fn_name']}`. "
                 "Put your final solution in a ```python code block.\n"
                 "Problem:\n" + question)
        if starter:
            instr += "\nStarter code:\n```python\n" + starter + "\n```"
    else:
        instr = ("Please solve the following coding problem. You should think "
                 "step-by-step, then give the final Python program.\n"
                 "Your program must read from standard input and print the "
                 "answer to standard output. "
                 "Put your final program in a ```python code block.\n"
                 "Problem:\n" + question)
    return instr
