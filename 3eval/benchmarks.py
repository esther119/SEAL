"""Three benchmarks for the eval: MBPP (pass@1), GSM8K (numeric accuracy),
LogiQA 2.0 (multiple-choice accuracy). Each exposes load/prompt/score with the same
signature so run_eval.py can loop over them."""
import json, re, subprocess, sys


def chat(tok, instr):
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)

# ----------------------------- extractors -----------------------------
def extract_code(gen):
    # Prefer the final answer after </think>; scope fenced-block search there so a
    # scratch snippet inside the reasoning isn't picked over the real function.
    ans = gen.split("</think>", 1)[1] if "</think>" in gen else gen
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", ans, re.DOTALL)
    return blocks[-1].strip() if blocks else ans.strip()


def extract_boxed(gen):
    m = re.findall(r"\\boxed\{([^{}]*)\}", gen)
    return m[-1].strip() if m else None


def extract_last_number(gen):
    o = re.sub(r"(\d),(\d)", r"\1\2", gen)
    nums = re.findall(r"-?\d+\.?\d*", o)
    return nums[-1] if nums else None


def norm_num(x):
    if x is None:
        return None
    x = str(x).strip().rstrip(".").replace(",", "")
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def extract_choice(gen):
    tail = gen.split("</think>")[-1] if "</think>" in gen else gen
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", tail, re.IGNORECASE)
    if not m:
        m = re.findall(r"\b([ABCD])\b", tail)
    return "ABCD".index(m[-1].upper()) if m else None


# ----------------------------- MBPP -----------------------------
def load_mbpp(n):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    return [{"problem": r["text"], "tests": r["test_list"], "setup": r.get("test_setup_code", "") or ""}
            for r in ds.select(range(min(n, len(ds))))]

def prompt_mbpp(ex, tok):
    return chat(tok, "Please solve the following Python programming problem. Think step-by-step, "
                     "then give the final function.\n\nProblem: " + ex["problem"] +
                     "\n\nYour function should pass these tests:\n" + "\n".join(ex["tests"]))

def score_mbpp(gen, ex, timeout=6):
    import os, signal
    code = extract_code(gen)
    if not code:
        return False
    script = code + "\n" + ex["setup"] + "\n" + "\n".join(ex["tests"])   # solution BEFORE setup (MBPP contract)
    try:
        p = subprocess.Popen([sys.executable, "-I", "-c", script],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
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


# ----------------------------- GSM8K -----------------------------
def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for r in ds.select(range(min(n, len(ds)))):
        out.append({"problem": r["question"], "gt": r["answer"].split("####")[1].strip().replace(",", "")})
    return out

def prompt_gsm8k(ex, tok):
    return chat(tok, "Answer the following question. Think step-by-step and put your final answer "
                     "within \\boxed{}.\n\nQuestion: " + ex["problem"])

def score_gsm8k(gen, ex):
    pred = extract_boxed(gen) or extract_last_number(gen)
    return pred is not None and norm_num(pred) == norm_num(ex["gt"])


# ----------------------------- LogiQA 2.0 -----------------------------
def load_logiqa(n):
    from datasets import load_dataset
    # STREAM the test split: LogiQA2.0 has a 63k-row train split, and a non-streaming
    # load materializes ALL splits onto the (slow, network-mounted) HF cache, which hangs.
    ds = load_dataset("datatune/LogiQA2.0", "default", split="test", streaming=True)
    out = []
    for r in ds:
        d = json.loads(r["text"])
        out.append({"passage": d["text"], "question": d["question"],
                    "options": d["options"], "gt": int(d["answer"])})
        if len(out) >= n:
            break
    return out

def prompt_logiqa(ex, tok):
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(ex["options"]))
    return chat(tok, "Read the passage and answer the multiple-choice question. Think step-by-step, "
                     "then end with 'Answer: X' where X is A, B, C, or D.\n\n"
                     f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}\n\nOptions:\n{opts}")

def score_logiqa(gen, ex):
    c = extract_choice(gen)
    return c is not None and c == ex["gt"]


BENCHMARKS = {
    "mbpp":   (load_mbpp,   prompt_mbpp,   score_mbpp),
    "gsm8k":  (load_gsm8k,  prompt_gsm8k,  score_gsm8k),
    "logiqa": (load_logiqa, prompt_logiqa, score_logiqa),
}
