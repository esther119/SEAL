"""Generate MBPP reasoning traces with vLLM (fast) and score them by running the
MBPP tests, emitting the SAME files SEAL's hidden_analysis.py already consumes:
  <save_dir>/math_eval.jsonl   rows: {prompt, problem, model_generation:[gen], all_eval:[bool], answer}
  <save_dir>/data.jsonl        rows: {problem, level}

So the v_code MBPP steering vector uses SEAL's own pipeline unchanged downstream:
  gen_mbpp_vllm.py  ->  hidden_analysis.py  ->  vector_generation.py
(mirrors eval_MATH_vllm.py's LLM/SamplingParams/chat-template/remove_bos setup).
"""
import argparse, json, os, re, subprocess, sys, signal
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset


def extract_code(gen):
    # Prefer the FINAL answer after </think>; only look for fenced blocks there, so a
    # scratch ```python snippet inside the reasoning trace isn't picked over the answer.
    ans = gen.split("</think>", 1)[1] if "</think>" in gen else gen
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", ans, re.DOTALL)
    return blocks[-1].strip() if blocks else ans.strip()


def passes(gen, tests, setup="", timeout=6):
    code = extract_code(gen)
    if not code or not tests:
        return False
    # MBPP contract: solution code must come BEFORE test_setup_code (setup references
    # classes/objects the solution defines).
    script = code + "\n" + (setup or "") + "\n" + "\n".join(tests)
    try:
        p = subprocess.Popen([sys.executable, "-I", "-c", script], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        return False
    try:
        return p.wait(timeout=timeout) == 0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            p.kill()          # ensure the direct child is signalled even if getpgid raced
        except Exception:
            pass
        try:
            p.wait(timeout=5)  # and reaped
        except Exception:
            pass
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--max_examples", type=int, default=374)
    ap.add_argument("--use_chat_format", action="store_true", default=True)
    ap.add_argument("--remove_bos", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # MBPP is parquet (no loading script) -> no trust_remote_code on any datasets version.
    # Fall back across config/split names for cross-version resilience.
    def _load_mbpp():
        last = None
        for cfg, sp in [("full", "train"), ("full", "test"), ("sanitized", "train"), ("sanitized", "test")]:
            try:
                return load_dataset("google-research-datasets/mbpp", cfg, split=sp)
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"could not load MBPP: {last}")
    ds = _load_mbpp()
    data = [ds[i] for i in range(min(args.max_examples, len(ds)))]

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    prompts = []
    for ex in data:
        problem = ex.get("text") or ex.get("prompt") or ""
        tests = ex.get("test_list", []) or []
        instr = ("Please solve the following Python programming problem. You should think "
                 "step-by-step, then give the final function.\nProblem: " + problem +
                 "\nYour function should pass these tests:\n" + "\n".join(tests))
        prompt = instr
        if args.use_chat_format:
            prompt = tok.apply_chat_template([{"role": "user", "content": instr}],
                                             tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tok.bos_token and prompt.startswith(tok.bos_token):
                prompt = prompt[len(tok.bos_token):]
        prompts.append(prompt)

    # TP=1: a 1.5B model never needs sharding, and TP>1 crashes (12 attn heads not
    # divisible by 8) on multi-GPU pods.
    model = LLM(model=args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.95,
                tensor_parallel_size=1, max_model_len=args.max_tokens + 2000)
    sampling = SamplingParams(n=1, temperature=0, max_tokens=args.max_tokens)
    outputs = model.generate(prompts=prompts, sampling_params=sampling)   # vLLM keeps input order

    math_eval, raw = [], []
    for ex, prompt, out in tqdm(list(zip(data, prompts, outputs)), desc="scoring MBPP tests"):
        gen = out.outputs[0].text
        problem = ex.get("text") or ex.get("prompt") or ""
        ok = passes(gen, ex.get("test_list", []) or [], ex.get("test_setup_code", "") or "")
        math_eval.append({"prompt": prompt, "problem": problem, "model_generation": [gen],
                          "all_eval": [bool(ok)], "answer": ""})
        raw.append({"problem": problem, "level": "1"})

    with open(os.path.join(args.save_dir, "math_eval.jsonl"), "w") as f:
        for r in math_eval:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.save_dir, "data.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    npass = sum(r["all_eval"][0] for r in math_eval)
    print(f"[gen_mbpp] {len(math_eval)} traces | pass {npass} ({npass/len(math_eval)*100:.1f}%) "
          f"| fail {len(math_eval)-npass}")
    print(f"[gen_mbpp] -> {args.save_dir}/math_eval.jsonl + data.jsonl")


if __name__ == "__main__":
    main()
