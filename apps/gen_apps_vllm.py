"""Stage 1 — APPS trace generation (vLLM, greedy) + interleaved scoring.

SEAL's trace-generation recipe (eval_MATH_vllm.py) transplanted to code:
greedy (temperature 0, n=1), max_tokens 10000, chat template with the BOS
stripped (single BOS — vLLM re-adds one), iterating the APPS **train split in
file order** with no shuffling.

Problems with missing/empty/broken test cases are filtered and logged (known
APPS defect). Generation proceeds in file-order chunks, each chunk is scored
by the Stage-2 harness (apps_scoring.py), and the loop stops as soon as BOTH
pools hold --target traces. Because Stage 3 selects the FIRST 500 of each pool
(--start 0 --sample 500, a deterministic prefix), early stopping cannot change
the selection.

Emits the SAME files SEAL's hidden_analysis.py consumes (plus provenance
extras, which downstream ignores):

  <save_dir>/math_eval.jsonl  {prompt, problem, model_generation:[gen],
                               all_eval:[bool], answer, problem_id, difficulty,
                               split, kind, n_tests, fail}
  <save_dir>/data.jsonl       {problem, level, problem_id, difficulty, split}
  <save_dir>/skipped.jsonl    problems excluded before generation (+reason)
  <save_dir>/gen_config.json  full run record for meta.json provenance
"""
import argparse
import json
import os
import sys

from apps.data import load_apps, parse_tests, build_instruction
from apps import scoring as apps_scoring

# In-repo APPS package always implements the code train set.
TRAIN_SET = "train_code"

# args that change trace content or pass/fail labels: a resumed run must not
# silently mix two of these configurations in one pool
LABEL_ARGS = ("model_name_or_path", "split", "max_tokens", "timeout",
              "max_tests", "max_prompt_tokens")


def _read_jsonl_tolerant(path):
    """Rows up to the first corrupt line (a crash mid-append can leave a
    partial trailing line); returns (rows, sawCorruption)."""
    rows, corrupt = [], False
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    corrupt = True
                    break
    return rows, corrupt


def repair_outputs(eval_path, data_path):
    """Truncate math_eval.jsonl/data.jsonl to their longest aligned prefix.
    flush() appends to two files; a crash between the appends (or mid-line)
    leaves them misaligned, which downstream zip() would silently absorb —
    repair instead, atomically, and return the surviving eval rows."""
    ev, c1 = _read_jsonl_tolerant(eval_path)
    da, c2 = _read_jsonl_tolerant(data_path)
    k = 0
    while k < min(len(ev), len(da)) and ev[k]["problem"] == da[k]["problem"]:
        k += 1
    if c1 or c2 or k != len(ev) or k != len(da):
        for path, rows in ((eval_path, ev[:k]), (data_path, da[:k])):
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, path)
        print(f"[gen_apps] repaired misaligned/corrupt output files -> "
              f"{k} aligned rows (had eval={len(ev)} data={len(da)})")
    return ev[:k]


def trim_output(output):
    # UPSTREAM eval_MATH_vllm.trim_output VERBATIM: SEAL truncates every
    # generation at these markers BEFORE math_eval.jsonl, so the trimmed text
    # is what gets scored AND what enters hidden-state extraction. Dropping it
    # would be a semantic deviation (1/5000 APPS questions even contains a
    # literal 'Comment:').
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def build_prompts(rows, tests_list, tokenizer, remove_bos=True):
    prompts = []
    for row, tests in zip(rows, tests_list):
        instr = build_instruction(row, tests)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}],
            tokenize=False, add_generation_prompt=True)
        if remove_bos and tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path",
                    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_tokens", type=int, default=10000)
    ap.add_argument("--target", type=int, default=500,
                    help="stop once BOTH pools (correct/incorrect) hold this many")
    ap.add_argument("--chunk_size", type=int, default=250,
                    help="problems generated+scored per vLLM round")
    ap.add_argument("--max_problems", type=int, default=0,
                    help="hard cap on problems attempted (0 = whole split)")
    ap.add_argument("--max_prompt_tokens", type=int, default=4096,
                    help="skip problems whose prompt exceeds this (logged)")
    ap.add_argument("--timeout", type=int, default=10, help="seconds per test case")
    ap.add_argument("--max_tests", type=int, default=0,
                    help="cap tests run per problem (0 = all; if set, it is logged "
                         "in gen_config for the meta.json)")
    ap.add_argument("--workers", type=int, default=None, help="scoring parallelism")
    ap.add_argument("--resume", action="store_true",
                    help="skip problem_ids already present in math_eval.jsonl")
    ap.add_argument("--force_config", action="store_true",
                    help="resume even if label-relevant args differ from the "
                         "original run's gen_config.json (NOT recommended)")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_apps(args.split)
    n_total = len(ds) if not args.max_problems else min(args.max_problems, len(ds))
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)

    eval_path = os.path.join(args.save_dir, "math_eval.jsonl")
    data_path = os.path.join(args.save_dir, "data.jsonl")
    skip_path = os.path.join(args.save_dir, "skipped.jsonl")

    done_ids, skipped_ids, n_correct, n_incorrect = set(), set(), 0, 0
    if args.resume and os.path.exists(eval_path):
        cfg_path = os.path.join(args.save_dir, "gen_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                old = json.load(f)
            diffs = [(k, old[k], getattr(args, k)) for k in LABEL_ARGS
                     if k in old and old[k] != getattr(args, k)]
            if old.get("train_set", TRAIN_SET) != TRAIN_SET:
                diffs.append(("train_set", old["train_set"], TRAIN_SET))
            if diffs and not args.force_config:
                raise SystemExit(
                    "[gen_apps] resume refused — label-relevant args changed "
                    f"vs gen_config.json: {diffs}. A pool must not mix scoring "
                    "configurations; rerun with matching args or --force_config.")
        rows = repair_outputs(eval_path, data_path)
        for r in rows:
            done_ids.add(r["problem_id"])
            if r["all_eval"][0]:
                n_correct += 1
            else:
                n_incorrect += 1
        skipped_rows, _ = _read_jsonl_tolerant(skip_path)
        skipped_ids = {r["problem_id"] for r in skipped_rows}
        print(f"[gen_apps] resume: {len(done_ids)} done "
              f"(correct={n_correct} incorrect={n_incorrect})")
    elif not args.resume:
        for p in (eval_path, data_path, skip_path):
            if os.path.exists(p):
                raise SystemExit(f"{p} exists — pass --resume to continue, "
                                 f"or remove it for a fresh run")

    # Filter in file order BEFORE generation: broken tests + oversize prompts.
    # Skipping never reorders the remaining problems, so trace order == file
    # order. Keep only dataset indices (rows re-materialize per chunk — parsed
    # test data for the whole split would hold ~GBs of Python objects).
    usable, skipped = [], []
    for i in range(n_total):
        row = ds[i]
        if row["problem_id"] in done_ids:
            continue
        if parse_tests(row) is None:
            if row["problem_id"] not in skipped_ids:
                skipped.append({"problem_id": row["problem_id"],
                                "difficulty": row["difficulty"],
                                "reason": "no_tests"})
            continue
        usable.append(i)
    print(f"[gen_apps] {args.split}: {n_total} problems | "
          f"{len(skipped)} skipped (no/broken tests) | "
          f"{len(usable)} usable pending")

    # TP=1: 1.5B never needs sharding, and TP>1 crashes on pods whose GPU count
    # doesn't divide the 12 attention heads (v1 lesson). +16 headroom: vLLM
    # re-adds BOS on top of max_prompt_tokens-long prompts.
    llm = LLM(model=args.model_name_or_path, swap_space=16,
              gpu_memory_utilization=0.95, tensor_parallel_size=1,
              max_model_len=args.max_tokens + args.max_prompt_tokens + 16)
    sampling = SamplingParams(n=1, temperature=0, max_tokens=args.max_tokens)

    def flush(rows_out, data_out, skipped_out):
        with open(eval_path, "a") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
        with open(data_path, "a") as f:
            for r in data_out:
                f.write(json.dumps(r) + "\n")
        with open(skip_path, "a") as f:
            for r in skipped_out:
                f.write(json.dumps(r) + "\n")

    flush([], [], skipped)  # record pre-generation skips immediately
    n_attempted = len(done_ids)
    example_prompt_saved = os.path.exists(os.path.join(args.save_dir, "example_prompt.txt"))

    def write_config(**extra):
        with open(os.path.join(args.save_dir, "gen_config.json"), "w") as f:
            json.dump({**vars(args),
                       "train_set": TRAIN_SET,
                       "decoding": {"temperature": 0, "n": 1,
                                    "max_tokens": args.max_tokens,
                                    "remove_bos": True, "use_chat_format": True},
                       **extra}, f, indent=2)

    write_config(status="running")  # provenance survives interrupted runs

    for c0 in range(0, len(usable), args.chunk_size):
        if n_correct >= args.target and n_incorrect >= args.target:
            break
        rows = [ds[i] for i in usable[c0:c0 + args.chunk_size]]
        tests_list = [parse_tests(row) for row in rows]
        prompts = build_prompts(rows, tests_list, tok)

        # drop oversize prompts (would exceed max_model_len with 10k generation)
        kept, chunk_skips = [], []
        for row, tests, prompt in zip(rows, tests_list, prompts):
            n_tok = len(tok.encode(prompt, add_special_tokens=False))
            if n_tok > args.max_prompt_tokens:
                if row["problem_id"] not in skipped_ids:
                    chunk_skips.append({"problem_id": row["problem_id"],
                                        "difficulty": row["difficulty"],
                                        "reason": f"prompt_too_long_{n_tok}"})
            else:
                kept.append((row, tests, prompt))
        if not kept:
            flush([], [], chunk_skips)
            continue
        rows, tests_list, prompts = map(list, zip(*kept))

        if not example_prompt_saved:
            with open(os.path.join(args.save_dir, "example_prompt.txt"), "w") as f:
                f.write(prompts[0])
            example_prompt_saved = True

        outputs = llm.generate(prompts=prompts, sampling_params=sampling)  # keeps order
        gens = [trim_output(o.outputs[0].text) for o in outputs]
        results = apps_scoring.score_many(gens, tests_list, timeout=args.timeout,
                                          max_tests=args.max_tests,
                                          workers=args.workers)

        rows_out, data_out = [], []
        for row, tests, prompt, gen, res in zip(rows, tests_list, prompts, gens, results):
            ok = bool(res["passed"])
            n_correct += ok
            n_incorrect += not ok
            rows_out.append({"prompt": prompt, "problem": row["question"],
                             "model_generation": [gen], "all_eval": [ok],
                             "answer": "", "problem_id": row["problem_id"],
                             "difficulty": row["difficulty"], "split": args.split,
                             "kind": res["kind"], "n_tests": res["n_tests"],
                             "fail": res["fail"]})
            data_out.append({"problem": row["question"], "level": row["difficulty"],
                             "problem_id": row["problem_id"],
                             "difficulty": row["difficulty"], "split": args.split})
        flush(rows_out, data_out, chunk_skips)
        n_attempted += len(rows_out)
        print(f"[gen_apps] attempted {n_attempted} | "
              f"correct {n_correct}/{args.target} | "
              f"incorrect {n_incorrect}/{args.target}", flush=True)

    done = n_correct >= args.target and n_incorrect >= args.target
    print(f"[gen_apps] {'DONE' if done else 'EXHAUSTED SPLIT'}: "
          f"correct={n_correct} incorrect={n_incorrect} attempted={n_attempted}")

    write_config(status="finished", n_attempted=n_attempted,
                 n_correct=n_correct, n_incorrect=n_incorrect,
                 reached_target=done)
    print(f"[gen_apps] -> {eval_path} + data.jsonl + gen_config.json")
    if not done:
        # hard-fail so the orchestrator (set -e) does not build 0_500 pools
        # from under-filled data with misleading names
        sys.exit(2)


if __name__ == "__main__":
    main()
