"""Stage 1 (logic) — LogiQA 2.0 trace generation (vLLM, greedy) + scoring.

Mirrors ``apps/gen_apps_vllm.py`` with the APPS loader/scorer swapped for LogiQA
and one LogiQA-specific addition: an ENGLISH-ONLY pool filter. Everything else —
greedy n=1 / temp 0 / max_tokens 10000 / single BOS, file order with no
shuffling, chunked generate+score, greedy accumulation until BOTH pools hold
``--target``, crash-tolerant resume with output repair, config-drift guard — is
identical to the APPS generator.

WHY THE ENGLISH FILTER: this model reasons in Chinese on ~50% of LogiQA
problems, and English keyword lists tag 0% of those segments. Unfiltered, the
reflection pool would be English and the execution pool largely Chinese, so the
layer-20 contrast would encode an English-vs-Chinese direction. Non-English
traces are generated, logged to ``skipped.jsonl`` with reason ``non_english``,
and excluded from both pools. The committed train snapshot is already
English-only; the filter is a safety net on generated CoTs.

Traces are deliberately NOT deduplicated (same as APPS).

Emits the same files ``hidden_analysis.py`` consumes:
  <save_dir>/evaluated_traces.jsonl
  <save_dir>/data.jsonl
  <save_dir>/skipped.jsonl
  <save_dir>/gen_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from logic.data import (
    build_instruction,
    cjk_fraction,
    is_english_trace,
    is_usable,
    load_train_logic,
    score_generation,
)

TRAIN_SET = "train_logic"

# Args that change trace content or pass/fail labels: a resumed run must not
# silently mix two of these configurations in one pool.
LABEL_ARGS = (
    "model_name_or_path",
    "split",
    "max_tokens",
    "max_prompt_tokens",
    "max_cjk",
    "train_path",
)


def trim_output(output: str) -> str:
    # Upstream eval_MATH_vllm.trim_output verbatim (same as the APPS generator).
    for prefix in ["Answer the following question", "Question:", "Comment:"]:
        if prefix in output:
            output = output.split(prefix)[0]
    return output


def _read_jsonl_tolerant(path: str):
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


def repair_outputs(eval_path: str, data_path: str):
    """Truncate evaluated_traces/data JSONL to their longest aligned prefix."""
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
        print(
            "[gen_logiqa] repaired misaligned/corrupt output files -> "
            f"{k} aligned rows (had eval={len(ev)} data={len(da)})"
        )
    return ev[:k]


def build_prompts(rows, tokenizer, remove_bos=True):
    prompts = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_instruction(row)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if remove_bos and tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token) :]
        prompts.append(prompt)
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate greedy LogiQA 2.0 traces for S_logic / S_general."
    )
    ap.add_argument(
        "--model_name_or_path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    ap.add_argument("--save_dir", required=True)
    ap.add_argument(
        "--train_path",
        default=None,
        help="Path to train_logic.jsonl (default: data/LogiQA2.0/train_logic.jsonl).",
    )
    ap.add_argument(
        "--split",
        default="train",
        help="Recorded in provenance only; rows come from --train_path.",
    )
    ap.add_argument("--max_tokens", type=int, default=10000)
    ap.add_argument(
        "--target",
        type=int,
        default=500,
        help="Stop once BOTH pools (correct/incorrect) hold this many.",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=250,
        help="Problems generated+scored per vLLM round.",
    )
    ap.add_argument(
        "--max_problems",
        type=int,
        default=0,
        help="Hard cap on problems attempted (0 = whole snapshot).",
    )
    ap.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=4096,
        help="Skip problems whose prompt exceeds this (logged).",
    )
    ap.add_argument(
        "--max_cjk",
        type=float,
        default=0.10,
        help="Max fraction of Chinese thought segments for a trace to count "
        "toward a pool. 1.0 disables the filter.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip problem_ids already present in evaluated_traces.jsonl.",
    )
    ap.add_argument(
        "--force_config",
        action="store_true",
        help="Resume even if label-relevant args differ (NOT recommended).",
    )
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_train_logic(args.train_path)
    n_total = len(ds) if not args.max_problems else min(args.max_problems, len(ds))
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)

    eval_path = os.path.join(args.save_dir, "evaluated_traces.jsonl")
    data_path = os.path.join(args.save_dir, "data.jsonl")
    skip_path = os.path.join(args.save_dir, "skipped.jsonl")

    done_ids, skipped_ids, n_correct, n_incorrect = set(), set(), 0, 0
    if args.resume and os.path.exists(eval_path):
        cfg_path = os.path.join(args.save_dir, "gen_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                old = json.load(f)
            diffs = [
                (k, old[k], getattr(args, k))
                for k in LABEL_ARGS
                if k in old and old[k] != getattr(args, k)
            ]
            if old.get("train_set", TRAIN_SET) != TRAIN_SET:
                diffs.append(("train_set", old["train_set"], TRAIN_SET))
            if diffs and not args.force_config:
                raise SystemExit(
                    "[gen_logiqa] resume refused — label-relevant args changed "
                    f"vs gen_config.json: {diffs}. A pool must not mix scoring "
                    "configurations; rerun with matching args or --force_config."
                )
        rows = repair_outputs(eval_path, data_path)
        for r in rows:
            done_ids.add(r["problem_id"])
            if r["all_eval"][0]:
                n_correct += 1
            else:
                n_incorrect += 1
        skipped_rows, _ = _read_jsonl_tolerant(skip_path)
        skipped_ids = {r["problem_id"] for r in skipped_rows}
        print(
            f"[gen_logiqa] resume: {len(done_ids)} pooled "
            f"(correct={n_correct} incorrect={n_incorrect})"
        )
        if n_correct >= args.target and n_incorrect >= args.target:
            print(
                f"[gen_logiqa] DONE (already reached target): "
                f"correct={n_correct} incorrect={n_incorrect}"
            )
            write_path = os.path.join(args.save_dir, "gen_config.json")
            with open(write_path, "w") as f:
                json.dump(
                    {
                        **vars(args),
                        "train_set": TRAIN_SET,
                        "status": "finished",
                        "n_correct": n_correct,
                        "n_incorrect": n_incorrect,
                        "reached_target": True,
                        "resumed_already_done": True,
                    },
                    f,
                    indent=2,
                )
            print(f"[gen_logiqa] -> {eval_path} + data.jsonl + gen_config.json")
            return
    elif not args.resume:
        for p in (eval_path, data_path, skip_path):
            if os.path.exists(p):
                raise SystemExit(
                    f"{p} exists — pass --resume to continue, "
                    "or remove it for a fresh run"
                )

    pending, prefilter_skips = [], []
    for i in range(n_total):
        row = ds[i]
        if row["problem_id"] in done_ids or row["problem_id"] in skipped_ids:
            continue
        if not is_usable(row):
            prefilter_skips.append(
                {"problem_id": row["problem_id"], "reason": "empty_row"}
            )
            continue
        pending.append(i)
    print(
        f"[gen_logiqa] {args.split}: {n_total} problems | "
        f"{len(prefilter_skips)} unusable (empty) | {len(pending)} pending"
    )

    # TP=1: 1.5B never needs sharding, and TP>1 crashes when GPU count does not
    # divide the 12 attention heads.
    llm = LLM(
        model=args.model_name_or_path,
        swap_space=16,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=1,
        max_model_len=args.max_tokens + args.max_prompt_tokens + 16,
    )
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

    flush([], [], prefilter_skips)
    n_attempted = len(done_ids)
    n_nonenglish = 0
    example_prompt_saved = os.path.exists(
        os.path.join(args.save_dir, "example_prompt.txt")
    )

    def write_config(**extra):
        with open(os.path.join(args.save_dir, "gen_config.json"), "w") as f:
            json.dump(
                {
                    **vars(args),
                    "train_set": TRAIN_SET,
                    "decoding": {
                        "temperature": 0,
                        "n": 1,
                        "max_tokens": args.max_tokens,
                        "remove_bos": True,
                        "use_chat_format": True,
                    },
                    "english_filter": {
                        "max_cjk": args.max_cjk,
                        "n_excluded_non_english": n_nonenglish,
                    },
                    **extra,
                },
                f,
                indent=2,
            )

    write_config(status="running")

    for c0 in range(0, len(pending), args.chunk_size):
        if n_correct >= args.target and n_incorrect >= args.target:
            break
        rows = [ds[i] for i in pending[c0 : c0 + args.chunk_size]]
        prompts = build_prompts(rows, tok)

        kept, chunk_skips = [], []
        for row, prompt in zip(rows, prompts):
            n_tok = len(tok.encode(prompt, add_special_tokens=False))
            if n_tok > args.max_prompt_tokens:
                chunk_skips.append(
                    {
                        "problem_id": row["problem_id"],
                        "reason": f"prompt_too_long_{n_tok}",
                    }
                )
            else:
                kept.append((row, prompt))
        if not kept:
            flush([], [], chunk_skips)
            continue
        rows, prompts = map(list, zip(*kept))

        if not example_prompt_saved:
            with open(os.path.join(args.save_dir, "example_prompt.txt"), "w") as f:
                f.write(prompts[0])
            example_prompt_saved = True

        outputs = llm.generate(prompts=prompts, sampling_params=sampling)
        gens = [trim_output(o.outputs[0].text) for o in outputs]

        rows_out, data_out = [], []
        for row, prompt, gen in zip(rows, prompts, gens):
            cjk = cjk_fraction(gen)
            if not is_english_trace(gen, args.max_cjk):
                n_nonenglish += 1
                chunk_skips.append(
                    {
                        "problem_id": row["problem_id"],
                        "reason": "non_english",
                        "cjk_fraction": round(cjk, 3),
                    }
                )
                continue
            res = score_generation(gen, row)
            ok = bool(res["passed"])
            n_correct += ok
            n_incorrect += not ok
            rows_out.append(
                {
                    "prompt": prompt,
                    "problem": row["question"],
                    "model_generation": [gen],
                    "all_eval": [ok],
                    "answer": "",
                    "problem_id": row["problem_id"],
                    "split": args.split,
                    "pred": res["pred"],
                    "gt": res["gt"],
                    "cjk_fraction": round(cjk, 3),
                }
            )
            data_out.append(
                {
                    "problem": row["question"],
                    "level": "logic",
                    "problem_id": row["problem_id"],
                    "split": args.split,
                }
            )
        flush(rows_out, data_out, chunk_skips)
        n_attempted += len(rows_out)
        print(
            f"[gen_logiqa] pooled {n_attempted} | "
            f"correct {n_correct}/{args.target} | "
            f"incorrect {n_incorrect}/{args.target} | "
            f"non-english excluded {n_nonenglish}",
            flush=True,
        )

    done = n_correct >= args.target and n_incorrect >= args.target
    print(
        f"[gen_logiqa] {'DONE' if done else 'EXHAUSTED SPLIT'}: "
        f"correct={n_correct} incorrect={n_incorrect} pooled={n_attempted} "
        f"non_english_excluded={n_nonenglish}"
    )

    write_config(
        status="finished",
        n_attempted=n_attempted,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        reached_target=done,
    )
    print(f"[gen_logiqa] -> {eval_path} + data.jsonl + gen_config.json")
    if not done:
        # Hard-fail so the orchestrator (set -e) does not build 0_500 pools
        # from under-filled data with misleading names.
        sys.exit(2)


if __name__ == "__main__":
    main()
