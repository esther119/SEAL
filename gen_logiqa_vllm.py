"""Generate LogiQA 2.0 reasoning traces with vLLM (fast) and score them against the
gold answer index, emitting the SAME files SEAL's hidden_analysis.py already consumes:
  <save_dir>/math_eval.jsonl   rows: {prompt, problem, model_generation:[gen], all_eval:[bool], answer}
  <save_dir>/data.jsonl        rows: {problem, level}

So the logic steering vector uses SEAL's own pipeline unchanged downstream:
  gen_logiqa_vllm.py  ->  hidden_analysis.py --keywords logic  ->  vector_generation.py
(mirrors gen_mbpp_vllm.py's LLM/SamplingParams/chat-template/remove_bos setup, and
reuses logic_utils.py's prompt/extraction so Harness A and this generator agree on
what counts as a correct answer.)

CJK filter: LogiQA 2.0 is roughly half untranslated Chinese-script content, which
breaks English keyword tagging downstream in hidden_analysis.py (a step's check/
switch tag search is a substring match against English phrases). Filtering is only
ever applied to this generator's BUILD data (--split train, used for vector
extraction). The eval side defaults to the full, unfiltered official split via
logic_utils.load_logiqa() in eval_MATH_steering.py, but can request the same
filter with --logiqa_english_only (shared is_cjk_heavy, so build and eval agree).
"""
import argparse
import json
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from logic_utils import build_logiqa_prompt, extract_choice, is_cjk_heavy, load_logiqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--split", type=str, default="train",
                     help="HF-native LogiQA2.0 split to stream (train for vector-building "
                          "extraction, test/val only if you have a specific reason to probe them).")
    ap.add_argument("--max_examples", type=int, default=None,
                     help="cap the number of loaded rows (None = all streamed rows)")
    ap.add_argument("--use_chat_format", action="store_true", default=True)
    ap.add_argument("--remove_bos", action="store_true", default=True)
    ap.add_argument("--filter_cjk", action="store_true", default=True,
                     help="Skip CJK-heavy rows (LogiQA 2.0 is ~50%% untranslated Chinese "
                          "content) so the check/switch keyword tagger in hidden_analysis.py "
                          "isn't diluted by text it can never tag. Only affects the traces "
                          "this script generates -- eval elsewhere stays on the full official "
                          "(mixed-language) split.")
    ap.add_argument("--no_filter_cjk", dest="filter_cjk", action="store_false")
    ap.add_argument("--cjk_threshold", type=float, default=0.1,
                     help="fraction of non-Latin-script chars above which a row is skipped")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[gen_logiqa] loading LogiQA2.0 split={args.split!r} (streamed) ...")
    data = load_logiqa(split=args.split)

    if args.filter_cjk:
        before = len(data)
        data = [ex for ex in data if not is_cjk_heavy(ex, args.cjk_threshold)]
        print(f"[gen_logiqa] CJK filter: kept {len(data)}/{before} rows "
              f"({before - len(data)} skipped as CJK-heavy, threshold={args.cjk_threshold})")

    if args.max_examples:
        data = data[:args.max_examples]

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    prompts, problems = [], []
    for ex in data:
        instr = build_logiqa_prompt(ex)
        # Stable id shared between data.jsonl and math_eval.jsonl (hidden_analysis.py
        # asserts these match line-for-line).
        problem = f"{ex['passage'].strip()}\n\n{ex['question'].strip()}"
        prompt = instr
        if args.use_chat_format:
            prompt = tok.apply_chat_template([{"role": "user", "content": instr}],
                                              tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tok.bos_token and prompt.startswith(tok.bos_token):
                prompt = prompt[len(tok.bos_token):]
        prompts.append(prompt)
        problems.append(problem)

    # TP=1: a 1.5B model never needs sharding, and TP>1 crashes (12 attn heads not
    # divisible by 8) on multi-GPU pods.
    model = LLM(model=args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.95,
                tensor_parallel_size=1, max_model_len=args.max_tokens + 2000)
    sampling = SamplingParams(n=1, temperature=0, max_tokens=args.max_tokens)
    outputs = model.generate(prompts=prompts, sampling_params=sampling)  # vLLM keeps input order

    math_eval, raw = [], []
    n_unfinished = 0
    for ex, problem, prompt, out in tqdm(list(zip(data, problems, prompts, outputs)),
                                         desc="scoring LogiQA"):
        gen = out.outputs[0].text
        pred = extract_choice(gen)
        if pred is None:
            # Never closed </think>: the model spent its whole budget reasoning and
            # never stated an answer. That is a FAILED attempt, so it is filed as
            # INCORRECT and kept in the pool -- not dropped.
            #
            # Keeping it matters for two reasons:
            #  1. Alignment. v_math and v_code exclude no traces; every attempt is
            #     either correct or incorrect. Dropping traces here would make the
            #     logic pool a different population from the other domains' pools.
            #  2. Signal. vector_generation pools boundaries across BOTH files into
            #     one check/switch/other contrast -- the correct/incorrect label is
            #     only a sampling control and never enters the formula. A
            #     non-terminating trace is dense, sustained reflection, i.e. exactly
            #     the behaviour the steering vector is meant to capture.
            #
            # The grading fix (extract_choice requiring </think>) is about not
            # FABRICATING an answer where none exists. It is not a reason to remove
            # the trace.
            n_unfinished += 1
        ok = pred is not None and pred == ex["gt"]
        math_eval.append({"prompt": prompt, "problem": problem, "model_generation": [gen],
                           "all_eval": [bool(ok)], "answer": ex["gt"]})
        raw.append({"problem": problem, "level": "1"})

    with open(os.path.join(args.save_dir, "math_eval.jsonl"), "w") as f:
        for r in math_eval:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.save_dir, "data.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    npass = sum(r["all_eval"][0] for r in math_eval)
    total = len(outputs)
    print(f"[gen_logiqa] {total} generated | unfinished (no </think>) {n_unfinished} "
          f"({n_unfinished / total * 100:.1f}%) — kept, filed as incorrect")
    print(f"[gen_logiqa] {len(math_eval)} traces | correct {npass} "
          f"({npass / max(1, len(math_eval)) * 100:.1f}%) | incorrect {len(math_eval) - npass}")
    print(f"[gen_logiqa] -> {args.save_dir}/math_eval.jsonl + data.jsonl")


if __name__ == "__main__":
    main()
