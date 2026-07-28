"""Probe how well a keyword set tags LogiQA reasoning steps, and -- unlike a plain
coverage number -- surface WHICH recurring phrases are being missed, ranked by how
many distinct steps each one would actually recover.

Given math_eval.jsonl + data.jsonl from gen_logiqa_vllm.py (tokenizer only, no GPU
needed), this reports:
  1. check/switch/other tag rates for a given --keywords set (the headline number:
     e.g. "math" leaves some fraction of steps in "other" even when they're
     genuinely reflective/transition thoughts, just phrased differently).
  2. a frequency table of word/short-phrase n-grams found in "other"-tagged steps,
     ranked by distinct-step count -- i.e. which candidate phrases would recover the
     most currently-untagged steps if added to KEYWORD_SETS["logic"]. This is what
     lets a keyword addition be justified by "recovers N/n_other steps" instead of
     "I saw this phrase in a couple of examples."
  3. sampled step texts (both tagged and untagged) for manual review: untagged
     samples validate recall candidates in context; tagged samples confirm the
     existing/candidate keywords are catching genuine self-verification/case-
     switching and not incidental substring matches (precision).

Uses hidden_analysis.py's own KEYWORD_SETS and step/think-boundary logic (--keywords
choices come directly from that dict, and the split_id / think_only / check-before-
switch priority below exactly mirrors generate_index) so tagging behavior here is
guaranteed identical to what the real extraction pipeline (hidden_analysis.py ->
vector_generation.py) will do -- generate_index() itself only returns token-position
indices, not step text, so this script re-derives step text alongside the same tags
for review purposes.

Workflow to *derive* KEYWORD_SETS["logic"] (do not invent it):
  1. bash scripts/discover_logic_keywords.sh          # probe gen + baseline coverage
  2. python logic_keyword_coverage.py --data_dir ... --keywords math
  3. Read candidate_phrases_math.json (ranked by distinct-step recovery) and
     review_samples_math.jsonl (untagged samples) -- confirm each high-ranked
     candidate is genuinely reflective/transition language, not coincidental.
  4. Add validated candidates to KEYWORD_SETS["logic"] in hidden_analysis.py.
  5. Re-run this script with --keywords logic; repeat 3-4 until new candidates only
     recover a small marginal fraction of the remaining "other" steps.
  6. Re-extract hidden states with --keywords logic for the real vector build.

  python logic_keyword_coverage.py \\
      --data_dir results/results_for_logic_vectors/LogiQA_train/DeepSeek-R1-Distill-Qwen-1.5B/keyword_probe \\
      --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
      --keywords math
"""
import argparse
import json
import os
import random
import re
from collections import Counter

from transformers import AutoTokenizer

from hidden_analysis import KEYWORD_SETS, generate_math_data

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "this", "that", "it", "to", "of",
    "in", "on", "for", "and", "or", "so", "then", "we", "i", "there", "be", "as",
    "with", "at", "by", "from", "which", "not", "but", "if", "can", "will", "would",
}

# LogiQA task-scaffolding nouns: every reasoning step mentions the passage/options/
# question/argument regardless of whether it's actually a reflective or transition
# step, so a bare mention of these is not evidence of check/switch behavior -- unlike
# _STOPWORDS (generic function words), these are excluded because they're topical,
# not because they're low-content. An n-gram is noise if every non-stopword word in
# it is one of these (so "the passage" / "option c" / "the question is" are dropped,
# but a phrase that pairs one of these with a real content word, e.g. "passage
# contradicts", survives for review).
_TOPICAL_NOISE = {
    "option", "options", "passage", "question", "argument", "conclusion",
    "premise", "statement", "a", "b", "c", "d",
}


def is_topical_noise(gram_words):
    content = [w for w in gram_words if w not in _STOPWORDS]
    return bool(content) and all(w in _TOPICAL_NOISE for w in content)


def split_steps(text, tokenizer, split_id, think_only, keywords):
    """Same step-splitting + check/switch/other tagging as hidden_analysis.generate_index,
    but returns (step_text, tag) pairs instead of token-position indices, since we need
    the actual text for frequency analysis and manual review."""
    kw = KEYWORD_SETS[keywords]
    check_words, check_prefix = kw["check_words"], kw["check_prefix"]
    switch_words, switch_prefix = kw["switch_words"], kw["switch_prefix"]

    tokens = tokenizer.encode(text)
    if think_only:
        think_begin_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        if think_begin_id not in tokens:
            return None  # no <think> block -- distinct from "no tags found"
        start = tokens.index(think_begin_id) + 1
        end = (tokens.index(think_end_id, start) if think_end_id in tokens[start:]
               else len(tokens))
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    steps = []
    for i in range(len(index) - 1):
        step_tokens = think_tokens[index[i] + 1:index[i + 1]]
        step_text = tokenizer.decode(step_tokens).strip(" ").strip("\n")
        if not step_text:
            continue
        if (any(step_text.lower().startswith(p.lower()) for p in check_prefix)
                or any(w.lower() in step_text.lower() for w in check_words)):
            tag = "check"
        elif (any(step_text.lower().startswith(p.lower()) for p in switch_prefix)
                or any(w.lower() in step_text.lower() for w in switch_words)):
            tag = "switch"
        else:
            tag = "other"
        steps.append((step_text, tag))
    return steps


def ngrams_for_step(step_text, ngram_max):
    words = re.findall(r"[a-z']+", step_text.lower())
    grams = set()
    for n in range(1, ngram_max + 1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if n == 1 and gram[0] in _STOPWORDS:
                continue
            if is_topical_noise(gram):
                continue
            grams.add(" ".join(gram))
    return grams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir with math_eval.jsonl (+ data.jsonl)")
    ap.add_argument("--data_path", default=None, help="data.jsonl path (default: <data_dir>/data.jsonl)")
    ap.add_argument("--model_path", required=True, help="only used to load the tokenizer")
    ap.add_argument("--keywords", default="math", choices=sorted(KEYWORD_SETS))
    ap.add_argument("--ngram_max", type=int, default=3, help="max n-gram size for candidate phrases")
    ap.add_argument("--min_step_count", type=int, default=3,
                     help="only report candidate phrases appearing in at least this many distinct 'other' steps")
    ap.add_argument("--top_k", type=int, default=40, help="how many ranked candidate phrases to report")
    ap.add_argument("--sample_size", type=int, default=50,
                     help="how many example step texts to dump per category for manual review")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = args.data_path or os.path.join(args.data_dir, "data.jsonl")
    correct, incorrect = generate_math_data(args.data_dir, data_path)
    examples = correct + incorrect

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vocab = tokenizer.get_vocab()
    split_id = [vocab[t] for t in vocab if "ĊĊ" in t]
    think_only = "deepseek" in args.model_path.lower()

    check_steps, switch_steps, other_steps = [], [], []
    no_think_block = 0

    for ex in examples:
        text = ex["prompt"] + ex["response"]
        steps = split_steps(text, tokenizer, split_id, think_only, args.keywords)
        if steps is None:
            no_think_block += 1
            continue
        for step_text, tag in steps:
            if tag == "check":
                check_steps.append(step_text)
            elif tag == "switch":
                switch_steps.append(step_text)
            else:
                other_steps.append(step_text)

    total = len(check_steps) + len(switch_steps) + len(other_steps)
    print(f"[coverage] keywords={args.keywords!r} examples={len(examples)} "
          f"(no <think> block: {no_think_block})")
    if total == 0:
        print("[coverage] no tagged steps found -- nothing to report")
        return
    print(f"[coverage] steps: check={len(check_steps)} ({len(check_steps)/total*100:.1f}%) "
          f"switch={len(switch_steps)} ({len(switch_steps)/total*100:.1f}%) "
          f"other={len(other_steps)} ({len(other_steps)/total*100:.1f}%) "
          f"| total={total}")

    # Rank candidate phrases by how many DISTINCT "other" steps they'd recover.
    phrase_step_counts = Counter()
    for step_text in other_steps:
        for gram in ngrams_for_step(step_text, args.ngram_max):
            phrase_step_counts[gram] += 1
    ranked = [(phrase, n) for phrase, n in phrase_step_counts.most_common()
              if n >= args.min_step_count][:args.top_k]

    print(f"\n[coverage] top candidate phrases in 'other' steps (>= {args.min_step_count} "
          f"distinct steps), out of {len(other_steps)} untagged steps:")
    for phrase, n in ranked:
        print(f"  {n:5d} steps ({n/len(other_steps)*100:4.1f}% of 'other')  {phrase!r}")

    rng = random.Random(args.seed)

    def sample(steps, k):
        return rng.sample(steps, min(k, len(steps)))

    out = {
        "keywords": args.keywords,
        "counts": {"check": len(check_steps), "switch": len(switch_steps),
                   "other": len(other_steps), "total": total, "no_think_block": no_think_block},
        "rates": {"check": len(check_steps) / total, "switch": len(switch_steps) / total,
                  "other": len(other_steps) / total},
    }
    with open(os.path.join(args.data_dir, f"coverage_{args.keywords}.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(args.data_dir, f"candidate_phrases_{args.keywords}.json"), "w") as f:
        json.dump([{"phrase": p, "distinct_other_steps": n, "pct_of_other": n / len(other_steps)}
                   for p, n in ranked], f, indent=2)

    review_rows = ([{"tag": "other", "step": s} for s in sample(other_steps, args.sample_size)]
                    + [{"tag": "check", "step": s} for s in sample(check_steps, args.sample_size)]
                    + [{"tag": "switch", "step": s} for s in sample(switch_steps, args.sample_size)])
    with open(os.path.join(args.data_dir, f"review_samples_{args.keywords}.jsonl"), "w") as f:
        for row in review_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n[coverage] wrote coverage_{args.keywords}.json, "
          f"candidate_phrases_{args.keywords}.json, review_samples_{args.keywords}.jsonl "
          f"-> {args.data_dir}")


if __name__ == "__main__":
    main()
