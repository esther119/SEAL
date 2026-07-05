#!/usr/bin/env python3
"""Build unified 60/20/20 train/val/test splits for GSM8K, MBPP, and LogiQA 2.0.

One deterministic scheme for all three cross-domain benchmarks so vector
extraction (train), alpha/layer tuning (val), and reported numbers (test) are
comparable across math / code / logic — per the research plan's rule:
extraction set SUBSET-OF train, tune on val, report on held-out test.

Per benchmark: pool all official splits, shuffle with a fixed seed, cut
60/20/20, and write data/splits/<bench>/{train,val,test}.jsonl in the
benchmark's existing row schema (so loaders/prompts/graders need no change).
A manifest (data/splits/manifest.json) records seed, ratios, and counts.

  python make_splits.py                     # all three benchmarks
  python make_splits.py --benchmarks gsm8k mbpp
  python make_splits.py --logiqa_cap 5000 --overwrite
"""
import argparse
import json
import os
import random

RATIOS = (0.6, 0.2, 0.2)
SPLIT_NAMES = ("train", "val", "test")


def cut(rows, seed):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(rows)
    n_train = int(n * RATIOS[0])
    n_val = int(n * RATIOS[1])
    parts = {
        "train": [rows[i] for i in idx[:n_train]],
        "val": [rows[i] for i in idx[n_train:n_train + n_val]],
        "test": [rows[i] for i in idx[n_train + n_val:]],
    }
    assert sum(len(v) for v in parts.values()) == n
    return parts


# --------------------------------------------------------------------------- #
# Benchmark poolers — each returns (rows, source_description)
# --------------------------------------------------------------------------- #
def pool_gsm8k(args):
    rows = []
    for f in ("data/gsm/train.jsonl", "data/gsm/test.jsonl"):
        with open(f) as fin:
            rows.extend(json.loads(line) for line in fin)
    return rows, "local data/gsm/{train,test}.jsonl pooled"


def pool_mbpp(args):
    from datasets import load_dataset

    rows = []
    for sp in ("train", "test", "validation", "prompt"):
        ds = load_dataset("google-research-datasets/mbpp", "full", split=sp)
        for r in ds:
            rows.append({
                "task_id": r["task_id"],
                "text": r["text"],
                "test_list": list(r["test_list"]),
                "test_setup_code": (r.get("test_setup_code") or ""),
            })
    return rows, "google-research-datasets/mbpp full train+test+validation+prompt pooled"


def pool_logiqa(args):
    from datasets import load_dataset

    rows = []
    skipped = 0
    # Stream: a non-streaming load materializes the 63k-row train split onto the
    # HF cache, which is slow / can hang on network filesystems.
    for sp in ("train", "validation", "test"):
        ds = load_dataset("datatune/LogiQA2.0", "default", split=sp, streaming=True)
        for r in ds:
            try:
                d = json.loads(r["text"])
                rows.append({
                    "passage": d["text"],
                    "question": d["question"],
                    "options": d["options"],
                    "gt": int(d["answer"]),
                })
            except (KeyError, ValueError, TypeError):
                # The dataset mixes two tasks: MRC rows (text/question/options/answer,
                # what we want) and NLI rows (major_premise/conclusion/label), plus a
                # few broken JSON lines. Skip everything that isn't a valid MRC row.
                skipped += 1
    if skipped:
        print(f"[logiqa] skipped {skipped} non-MRC or malformed rows")
    if args.logiqa_cap and len(rows) > args.logiqa_cap:
        # Seeded shuffle BEFORE the cap so the kept subset is itself a uniform
        # sample of the pool, then cut() reshuffles for the split.
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[:args.logiqa_cap]
    return rows, f"datatune/LogiQA2.0 default train+validation+test pooled (cap={args.logiqa_cap})"


POOLERS = {"gsm8k": pool_gsm8k, "mbpp": pool_mbpp, "logiqa": pool_logiqa}


def dedupe_check(parts, key_fn):
    keys = {}
    for split, rows in parts.items():
        for r in rows:
            k = key_fn(r)
            assert k not in keys, f"row appears in both {keys[k]} and {split}: {k!r:.80}"
            keys[k] = split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mbpp", "logiqa"],
                    choices=list(POOLERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logiqa_cap", type=int, default=5000,
                    help="cap the pooled LogiQA rows (63k train is overkill); 0 = no cap")
    ap.add_argument("--out_dir", type=str, default="data/splits")
    ap.add_argument("--overwrite", action="store_true",
                    help="required to regenerate existing split files")
    args = ap.parse_args()

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    key_fns = {
        "gsm8k": lambda r: r["question"],
        "mbpp": lambda r: r["task_id"],
        "logiqa": lambda r: (r["passage"], r["question"]),
    }

    for bench in args.benchmarks:
        bench_dir = os.path.join(args.out_dir, bench)
        existing = [s for s in SPLIT_NAMES
                    if os.path.exists(os.path.join(bench_dir, f"{s}.jsonl"))]
        if existing and not args.overwrite:
            print(f"[{bench}] splits already exist ({', '.join(existing)}); skipping. "
                  f"Use --overwrite to regenerate.")
            continue

        rows, source = POOLERS[bench](args)
        # Dedupe the pool first (LogiQA repeats some (passage, question) pairs across
        # its official splits); otherwise a duplicate could land in two splits and
        # leak train content into test.
        key_fn = key_fns[bench]
        seen, unique_rows = set(), []
        for r in rows:
            k = key_fn(r)
            if k not in seen:
                seen.add(k)
                unique_rows.append(r)
        if len(unique_rows) < len(rows):
            print(f"[{bench}] dropped {len(rows) - len(unique_rows)} duplicate rows before splitting")
        rows = unique_rows
        parts = cut(rows, args.seed)
        dedupe_check(parts, key_fn)

        os.makedirs(bench_dir, exist_ok=True)
        for split, split_rows in parts.items():
            path = os.path.join(bench_dir, f"{split}.jsonl")
            with open(path, "w") as f:
                for r in split_rows:
                    f.write(json.dumps(r) + "\n")

        counts = {s: len(parts[s]) for s in SPLIT_NAMES}
        manifest[bench] = {
            "seed": args.seed,
            "ratios": list(RATIOS),
            "counts": counts,
            "total": len(rows),
            "source": source,
        }
        print(f"[{bench}] total={len(rows)}  " +
              "  ".join(f"{s}={counts[s]}" for s in SPLIT_NAMES))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
