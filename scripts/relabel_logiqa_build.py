"""Re-label the LogiQA extraction set with the reasoning-model grader.

The original build labelled correct/incorrect with extract_choice_legacy, whose
fallback assigns a letter to generations that never closed </think>. Audited on
the 500+500 that built the shipped vector, 152 "correct" and 234 "incorrect"
traces had never stated an answer at all.

This writes a FINISHED-ONLY copy of the build set: traces that closed </think>
and stated an answer, with all_eval recomputed. Unfinished traces are dropped
entirely rather than pushed into the incorrect pool, so both pools contain only
traces where the model actually completed its reasoning.

hidden_analysis.generate_math_data zips data.jsonl and math_eval.jsonl
positionally and asserts problem equality, so both files are filtered together.

    python scripts/relabel_logiqa_build.py --build_dir <dir> --out_dir <dir>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic_utils import extract_choice, extract_choice_legacy, options_from_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(os.path.join(args.build_dir, "data.jsonl"))]
    ev = [json.loads(l) for l in open(os.path.join(args.build_dir, "math_eval.jsonl"))]
    if len(data) != len(ev):
        raise SystemExit(f"data.jsonl has {len(data)} rows, math_eval.jsonl has {len(ev)}")

    kept_data, kept_ev = [], []
    n_unfinished = n_flipped = 0
    for d, e in zip(data, ev):
        assert d["problem"] == e["problem"], "data/math_eval misaligned"
        gen = e["model_generation"][0]
        gt = int(e["answer"])
        opts = options_from_prompt(e.get("prompt", ""))
        pred = extract_choice(gen, options=opts)
        if pred is None:
            n_unfinished += 1
            continue
        new_eval = [pred == gt]
        if new_eval[0] != bool(e["all_eval"][0]):
            n_flipped += 1
        e = dict(e)
        e["all_eval"] = new_eval
        e["all_pred"] = [pred]
        e["legacy_pred"] = extract_choice_legacy(gen)
        kept_data.append(d)
        kept_ev.append(e)

    n_correct = sum(1 for e in kept_ev if e["all_eval"][0])
    n_incorrect = len(kept_ev) - n_correct

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "data.jsonl"), "w") as f:
        for d in kept_data:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(args.out_dir, "math_eval.jsonl"), "w") as f:
        for e in kept_ev:
            f.write(json.dumps(e) + "\n")

    print(f"input traces           : {len(ev)}")
    print(f"dropped as UNFINISHED  : {n_unfinished} ({n_unfinished/len(ev)*100:.1f}%)")
    print(f"kept (finished)        : {len(kept_ev)}")
    print(f"  labelled correct     : {n_correct}")
    print(f"  labelled incorrect   : {n_incorrect}")
    print(f"labels flipped vs legacy: {n_flipped}")
    print(f"accuracy among finished: {n_correct/len(kept_ev):.3f}")
    print(f"\ncan fill 500+500? correct {'YES' if n_correct >= 500 else 'NO'}"
          f" ({n_correct}), incorrect {'YES' if n_incorrect >= 500 else 'NO'} ({n_incorrect})")
    print(f"wrote -> {args.out_dir}")


if __name__ == "__main__":
    main()
