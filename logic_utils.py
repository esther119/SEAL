"""LogiQA 2.0 support for the SEAL MATH-style eval scripts (Harness A).

Lets `eval_MATH_vllm.py` (baseline) and `eval_MATH_steering.py` (steered) target
LogiQA with the *same* generation + steering core they use for MATH/GSM. LogiQA is
the same "shape" as GSM (reason, then a short answer) — only the answer is
multiple-choice, so scoring is choice accuracy instead of numeric equality.

  - load:    datatune/LogiQA2.0 (config "default", split "test"), STREAMED — a
             non-streaming load materializes the 63k-row train split onto the
             (slow) HF cache and hangs.
  - prompt:  a chat instruction that elicits a <think> block + a final
             'Answer: X' (so the SEAL steering trigger on `\\n\\n` inside <think>
             still fires).
  - grade:   `logic_eval_main` mirrors get_math_results.main's output schema
             (all_pred / all_eval / mv_pred / mv_eval / mv_index + math_eval.jsonl
             + metrics.json), so visualize_results.py works unchanged.
"""
import json
import os
import re
from collections import Counter


def load_logiqa(split="test", config="default"):
    from datasets import load_dataset

    ds = load_dataset("datatune/LogiQA2.0", config, split=split, streaming=True)
    out, skipped = [], 0
    for r in ds:
        # LogiQA 2.0 mixes MRC rows (text/question/options/answer — what we want)
        # with NLI rows (major_premise/conclusion/label) and a few malformed JSON
        # lines. Skip anything that isn't a valid MRC row.
        try:
            d = json.loads(r["text"])
            out.append({
                "passage": d["text"],
                "question": d["question"],
                "options": d["options"],
                "gt": int(d["answer"]),
            })
        except (KeyError, ValueError, TypeError):
            skipped += 1
    if skipped:
        print(f"[logiqa] skipped {skipped} non-MRC or malformed rows")
    return out


LOGIQA_INSTRUCTION = (
    "Read the passage and answer the multiple-choice question. "
    "Think step-by-step, then end your response with 'Answer: X' where X is A, B, C, or D.\n\n"
    "Passage: {passage}\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}"
)


def build_logiqa_prompt(ex):
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(ex["options"]))
    return LOGIQA_INSTRUCTION.format(
        passage=ex["passage"].strip(),
        question=ex["question"].strip(),
        options=opts,
    )


def extract_choice(gen):
    # Only look after </think> so a tentative choice inside the reasoning isn't
    # picked over the final answer.
    tail = gen.split("</think>")[-1] if "</think>" in gen else gen
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", tail, re.IGNORECASE)
    if not m:
        m = re.findall(r"\b([ABCD])\b", tail)
    return "ABCD".index(m[-1].upper()) if m else None


def logic_eval_main(res_path, save=False, output_dir=None):
    """Grade LogiQA predictions with the same output schema as
    get_math_results.main so downstream tooling (visualize_results.py) is shared."""
    with open(res_path) as f:
        data = [json.loads(line) for line in f]

    for example in data:
        gens = example.get("model_generation") or [example.get("model_output", "")]
        gt = int(example["answer"])
        all_pred = [extract_choice(g) for g in gens]
        all_eval = [(p is not None and p == gt) for p in all_pred]

        valid = [p for p in all_pred if p is not None]
        if valid:
            pred = Counter(valid).most_common(1)[0][0]
            index = all_pred.index(pred)
        else:
            pred, index = None, 0

        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = bool(all_eval[index]) if all_eval else False
        example["mv_index"] = index

    acc = sum(e["mv_eval"] for e in data) / len(data) if data else 0.0
    print(f"Accuracy: {acc:.3f}")

    if save:
        with open(os.path.join(output_dir, "math_eval.jsonl"), "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({"acc": acc}, f)

    return acc
