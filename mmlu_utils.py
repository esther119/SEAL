"""MMLU support for the SEAL MATH-style eval scripts (Harness A).

Lets `eval_MATH_vllm.py` (baseline) and `eval_MATH_steering.py` (steered) target
an MMLU subject (default: philosophy) with the *same* generation + steering core
used for MATH/GSM/LogiQA. MMLU is a 4-way multiple-choice task like LogiQA (just
without the reading passage), so grading reuses `logic_utils.extract_choice` /
`logic_utils.logic_eval_main` — the answer is a 0-3 index scored as choice
accuracy, and the output schema matches get_math_results.main so
visualize_results.py works unchanged.

  - load:   cais/mmlu (config = subject, e.g. "philosophy"), split "test".
  - prompt: a chat instruction that elicits a <think> block + a final
            'Answer: X' (so the SEAL steering trigger on `\\n\\n` inside <think>
            still fires), identical in shape to the LogiQA prompt.
"""
import json


def load_mmlu(subject="philosophy", split="test"):
    from datasets import load_dataset

    # MMLU ships test / validation / dev splits; map the unified split vocabulary.
    hf_split = {"val": "validation"}.get(split, split)
    ds = load_dataset("cais/mmlu", subject, split=hf_split)
    out = []
    for r in ds:
        out.append({
            "question": r["question"],
            "options": list(r["choices"]),
            "gt": int(r["answer"]),
        })
    return out


MMLU_INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "Think step-by-step, then end your response with 'Answer: X' where X is A, B, C, or D.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}"
)


def build_mmlu_prompt(ex):
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(ex["options"]))
    return MMLU_INSTRUCTION.format(question=ex["question"].strip(), options=opts)
