"""LogiQA 2.0 dataset loading + prompting + scoring (logic domain).

Loads the committed English-only train snapshot under
``data/LogiQA2.0/train_logic.jsonl`` (12,568 MRC rows). Prompt and scorer
mirror ``logic_utils.py`` so extraction happens in the same distribution the
eval measures.

Two LogiQA-specific concerns (measured on unsteered traces):

1. LANGUAGE. LogiQA 2.0 is bilingual. English keyword lists tag 0% of Chinese
   segments, so a mixed extraction would encode English-vs-Chinese rather than
   reflection-vs-execution. The committed train file is already English-only;
   ``is_english_trace`` is an additional safety net on generated CoTs.
2. Usability. A few upstream rows have blank options; ``is_usable`` mirrors
   APPS' ``parse_tests`` returning None for no-test rows.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

HAN = re.compile(r"[一-鿿]")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_PATH = os.path.join(_ROOT, "data", "LogiQA2.0", "train_logic.jsonl")

LOGIQA_INSTRUCTION = (
    "Read the passage and answer the multiple-choice question. "
    "Think step-by-step, then end your response with 'Answer: X' where X is A, B, C, or D.\n\n"
    "Passage: {passage}\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}"
)


def load_train_logic(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load the committed English-only LogiQA 2.0 train snapshot in file order."""
    path = path or DEFAULT_TRAIN_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing LogiQA train snapshot at {path}. "
            "Expected data/LogiQA2.0/train_logic.jsonl"
        )
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_english_problem(row: Dict[str, Any], max_cjk: float = 0.05) -> bool:
    """True if the problem text itself is English."""
    text = (row.get("passage") or "") + (row.get("question") or "")
    if not text:
        return False
    return len(HAN.findall(text)) / len(text) <= max_cjk


def is_usable(row: Dict[str, Any]) -> bool:
    """False for LogiQA 2.0 empty/defective rows (blank question or options)."""
    if not (row.get("question") or "").strip():
        return False
    opts = row.get("options") or []
    return len(opts) == 4 and all((o or "").strip() for o in opts)


def build_instruction(row: Dict[str, Any]) -> str:
    """User-turn instruction — mirrors ``logic_utils.build_logiqa_prompt``."""
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(row["options"]))
    return LOGIQA_INSTRUCTION.format(
        passage=row["passage"].strip(),
        question=row["question"].strip(),
        options=opts,
    )


def extract_choice(gen: str) -> Optional[int]:
    """0–3 index of the model's final answer, or None.

    Only looks after ``</think>`` so a tentative choice inside reasoning is
    not preferred over the final answer.
    """
    tail = gen.split("</think>")[-1] if "</think>" in gen else gen
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", tail, re.IGNORECASE)
    if not m:
        m = re.findall(r"\b([ABCD])\b", tail)
    return "ABCD".index(m[-1].upper()) if m else None


def score_generation(gen: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """``{passed, pred, gt}`` — choice accuracy. Unparseable answers count wrong."""
    pred = extract_choice(gen)
    return {
        "passed": pred is not None and pred == int(row["gt"]),
        "pred": pred,
        "gt": int(row["gt"]),
    }


def think_segments(gen: str) -> List[str]:
    """Thoughts inside ``<think>``, blank-line split (same as boundary tagging)."""
    body = gen.split("<think>", 1)[1] if "<think>" in gen else gen
    body = body.split("</think>", 1)[0]
    return [s.strip() for s in re.split(r"\n\s*\n", body) if s.strip()]


def cjk_fraction(gen: str) -> float:
    """Fraction of this trace's thought segments that are Chinese-dominant."""
    segs = think_segments(gen)
    if not segs:
        return 0.0
    cjk = sum(1 for s in segs if len(HAN.findall(s)) / max(len(s), 1) > 0.15)
    return cjk / len(segs)


def is_english_trace(gen: str, max_cjk: float = 0.10) -> bool:
    """True if the trace reasons in English (safety net for pool membership)."""
    return cjk_fraction(gen) <= max_cjk
