"""CPU tests for LogiQA loading, scoring, English filters, and repair."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from logic.data import (
    build_instruction,
    cjk_fraction,
    extract_choice,
    is_english_problem,
    is_english_trace,
    is_usable,
    load_train_logic,
    score_generation,
)
from logic.gen_logiqa2_vllm import repair_outputs, trim_output


def _classify_step(step: str, check_words, check_prefix, switch_words, switch_prefix) -> str:
    """Mirror hidden_analysis.generate_index matching (reflection first)."""
    s = step.strip(" ").strip("\n")
    if any(s.lower().startswith(p.lower()) for p in check_prefix) or any(
        w.lower() in s.lower() for w in check_words
    ):
        return "check"
    if any(s.lower().startswith(p.lower()) for p in switch_prefix) or any(
        w.lower() in s.lower() for w in switch_words
    ):
        return "switch"
    return "other"


class LogicDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_train_logic()

    def test_train_snapshot_is_english_usable(self):
        self.assertGreaterEqual(len(self.rows), 10000)
        sample = self.rows[:50]
        self.assertTrue(all(r.get("lang") == "en" for r in sample))
        self.assertTrue(all(is_usable(r) for r in sample))
        self.assertTrue(all(is_english_problem(r) for r in sample))
        self.assertIn("problem_id", sample[0])
        self.assertEqual(len(sample[0]["options"]), 4)

    def test_build_instruction_contains_options(self):
        instr = build_instruction(self.rows[0])
        self.assertIn("Passage:", instr)
        self.assertIn("Question:", instr)
        self.assertIn("A.", instr)
        self.assertIn("D.", instr)

    def test_extract_choice_prefers_post_think_answer(self):
        gen = (
            "<think>\nI guess A for now.\n\nWait, maybe B.\n</think>\n"
            "Answer: C"
        )
        self.assertEqual(extract_choice(gen), 2)

    def test_score_generation_marks_correct_and_incorrect(self):
        row = {
            "gt": 1,
            "options": ["a", "b", "c", "d"],
            "question": "q",
            "passage": "p",
        }
        ok = score_generation("</think>\nAnswer: B", row)
        bad = score_generation("</think>\nAnswer: A", row)
        none = score_generation("</think>\nno choice here", row)
        self.assertTrue(ok["passed"])
        self.assertFalse(bad["passed"])
        self.assertFalse(none["passed"])
        self.assertIsNone(none["pred"])

    def test_cjk_filters(self):
        english = (
            "<think>\nWait, let me check the premise.\n\n"
            "Alternatively, try another approach.\n</think>\nAnswer: A"
        )
        chinese = (
            "<think>\n这是中文推理过程。\n\n继续思考这个问题。\n</think>\nAnswer: A"
        )
        self.assertLessEqual(cjk_fraction(english), 0.10)
        self.assertTrue(is_english_trace(english))
        self.assertGreater(cjk_fraction(chinese), 0.10)
        self.assertFalse(is_english_trace(chinese))
        self.assertFalse(
            is_english_problem(
                {"passage": "这是中文段落" * 20, "question": "这是中文问题"}
            )
        )

    def test_is_usable_rejects_blank_options(self):
        row = {
            "question": "Q",
            "options": ["A", "", "C", "D"],
            "passage": "P",
            "gt": 0,
        }
        self.assertFalse(is_usable(row))

    def test_repair_outputs_truncates_to_aligned_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_path = os.path.join(tmp, "math_eval.jsonl")
            data_path = os.path.join(tmp, "data.jsonl")
            with open(eval_path, "w") as f:
                f.write(json.dumps({"problem": "p1", "all_eval": [True]}) + "\n")
                f.write(json.dumps({"problem": "p2", "all_eval": [False]}) + "\n")
                f.write("{not-json\n")
            with open(data_path, "w") as f:
                f.write(json.dumps({"problem": "p1", "level": "logic"}) + "\n")
                f.write(json.dumps({"problem": "DIFFERENT", "level": "logic"}) + "\n")
            repaired = repair_outputs(eval_path, data_path)
            self.assertEqual(len(repaired), 1)
            self.assertEqual(repaired[0]["problem"], "p1")

    def test_trim_output_cuts_at_markers(self):
        text = "reasoning...\nAnswer: A\nQuestion: leaked next prompt"
        self.assertEqual(trim_output(text), "reasoning...\nAnswer: A\n")


class LogicKeywordTests(unittest.TestCase):
    def test_logic_keyword_set_present_in_hidden_analysis(self):
        with open("hidden_analysis.py", encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"logic":', src)
        for word in (
            "missing something",
            "overcomplicating",
            "let me try to think",
            "think of it differently",
        ):
            self.assertIn(f'"{word}"', src)
        self.assertIn("'logic' (LogiQA-adapted)", src)

    def test_logic_keyword_matching_priority(self):
        # Evidence-derived lists from hidden_analysis.KEYWORD_SETS["logic"].
        check_words = [
            "wait",
            "think again",
            "missing something",
            "overcomplicating",
            "stuck",
            "hmm",
            "let me check",
            "not making progress",
            "but wait",
            "double-check",
            "'s correct",
            "'s incorrect",
        ]
        switch_words = [
            "alternatively",
            "let me try to think",
            "think differently",
            "think of it differently",
            "another approach",
        ]
        self.assertEqual(
            _classify_step("Wait, I am stuck here.", check_words, [], switch_words, []),
            "check",
        )
        self.assertEqual(
            _classify_step(
                "Alternatively, let me try to think.", check_words, [], switch_words, []
            ),
            "switch",
        )
        self.assertEqual(
            _classify_step("Compute the next option.", check_words, [], switch_words, []),
            "other",
        )


if __name__ == "__main__":
    unittest.main()
