# LogiQA 2.0 (English-only) — extraction pool, eval pool, and shipped S_logic traces.
# Source: datatune/LogiQA2.0, MRC rows only (NLI rows and malformed lines dropped by logic_utils.load_logiqa).
# Home vector: ../../vectors/logiqa_v_logic.pt
#
# LANGUAGE FILTER — LogiQA 2.0 ships an untranslated Chinese mirror, so every split is ~50% CJK.
# Both pools below are filtered with is_cjk_heavy (logic_utils.py, threshold 0.1), the same
# function gen_logiqa_vllm.py uses, so build and eval filter identically.
#   train.jsonl  12,567 English of 25,318 rows   <- extraction source
#   test.jsonl    1,572 English of  3,166 rows   <- eval source
#
# EVAL SET — eval_rand42_500.json pins the exact 500 problems scored, as pool indices into
# test.jsonl. Reproduces eval_MATH_steering.py's rule: shuffle range(n) with sample_seed=42,
# keep the first 500, restore file order. Verified 500/500 against the prompts actually
# evaluated in results/results_for_logic_vectors/LogiQA/baseline/.../predictions.jsonl.
#
# KNOWN OVERLAP — LogiQA 2.0's own train and test splits are not disjoint. 16 of the 500 items
# in eval_rand42_500.json also appear among the 2,000 attempted train rows; 4 of those reached
# the 500+500 subset that built the vector (eval ranks 240, 264, 419, 424 -> pool idx 763, 823,
# 1286, 1294). Identity is the full prompt (passage + question + options) — question text alone
# repeats across different passages and is not a valid key.
#
# USE eval_rand42_500_clean.json FOR ALL FUTURE RUNS. It replaces all 16 overlapping items with
# backfills drawn from the same seed-42 shuffle order (positions 500+), skipping any row present
# in the attempted train rows. Result: 500 items, zero overlap with anything ever generated for
# extraction. eval_rand42_500.json is retained only because the first LogiQA baseline/steered
# results were scored against it.
#
# Legacy traces: baseline_3000/ preserves the original pre-fix labels for provenance.
# Canonical strict labels: ../../results/results_for_logic_vectors/LogiQA_train/
#   DeepSeek-R1-Distill-Qwen-1.5B/baseline_3000_regraded/. Unfinished generations are
#   retained as incorrect, never inferred from letters inside an unclosed reasoning block.
#   The first 500 correct + 500 incorrect traces built vectors/logiqa_v_logic.pt.
# Durable hidden.pt: hidden_{correct,incorrect}_0_500/ (filled by generate_vector_logiqa.sh
#   or build_general_vector.sh on a GPU — commit after)
# Regenerate traces (GPU): MAX_EXAMPLES=2000 SAMPLE=500 KEYWORDS=logic bash scripts/generate_vector_logiqa.sh
