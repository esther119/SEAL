# Archived — parallel LogiQA2 pipeline (S_general integration)

Author: swimmingcircle119 <estheryang1109@gmail.com>, merged to main 2026-07-29
("Add LogiQA domain to S_general pipeline" / "Use domain-neutral LogiQA trace
filename").

## What this was

An independent LogiQA2.0 generation+scoring pipeline built to add a logic domain
to the S_general multi-domain vector:

* `logic/gen_logiqa2_vllm.py`, `logic/data.py` — LogiQA2 trace generation + scoring
* `data_LogiQA2.0/train_logic.jsonl` — English-only LogiQA train snapshot (12,568 rows)
* `test_logic_data.py` — its unit tests

It also shipped its own `KEYWORD_SETS["logic"]` in hidden_analysis.py.

## Why archived

The repo now has a single canonical LogiQA pipeline (the one that built
`vectors/logiqa_v_logic.pt`):

* `gen_logiqa_vllm.py` + `logic_utils.py` (generation, English filter, the
  `</think>`-anchored grader), and
* `data/LogiQA/` (train/test pools + the contamination-free `eval_rand42_500_clean.json`).

Two LogiQA pipelines sharing one `KEYWORD_SETS["logic"]` was ambiguous, and the
alternate keyword set that arrived with this pipeline did not match the set that
built the committed vector. Per decision, the canonical pipeline's keyword set
was restored in hidden_analysis.py and this pipeline was moved here.

## For the S_general owner (build_general_vector) — left untouched on purpose

`build_general_vector.py` and `scripts/build_general_vector.sh` are the S_general
orchestrator and are **not modified** by this archive — they remain exactly as on
main. They still reference the now-archived pipeline:

* `scripts/build_general_vector.sh` calls `python -m logic.gen_logiqa2_vllm` and
  reads `data/LogiQA2.0/…`
* `build_general_vector.py` references `data/LogiQA2.0/hidden_*` in its logic-domain
  example

So the S_general **LogiQA leg will not run as-is** after this archive. The LogiQA
generation/grading piece is now owned by the canonical pipeline
(`gen_logiqa_vllm.py` + `logic_utils.py` + `data/LogiQA/`, `--keywords logic`),
which fixes the grading bug still present in the archived `logic/data.py`. Point
the S_general LogiQA leg at that pipeline instead of the archived module. Left to
the S_general owner to reconcile — deliberately not done here.
