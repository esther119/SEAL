# v_code 3eval — evaluate a steering vector on 3 benchmarks

Takes a SEAL steering vector as **input** and measures baseline-vs-steered on
**MBPP**, **GSM8K**, and **LogiQA 2.0** (100 tasks each): does the vector cut
reasoning tokens while holding/raising accuracy?

| Benchmark | Dataset | Metric |
|---|---|---|
| MBPP | `google-research-datasets/mbpp` (full/test) | pass@1 — run generated code vs `test_list` |
| GSM8K | `openai/gsm8k` (main/test) | numeric accuracy — `\boxed{}` / last-number vs `#### gt` |
| LogiQA 2.0 | `datatune/LogiQA2.0` (default/test) | multiple-choice accuracy — extract A/B/C/D vs gold index |

For every task: greedy decode **baseline** and **steered**, score, and record token counts.

## Steering (no custom modeling)
`steering.py` applies the vector via **forward hooks** (version-safe, works with transformers 5.x):
an embed pre-hook tracks the `<think>` block and flags each `\n\n` token; a hook on
`model.model.layers[LAYER-1]` adds `coef · S` to the last-token residual there — the same
intervention point SEAL uses (verified `hidden_states[L] == layers[L-1].output`).
Default `coef = +1.0` for `S = H_E − H_RT` (push toward execution / away from reflection).

Runs in the **SEAL vLLM env** (transformers 4.47.1) — it only needs HF (steering hook),
no vLLM — so it can share the extraction pod. From the fork:
```bash
cd /workspace/seal-vllm/3eval          # after git pull
VECTOR=/workspace/seal-vllm/results/MBPP/.../vector_mbpp_120_120/layer_20_transition_reflection_steervec.pt \
    bash run_eval.sh
```
Knobs (env): `N` (tasks/benchmark, 100), `COEF` (**-1.0**, for SEAL's `H_RT-H_E` vector),
`LAYER` (20), `MAX_TOKENS` (3072), `BATCH_SIZE` (16), `BENCHMARKS` ("mbpp gsm8k logiqa").

Output: `results/summary.json` + `results/<benchmark>.json` (per-task correctness,
token counts, full generations), and a console summary table.

## Runtime
**Batched** greedy decode (per-sequence steering mask, `BATCH_SIZE=16`): 100 tasks ×
2 conditions × 3 benchmarks = 600 generations ≈ **~1–1.5 hr on an A100** (vs ~5–12 hr at
batch-1). Note: running the 3 benchmarks as parallel processes on ONE GPU does NOT help —
they share the GPU; batching is the real speedup. Lower `MAX_TOKENS`/`N` to shorten further.

## Files
- `steering.py` — forward-hook steering
- `benchmarks.py` — 3 benchmarks (load / prompt / score)
- `run_eval.py` — orchestrator (baseline vs steered, metrics, JSON)
- `setup_runpod.sh`, `run_eval.sh`, `requirements.txt`, `.env.example`
