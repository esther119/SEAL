# v_code MBPP steering vector — how it maps to SEAL, and runtime

Builds an MBPP steering vector using **SEAL's own pipeline**, reused stage-for-stage.
Run in the SEAL env (vLLM): `bash scripts/generate_vector_mbpp.sh`.

## Ours vs SEAL original (MATH-500), stage by stage
| Stage | SEAL original (MATH) | Ours (MBPP) |
|---|---|---|
| Generate traces | `eval_MATH_vllm.py` — vLLM, MATH-train, math prompt | `gen_mbpp_vllm.py` — vLLM, MBPP-full train, code prompt (**same** `LLM`/`SamplingParams`/chat-template/`remove_bos`) |
| Correctness label | math grader (`get_math_results`, `math_equal`) | run `test_list` in a hardened subprocess |
| Split correct/incorrect | `hidden_analysis.generate_math_data` reads `math_eval.jsonl` `all_eval` | **same** — our generator writes the identical `math_eval.jsonl` format |
| Thought classification | `generate_index` keywords (original) | `generate_index` keywords = **our code-adapted set** (contains-based; +hmm/what-if/instead/…; typo fixed) |
| Hidden-state extraction | `hidden_analysis.generate` — batch-1, all layers | **same file** — now **batched** forward (+`position_ids`) and **layer-20 only** (`--keep_layers 20`) |
| Steering vector | `vector_generation.py` → `H_RT − H_E` | **verbatim** |
| Apply at inference | add with coef −1.0 | **same** — coef −1.0 |

**Only genuine changes:** MBPP dataset + code prompt, code-execution scoring (vs math grading), our keyword set, and a *speed* change (batched + layer-only hidden extraction). Vector construction and application are SEAL's, unchanged.

## Correctness of the batching
Batched left-padded forward passes reproduce the batch-1 step-boundary activations to **~2e-8** (fp32) — verified. We pass `position_ids` that skip pad tokens (so RoPE matches) and offset each sequence's step indices by its left-pad count.

## Runtime estimate (A100-80GB, ~1/10-SEAL vector: SAMPLE=120/class)
| Stage | est. |
|---|---|
| `gen_mbpp_vllm.py` (vLLM, 374 traces + test scoring) | ~10–15 min |
| `hidden_analysis.py` × 2 (240 traces, batched B=8, layer-20) | ~8–12 min |
| `vector_generation.py` | <1 min |
| **Total** | **~20–30 min** |

For reference: SEAL's full MATH extraction (1000 traces, batch-1) is ~2 hr; the earlier pure-HF reimplementation (HF-generate per trace) was ~5 hr. This is faster because it (a) uses vLLM for generation, (b) targets 1/10 the activations, and (c) batches the forward pass. Bump `BATCH_SIZE=16` on the A100 to shave a few more minutes.
