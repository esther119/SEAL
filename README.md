official code for "SEAL: Steerable Reasoning Calibration of Large Language Models for Free"



## Extract Steering Vector
```
bash scripts/generate_vector.sh
```

## Build S_general (MATH + APPS + LogiQA)

One GPU command extracts layer-20 boundary hidden states for MATH, APPS, and
LogiQA (logic keywords), then pools them into Phase 1 `S_general`:

```bash
bash scripts/build_general_vector.sh
```

Requirements on the GPU pod:
- MATH traces already under `results/results_for_math_vectors/MATH_train/.../baseline_10000/`
- shipped APPS traces under `data/APPS/baseline_10000/`
- committed English LogiQA snapshot at `data/LogiQA2.0/train_logic.jsonl`

Useful flags:
```bash
SKIP_HIDDEN=1 bash scripts/build_general_vector.sh   # pool-only (needs durable hidden.pt)
BALANCE=1 bash scripts/build_general_vector.sh       # Phase 2 equal-domain subsample
SKIP_LOGIQA_GEN=1 bash scripts/build_general_vector.sh  # reuse existing LogiQA traces
```

Outputs:
- `results/general/S_general_math_apps_logic_phase1.pt` (+ `.meta.json`)
- durable `data/{MATH,APPS,LogiQA2.0}/hidden_{correct,incorrect}_0_500/hidden.pt`

Apply with coefficient `-1.0` at layer 20 (same sign convention as domain vectors).
A standalone `logiqa2_v_logic.pt` is optional; `S_general` needs the labeled
`hidden.pt` pools, not the packaged home vector.

## Steering
```
bash scripts/steering.sh
```

## Evaluate any steering vector

Specify the vector file, a result label, and the datasets to evaluate:

```bash
DATASETS=math,apps,livecodebench \
bash scripts/eval_steering_vector_benchmarks.sh \
  results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt \
  math_vector \
  0
```

Defaults are all 500 MATH-500 problems, 500 seeded-random APPS test problems,
and all 400 LiveCodeBench `release_v1` problems. Select one or more datasets:

```bash
DATASETS=apps bash scripts/eval_steering_vector_benchmarks.sh CODE_VECTOR.pt code_vector 0
DATASETS=math,livecodebench bash scripts/eval_steering_vector_benchmarks.sh MATH_VECTOR.pt math_vector 0
```

Valid names are `math`, `apps`, and `livecodebench`. Override counts with
`MATH_MAX_EXAMPLES`, `APPS_MAX_EXAMPLES`, and `LCB_MAX_EXAMPLES`; use
`APPS_SPLIT=train` for research comparisons or `RUN_BASELINE=0` to skip
baselines. Set `MODEL`, `LAYER`, and `COEF` when evaluating a vector built for
different model or steering settings. APPS metrics include breakdowns by
difficulty and problem kind.

APPS generations are executed during grading. Run this only in an isolated
environment intended for evaluating untrusted model-generated code.
