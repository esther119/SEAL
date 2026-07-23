official code for "SEAL: Steerable Reasoning Calibration of Large Language Models for Free"



## Extract Steering Vector
```
bash scripts/generate_vector.sh
```

## Steering
```
bash scripts/steering.sh
```

## Evaluate a MATH vector on MATH-500, APPS, and LiveCodeBench

Run matched baseline and steered evaluations on all three test datasets:

```bash
bash scripts/eval_math_vector_apps_math.sh \
  results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt \
  0
```

Defaults are all 500 MATH-500 problems, 500 seeded-random APPS test problems,
and all 400 LiveCodeBench `release_v1` problems. Select one or more datasets:

```bash
DATASETS=apps bash scripts/eval_math_vector_apps_math.sh VECTOR.pt 0
DATASETS=math,livecodebench bash scripts/eval_math_vector_apps_math.sh VECTOR.pt 0
```

Valid names are `math`, `apps`, and `livecodebench`. Override counts with
`MATH_MAX_EXAMPLES`, `APPS_MAX_EXAMPLES`, and `LCB_MAX_EXAMPLES`; use
`APPS_SPLIT=train` for research comparisons or `RUN_BASELINE=0` to skip
baselines. APPS metrics include breakdowns by difficulty and problem kind.

APPS generations are executed during grading. Run this only in an isolated
environment intended for evaluating untrusted model-generated code.