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

Run matched baseline and steered evaluations on MATH-500, balanced APPS test
data, and LiveCodeBench:

```bash
bash scripts/eval_math_vector_apps_math.sh \
  results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt \
  0
```

The default APPS run uses the official test split, sampling 100 usable problems
from each difficulty tier (`n=300`, seed 42). Change sample size with
`MAX_EXAMPLES`, use `APPS_SPLIT=train` for research comparisons, or skip
baselines with `RUN_BASELINE=0`. APPS metrics include breakdowns by difficulty
and by call-based versus stdin/stdout problems. LiveCodeBench defaults to
`release_v1`; override it with `LCB_RELEASE`.

APPS generations are executed during grading. Run this only in an isolated
environment intended for evaluating untrusted model-generated code.