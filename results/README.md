# Results layout

Results are grouped by the steering vector they were produced with, so multiple
contributors can add runs without clobbering each other.

```
results/
  results_for_math_vectors/     # runs using the MATH-derived steering vector
    GSM/                        # math benchmark (GSM8K)
    MATH500/                    # math benchmark (MATH-500)
    MBPP/                       # code benchmark
    APPS/                       # code benchmark
    LiveCodeBench/              # code benchmark
    LogiQA/                     # logic benchmark
    MMLU/                       # knowledge benchmark
    MATH_train/                 # vector-extraction artifacts
    GSM_MBPP_LogiQA/            # cross-domain summary (all_domains.png, summary.md)
    MATH500_APPS_LogiQA/        # cross-domain summary (math_apps_logiqa.png, summary.json)
```

Cross-domain summary folders are named after the datasets they combine
(e.g. `GSM_MBPP_LogiQA/`), never a generic name like `combined/`.

## Adding your own results

If you steer with a different vector, create a sibling folder named
`results_for_<source>_vectors/` (e.g. `results_for_code_vectors/`) and keep the
same per-benchmark subfolder structure. Point your visualization scripts at your
own root via the `RESULTS_ROOT` constant (see `visualize_all.py`).
