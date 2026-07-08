# Results layout

Results are grouped by the steering vector they were produced with, so multiple
contributors can add runs without clobbering each other.

```
results/
  results_for_math_vectors/     # runs using the MATH-derived steering vector
    GSM/                        # math benchmark (GSM8K)
    MBPP/                       # code benchmark
    LogiQA/                     # logic benchmark
    MATH_train/                 # vector-extraction artifacts
    combined/                   # cross-domain summary (all_domains.png, summary.md)
```

## Adding your own results

If you steer with a different vector, create a sibling folder named
`results_for_<source>_vectors/` (e.g. `results_for_code_vectors/`) and keep the
same per-benchmark subfolder structure. Point your visualization scripts at your
own root via the `RESULTS_ROOT` constant (see `visualize_all.py`).
