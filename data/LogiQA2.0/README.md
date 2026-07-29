# LogiQA 2.0 English-only train snapshot + durable hidden.pt for S_general.
#
# Snapshot: data/LogiQA2.0/train_logic.jsonl
#   - 12,568 MRC rows from datatune/LogiQA2.0 train, English-only
#   - Source: Andwwy/v_code-SEAL (materialized by make_train_jsonl_logiqa2.py)
#   - Fields: problem_id, passage, question, options, gt, type, level, lang,
#     usable, split, problem
#   - All rows are lang="en" and usable=true
#
# Durable hidden.pt: hidden_{correct,incorrect}_0_500/
#   Filled by scripts/build_general_vector.sh after GPU extraction — commit them.
#
# Generate traces (GPU):
#   python -m logic.gen_logiqa2_vllm \
#     --save_dir results/LogiQA_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
#     --resume
#
# Extract layer-20 hidden states (GPU):
#   python hidden_analysis.py --keywords logic --keep_layers 20 \
#     --type correct --start 0 --sample 500 ...
