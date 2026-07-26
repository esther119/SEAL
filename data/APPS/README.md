# APPS train split (file order) + shipped Phase-1 traces for S_general.
# Traces: baseline_10000/ from Andwwy/v_code-SEAL (500+500 greedy extraction).
# Home vector: ../../vectors/apps_v_code.pt
# Durable hidden.pt: hidden_{correct,incorrect}_0_500/ (filled by scripts/build_general_vector.sh — commit after GPU run)
# Regenerate traces (GPU): python -m apps.gen_apps_vllm --save_dir results/APPS_train/.../baseline_10000 --resume
