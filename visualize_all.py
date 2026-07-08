import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import visualize_results as vr

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TAG = "DeepSeek-R1-Distill-Qwen-1.5B"

# All results produced with the MATH-derived steering vector live under this root.
# Teammates using other vectors add sibling folders (e.g. results/results_for_<X>_vectors/).
RESULTS_ROOT = "results/results_for_math_vectors"

runs = [
    ("Math (GSM8K)",  f"{RESULTS_ROOT}/GSM/{TAG}/paper_baseline_10000", f"{RESULTS_ROOT}/GSM/{TAG}/paper_steer_10000"),
    ("Code (MBPP)",   f"{RESULTS_ROOT}/MBPP/{TAG}/transfer_baseline",   f"{RESULTS_ROOT}/MBPP/{TAG}/transfer_steered"),
    ("Logic (LogiQA)",f"{RESULTS_ROOT}/LogiQA/{TAG}/transfer_baseline", f"{RESULTS_ROOT}/LogiQA/{TAG}/transfer_steered"),
]

summaries = []
for name, b, s in runs:
    try:
        a = vr.analyze(b, s, model=MODEL, dataset=name)
        summaries.append((name, a["summary"]))
    except Exception as e:
        print(f"[skip] {name}: {e}")

labels    = [n for n, _ in summaries]
base_acc  = [s["baseline"]["accuracy"] * 100 for _, s in summaries]
steer_acc = [s["steered"]["accuracy"] * 100  for _, s in summaries]
base_tok  = [s["baseline"]["avg_total"] for _, s in summaries]
steer_tok = [s["steered"]["avg_total"]  for _, s in summaries]

x = np.arange(len(labels)); w = 0.38
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def lab(ax, bars, fmt):
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()),
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontweight="bold")

b1 = ax1.bar(x - w/2, base_acc, w, label="baseline", color="#9aa0a6")
b2 = ax1.bar(x + w/2, steer_acc, w, label="steered", color="#1a73e8")
lab(ax1, b1, "{:.1f}"); lab(ax1, b2, "{:.1f}")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylabel("Accuracy / pass@1 (%)"); ax1.set_title("Accuracy: baseline vs steered"); ax1.legend()

b3 = ax2.bar(x - w/2, base_tok, w, label="baseline", color="#9aa0a6")
b4 = ax2.bar(x + w/2, steer_tok, w, label="steered", color="#f9ab00")
lab(ax2, b3, "{:.0f}"); lab(ax2, b4, "{:.0f}")
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel("avg total tokens"); ax2.set_title("Tokens: baseline vs steered"); ax2.legend()

fig.suptitle(f"SEAL steering across domains — {TAG}", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(f"{RESULTS_ROOT}/combined", exist_ok=True)
fig.savefig(f"{RESULTS_ROOT}/combined/all_domains.png", dpi=150)

lines = ["# SEAL steering across domains", "",
         "| Domain | Acc base | Acc steered | Δacc | Tok base | Tok steered | Δtok |",
         "|---|---|---|---|---|---|---|"]
for name, s in summaries:
    b, st, d = s["baseline"], s["steered"], s["delta"]
    lines.append(f"| {name} | {b['accuracy']*100:.1f}% | {st['accuracy']*100:.1f}% | "
                 f"{d['accuracy_pts']:+.1f} | {b['avg_total']:.0f} | {st['avg_total']:.0f} | "
                 f"-{d['total_reduction_pct']:.0f}% |")
md = "\n".join(lines)
open(f"{RESULTS_ROOT}/combined/summary.md", "w").write(md)
print("\n" + md)
print(f"\nWrote {RESULTS_ROOT}/combined/all_domains.png and {RESULTS_ROOT}/combined/summary.md")
