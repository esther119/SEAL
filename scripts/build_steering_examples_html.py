#!/usr/bin/env python3
"""Regenerate the examples section for docs/steering_transfer_results.html."""
import html as htmlmod
import json
import re
import subprocess
import sys

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TAG = "DeepSeek-R1-Distill-Qwen-1.5B"
R = "results/results_for_math_vectors"
STEER = (
    "baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/"
    "coef_-1.0_remove_bos/math_eval.jsonl"
)
LABELS = "ABCD"
COMBOS = [
    ("fixed", "Wrong → correct", lambda b, s: not ok(b) and ok(s)),
    ("broke", "Correct → wrong", lambda b, s: ok(b) and not ok(s)),
    ("both_ok", "Both correct", lambda b, s: ok(b) and ok(s)),
    ("both_wrong", "Both wrong", lambda b, s: not ok(b) and not ok(s)),
]
PILL = {
    "Wrong → correct": "pill good",
    "Correct → wrong": "pill bad",
    "Both correct": "pill neutral",
    "Both wrong": "pill neutral",
}


def git_show(path, ref="origin/main"):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)


def load_gitl(path, ref="origin/main"):
    return [json.loads(line) for line in git_show(path, ref).splitlines() if line.strip()]


def get_counter():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)

        def count(text):
            if not text:
                return 0
            return len(tok.encode(text, add_special_tokens=False))

        return count, "tokens"
    except Exception:
        return lambda text: max(0, len(text) // 4) if text else 0, "est. tokens"


COUNT, UNIT = get_counter()


def gen_text(row):
    gens = row.get("model_generation") or [row.get("model_output", "")]
    idx = row.get("mv_index", 0)
    return gens[idx] if gens else ""


def ok(row):
    if "mv_eval" in row:
        return bool(row["mv_eval"])
    if "correct" in row:
        return bool(row["correct"])
    ev = row.get("all_eval") or []
    return bool(ev[0]) if ev else False


def split_parts(text):
    if THINK_CLOSE in text:
        think, answer = text.split(THINK_CLOSE, 1)
        if THINK_OPEN in think:
            think = think.split(THINK_OPEN, 1)[1]
        return think.strip(), answer.strip()
    if THINK_OPEN in text:
        body = text.split(THINK_OPEN, 1)[1]
        return body.strip(), ""
    return "", text.strip()


def preview(text, n=72):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


def esc(text):
    return htmlmod.escape(str(text or ""))


def meta_line(domain, meta):
    if domain.startswith("Code"):
        return f"pass@1: {meta.get('pass@1', '?')}"
    if domain.startswith("Math"):
        return f"baseline pred {meta.get('b_pred', '?')} → steered pred {meta.get('s_pred', '?')}"
    return (
        f"GT {meta.get('gt', '?')} · baseline {meta.get('b_pred', '?')} "
        f"→ steered {meta.get('s_pred', '?')}"
    )


def meta(domain, base, steer):
    if domain.startswith("Code"):
        bp = "pass" if ok(base) else "fail"
        sp = "pass" if ok(steer) else "fail"
        return {"pass@1": f"{bp} → {sp}"}
    if domain.startswith("Math"):
        return {"b_pred": str(base.get("mv_pred")), "s_pred": str(steer.get("mv_pred"))}
    return {
        "gt": LABELS[base["answer"]],
        "b_pred": LABELS[base["mv_pred"]] if base.get("mv_pred") is not None else "?",
        "s_pred": LABELS[steer["mv_pred"]] if steer.get("mv_pred") is not None else "?",
    }


def prompt_for(domain, row):
    if domain.startswith("Logic"):
        prompt = row.get("prompt", "")
        match = re.search(r"Question:\s*(.+?)(?:\n\nOptions:|$)", prompt, re.S)
        return (match.group(1).strip() if match else row.get("problem", ""))
    return row.get("problem", "")


def score_pair(key, base_tok, steer_tok):
    if key in ("fixed", "both_ok"):
        return base_tok - steer_tok
    if key == "broke":
        return steer_tok - base_tok
    return -(base_tok + steer_tok)


def measure(text):
    think, answer = split_parts(text)
    think_tok = COUNT(think)
    answer_tok = COUNT(answer)
    total_tok = COUNT(text)
    return {
        "full": text,
        "think": think,
        "answer": answer,
        "think_tok": think_tok,
        "answer_tok": answer_tok,
        "total_tok": total_tok,
    }


def tok_delta(base_tok, steer_tok):
    if not base_tok:
        return 0
    return round((1 - steer_tok / base_tok) * 100)


def tok_chip(base_total, steer_total):
    delta = tok_delta(base_total, steer_total)
    sign = "&minus;" if delta >= 0 else "+"
    cls = "down" if delta >= 0 else "up"
    return (
        f'<span class="chip tokens {cls}"><strong>Total {UNIT}</strong> '
        f"{base_total:,} → {steer_total:,} ({sign}{abs(delta)}%)</span>"
    )


def breakdown_chip(label, base_m, steer_m):
    if base_m["think_tok"] or steer_m["think_tok"]:
        return (
            f'<span class="chip breakdown"><strong>{label}</strong> '
            f"thinking {base_m['think_tok']:,} → {steer_m['think_tok']:,} · "
            f"answer {base_m['answer_tok']:,} → {steer_m['answer_tok']:,}</span>"
        )
    return ""


def output_block(kind, title, measured):
    think = measured["think"]
    answer = measured["answer"]
    total = measured["total_tok"]
    think_tok = measured["think_tok"]
    answer_tok = measured["answer_tok"]

    if think:
        meta = (
            f"{total:,} {UNIT} "
            f"({think_tok:,} thinking + {answer_tok:,} answer) · {preview(measured['full'])}"
        )
        open_attr = "" if total > 1200 else " open"
        parts = [
            f'<details class="out-part think">',
            f'  <summary>Thinking · {think_tok:,} {UNIT}</summary>',
            f'  <pre class="out-text">{esc(think)}</pre>',
            "</details>",
            f'<details class="out-part answer"{(" open" if answer_tok < 900 else "")}>',
            f'  <summary>Final answer · {answer_tok:,} {UNIT}</summary>',
            f'  <pre class="out-text">{esc(answer)}</pre>',
            "</details>",
        ]
        body = "\n".join(parts)
    else:
        meta = f"{total:,} {UNIT} · {preview(measured['full'])}"
        open_attr = " open" if total < 900 else ""
        body = f'<pre class="out-text">{esc(measured["full"])}</pre>'

    return (
        f'<details class="out {kind}"{open_attr}>\n'
        f'  <summary><span class="out-name">{esc(title)}</span>'
        f'<span class="out-meta">{esc(meta)}</span></summary>\n'
        f"  <div class=\"out-body\">\n{body}\n  </div>\n"
        f"</details>"
    )


def build_example(domain, combo_label, example):
    base_m = measure(example["b_full"])
    steer_m = measure(example["s_full"])
    delta = tok_delta(base_m["total_tok"], steer_m["total_tok"])

    chips = [
        f'<span class="chip"><strong>Result</strong> {esc(meta_line(domain, example["meta"]))}</span>',
        tok_chip(base_m["total_tok"], steer_m["total_tok"]),
        breakdown_chip("Split", base_m, steer_m),
    ]
    chips_html = "\n    ".join(c for c in chips if c)

    return (
        f'<div class="combo">\n'
        f'  <div class="combo-head"><span class="{PILL[combo_label]}">{esc(combo_label)}</span></div>\n'
        f'  <div class="task-box"><span class="task-label">Task</span>{esc(example["prompt"])}</div>\n'
        f'  <div class="metric-row">\n    {chips_html}\n  </div>\n'
        f'  <div class="outputs">\n'
        f'    {output_block("base", "Baseline", base_m)}\n'
        f'    {output_block("steer", "Steered", steer_m)}\n'
        f"  </div>\n"
        f"</div>"
    )


def build_domain(domain, base_path, steer_path, ref="origin/main"):
    base_rows = load_gitl(base_path, ref)
    steer_rows = load_gitl(steer_path, ref)
    combos_html = []

    for key, label, pred in COMBOS:
        candidates = []
        for base, steer in zip(base_rows, steer_rows):
            if not pred(base, steer):
                continue
            b_full = gen_text(base)
            s_full = gen_text(steer)
            b_m = measure(b_full)
            s_m = measure(s_full)
            candidates.append({
                "label": label,
                "prompt": prompt_for(domain, base),
                "meta": meta(domain, base, steer),
                "b_full": b_full,
                "s_full": s_full,
                "_score": score_pair(key, b_m["total_tok"], s_m["total_tok"]),
            })
        candidates.sort(key=lambda row: row["_score"], reverse=True)
        if not candidates:
            combos_html.append(
                f'<div class="combo missing"><span class="pill neutral">{esc(label)}</span>'
                f"<p>No example in this run.</p></div>"
            )
            continue
        ex = candidates[0]
        combos_html.append(build_example(domain, label, ex))

    return (
        f'<section class="domain-examples">\n'
        f"  <h3>{esc(domain)}</h3>\n"
        f'  <div class="combo-grid">{"".join(combos_html)}</div>\n'
        f"</section>"
    )


def build_all():
    domains = [
        ("Math (GSM8K)", f"{R}/GSM/{TAG}/paper_baseline_10000/math_eval.jsonl",
         f"{R}/GSM/{TAG}/paper_steer_10000/{STEER}"),
        ("Code (MBPP)", f"{R}/MBPP/{TAG}/transfer_baseline/math_eval.jsonl",
         f"{R}/MBPP/{TAG}/transfer_steered/{STEER}"),
        ("Logic (LogiQA)", f"{R}/LogiQA/{TAG}/transfer_baseline/rand42_300/math_eval.jsonl",
         f"{R}/LogiQA/{TAG}/transfer_steered/"
         f"baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/"
         f"coef_-1.0_remove_bos/rand42_300/math_eval.jsonl"),
        ("Knowledge (MMLU)", f"{R}/MMLU/{TAG}/transfer_baseline/math_eval.jsonl",
         f"{R}/MMLU/{TAG}/transfer_steered/{STEER}", "esther119/mmlu-philosophy"),
    ]
    sections = []
    for item in domains:
        ref = item[3] if len(item) > 3 else "origin/main"
        sections.append(build_domain(item[0], item[1], item[2], ref))
    return "\n".join(sections)


def main():
    frag = build_all()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(frag)
        print(f"wrote {sys.argv[1]} ({len(frag):,} chars) using {UNIT}")
    else:
        print(frag)


if __name__ == "__main__":
    main()
