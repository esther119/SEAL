#!/usr/bin/env python3
"""3-benchmark eval of a SEAL steering vector: MBPP / GSM8K / LogiQA 2.0.

For each benchmark, N tasks x {baseline, steered}, BATCHED greedy decode; score with the
benchmark's metric (pass@1 / numeric acc / MC acc) and record token counts. The vector is
an INPUT (--vector), applied via per-sequence forward hooks (steering.py).

Batching (default 16) is the real speedup vs running benchmarks in parallel on one GPU.

  python run_eval.py --vector vectors/mbpp_v_code.pt --n 100 \
      --benchmarks mbpp gsm8k logiqa --coef -1.0 --layer 20 --batch_size 16
"""
import argparse, json, os, time
from contextlib import nullcontext
import torch
import benchmarks as B
from steering import Steerer


def generate_batch(model, tok, prompts, max_tokens, steerer=None):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=6000).to(model.device)
    plen = enc["input_ids"].shape[1]
    with torch.no_grad(), (steerer if steerer else nullcontext()):
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_tokens,
                             pad_token_id=tok.eos_token_id)
    eos = tok.eos_token_id
    gens, ntoks = [], []
    for j in range(len(prompts)):
        gen_ids = out[j, plen:]
        nz = (gen_ids == eos).nonzero(as_tuple=False)          # trim at first eos (pad==eos)
        n = int(nz[0].item()) + 1 if len(nz) else int(gen_ids.shape[0])
        gens.append(tok.decode(gen_ids[:n], skip_special_tokens=False))
        ntoks.append(n)
    return gens, ntoks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True, help="steering vector .pt (SEAL: S = H_RT - H_E)")
    ap.add_argument("--benchmarks", nargs="+", default=["mbpp", "gsm8k", "logiqa"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--coef", type=float, default=-1.0, help="SEAL vector = H_RT-H_E, coef -1.0 pushes toward execution")
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[eval] device={device} dtype={args.dtype} layer={args.layer} coef={args.coef} batch={args.batch_size}", flush=True)

    vec = torch.load(args.vector, map_location="cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    steerer = Steerer(model, tok, vec, layer=args.layer, coef=args.coef)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for bench in args.benchmarks:
        load, prompt, score = B.BENCHMARKS[bench]
        exs = load(args.n)
        prompts = [prompt(ex, tok) for ex in exs]
        rows, agg = [], {"bc": 0, "sc": 0, "bt": 0, "st": 0}
        print(f"\n===== {bench}  ({len(exs)} tasks, batch {args.batch_size}) =====", flush=True)
        t0 = time.time()
        for i in range(0, len(exs), args.batch_size):
            bex, bp = exs[i:i + args.batch_size], prompts[i:i + args.batch_size]
            b_gens, b_tok = generate_batch(model, tok, bp, args.max_tokens)            # baseline
            s_gens, s_tok = generate_batch(model, tok, bp, args.max_tokens, steerer)   # steered
            for ex, bg, bt, sg, st in zip(bex, b_gens, b_tok, s_gens, s_tok):
                bok, sok = score(bg, ex), score(sg, ex)
                agg["bc"] += bok; agg["sc"] += sok; agg["bt"] += bt; agg["st"] += st
                rows.append({"base_correct": bok, "steer_correct": sok,
                             "base_tokens": bt, "steer_tokens": st, "base_gen": bg, "steer_gen": sg})
            done = min(i + args.batch_size, len(exs))
            print(f"[{bench} {done}/{len(exs)}] acc base={agg['bc']/done:.2f} steer={agg['sc']/done:.2f} "
                  f"| tok base={agg['bt']/done:.0f} steer={agg['st']/done:.0f} "
                  f"| {(time.time()-t0)/done:.1f}s/task", flush=True)
        n = len(exs)
        res = {"benchmark": bench, "n": n,
               "base_acc": agg["bc"] / n, "steer_acc": agg["sc"] / n,
               "base_avg_tokens": agg["bt"] / n, "steer_avg_tokens": agg["st"] / n,
               "acc_delta_pts": (agg["sc"] - agg["bc"]) / n * 100,
               "token_reduction_pct": (1 - agg["st"] / agg["bt"]) * 100 if agg["bt"] else 0}
        summary[bench] = res
        json.dump({"summary": res, "rows": rows}, open(f"{args.out_dir}/{bench}.json", "w"), indent=2)
        print(f"  -> {res['base_acc']*100:.1f}% -> {res['steer_acc']*100:.1f}% ({res['acc_delta_pts']:+.1f} pts) "
              f"| tokens {res['base_avg_tokens']:.0f} -> {res['steer_avg_tokens']:.0f} "
              f"(-{res['token_reduction_pct']:.0f}%)", flush=True)

    json.dump(summary, open(f"{args.out_dir}/summary.json", "w"), indent=2)
    print("\n================ SUMMARY ================", flush=True)
    print(f"{'benchmark':10} {'base_acc':>9} {'steer_acc':>10} {'Δpts':>7} {'base_tok':>9} {'steer_tok':>10} {'tok_red':>8}")
    for b, r in summary.items():
        print(f"{b:10} {r['base_acc']*100:8.1f}% {r['steer_acc']*100:9.1f}% {r['acc_delta_pts']:+6.1f} "
              f"{r['base_avg_tokens']:9.0f} {r['steer_avg_tokens']:10.0f} {r['token_reduction_pct']:7.0f}%")
    print(f"\n[eval] saved -> {args.out_dir}/summary.json (+ per-benchmark json)", flush=True)


if __name__ == "__main__":
    main()
