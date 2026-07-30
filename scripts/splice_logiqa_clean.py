"""Restate the LogiQA results on the contamination-free eval set.

The original runs scored data/LogiQA/eval_rand42_500.json, 16 of whose items also
appear in the attempted train rows. eval_rand42_500_clean.json swaps those 16 for
backfills. This script takes the original 500-row results plus a 16-row patch run
over just the backfills, and writes the restated 500-row result.

  python scripts/splice_logiqa_clean.py \
      --orig_dir  results/.../rand42_500 \
      --patch_dir results/.../patch16 \
      --out_dir   results/.../rand42_500_clean
"""
import argparse
import json
import os
import sys

# This lives in scripts/, so sys.path[0] is scripts/ — put the repo root on the
# path so `python scripts/splice_logiqa_clean.py` works from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic_utils import logic_eval_main


def load(path):
    with open(path) as fin:
        return [json.loads(line) for line in fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="500-row run scored on eval_rand42_500.json")
    ap.add_argument("--patch_dir", required=True, help="16-row run scored on eval_patch16.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--orig_sel", default="data/LogiQA/eval_rand42_500.json")
    ap.add_argument("--clean_sel", default="data/LogiQA/eval_rand42_500_clean.json")
    ap.add_argument("--patch_sel", default="data/LogiQA/eval_patch16.json")
    args = ap.parse_args()

    orig_idx = [i["pool_idx"] for i in json.load(open(args.orig_sel))["items"]]
    clean_idx = [i["pool_idx"] for i in json.load(open(args.clean_sel))["items"]]
    patch_idx = [i["pool_idx"] for i in json.load(open(args.patch_sel))["items"]]

    orig = load(os.path.join(args.orig_dir, "predictions.jsonl"))
    patch = load(os.path.join(args.patch_dir, "predictions.jsonl"))
    if len(orig) != len(orig_idx):
        raise SystemExit(f"orig has {len(orig)} rows, selection has {len(orig_idx)}")
    if len(patch) != len(patch_idx):
        raise SystemExit(f"patch has {len(patch)} rows, selection has {len(patch_idx)}")

    by_pool = dict(zip(orig_idx, orig))
    by_pool.update(zip(patch_idx, patch))

    missing = [i for i in clean_idx if i not in by_pool]
    if missing:
        raise SystemExit(f"no prediction for pool_idx {missing[:5]} (n={len(missing)})")

    merged = [by_pool[i] for i in clean_idx]
    kept = sum(1 for i in clean_idx if i in orig_idx)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "predictions.jsonl")
    with open(out, "w") as fout:
        for row in merged:
            fout.write(json.dumps(row) + "\n")
    print(f"merged {len(merged)} rows -> {out}  ({kept} reused, {len(merged) - kept} from patch)")

    acc = logic_eval_main(out, save=True, output_dir=args.out_dir)
    print(f"restated accuracy on the clean set: {acc:.4f}")


if __name__ == "__main__":
    main()
