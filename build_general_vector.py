"""Phase 1 / Phase 2 pooler for a multi-domain S_general steering vector.

Phase 1 (default): pool ALL layer-L boundary vectors from every domain's
hidden.pt files (count-weighted — no balancing) and compute:

    S = mean(check ∪ switch) − mean(other)

Phase 2 (--balance): equalize boundary-vector counts across domains via
budget = min(n_domain) then seeded subsample, then the same contrast.

Domain keyword tagging already happened at hidden_analysis time; this
script only reads the tagged check/switch/other indices from hidden.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch


def _parse_domain(spec: str) -> Tuple[str, List[str]]:
    """Parse --domain name=path1,path2[,...] into (name, [paths])."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--domain expected name=path[,path...], got: {spec!r}"
        )
    name, paths_str = spec.split("=", 1)
    name = name.strip()
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    if not name or not paths:
        raise argparse.ArgumentTypeError(
            f"--domain expected name=path[,path...], got: {spec!r}"
        )
    return name, paths


def collect_from_hidden(
    path: str, layer: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Load one hidden.pt and return lists of check / switch / other row tensors."""
    data = torch.load(path, weights_only=False, map_location="cpu")
    if layer >= len(data) or data[layer] is None:
        raise ValueError(f"layer {layer} missing in {path} (len={len(data)})")
    layer_data = data[layer]
    check, switch, other = [], [], []
    for k in layer_data:
        h = layer_data[k]["step"]
        check_index = layer_data[k]["check_index"]
        switch_index = layer_data[k]["switch_index"]
        if h.shape[0] == 0:
            continue
        check.append(h[check_index])
        switch.append(h[switch_index])
        all_indices = torch.arange(h.shape[0])
        mask = ~(torch.isin(all_indices, check_index) | torch.isin(all_indices, switch_index))
        other.append(h[mask])
    return check, switch, other


def _cat_or_empty(parts: List[torch.Tensor], dim: int) -> torch.Tensor:
    nonempty = [p for p in parts if p.numel() > 0]
    if not nonempty:
        return torch.empty(0, dim)
    return torch.cat(nonempty, dim=0)


def load_domain(
    name: str, paths: List[str], layer: int
) -> Dict[str, torch.Tensor]:
    """Concatenate check/switch/other across all hidden.pt paths for one domain."""
    check_parts, switch_parts, other_parts = [], [], []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"domain {name}: missing hidden.pt at {path}")
        c, s, o = collect_from_hidden(path, layer)
        check_parts.extend(c)
        switch_parts.extend(s)
        other_parts.extend(o)

    # Infer dim from any non-empty block; fall back after we know something exists.
    dim = None
    for parts in (check_parts, switch_parts, other_parts):
        for p in parts:
            if p.numel() > 0:
                dim = p.shape[-1]
                break
        if dim is not None:
            break
    if dim is None:
        raise ValueError(f"domain {name}: no boundary vectors found in {paths}")

    check = _cat_or_empty(check_parts, dim)
    switch = _cat_or_empty(switch_parts, dim)
    other = _cat_or_empty(other_parts, dim)
    return {"check": check, "switch": switch, "other": other}


def subsample_domain(
    rows: Dict[str, torch.Tensor], budget: int, rng: random.Random
) -> Dict[str, torch.Tensor]:
    """Phase 2 stub: keep exactly `budget` labeled boundary vectors, preserving labels.

    Draws a random subset of the concatenated (vector, kind) rows so that the
    label mix of the kept set mirrors a uniform sample over all boundary vectors.
    """
    labeled: List[Tuple[str, int]] = []
    for kind in ("check", "switch", "other"):
        n = rows[kind].shape[0]
        labeled.extend((kind, i) for i in range(n))
    if budget > len(labeled):
        raise ValueError(
            f"budget={budget} exceeds domain boundary count={len(labeled)}"
        )
    kept = rng.sample(labeled, budget)
    by_kind: Dict[str, List[int]] = {"check": [], "switch": [], "other": []}
    for kind, i in kept:
        by_kind[kind].append(i)
    out = {}
    for kind in ("check", "switch", "other"):
        idx = by_kind[kind]
        if not idx:
            out[kind] = rows[kind][:0]
        else:
            out[kind] = rows[kind][torch.tensor(idx, dtype=torch.long)]
    return out


def counts(rows: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {
        "check": int(rows["check"].shape[0]),
        "switch": int(rows["switch"].shape[0]),
        "other": int(rows["other"].shape[0]),
        "total": int(
            rows["check"].shape[0] + rows["switch"].shape[0] + rows["other"].shape[0]
        ),
        "check_switch": int(rows["check"].shape[0] + rows["switch"].shape[0]),
    }


def build_vector(pooled: Dict[str, torch.Tensor]) -> torch.Tensor:
    rt = torch.cat([pooled["check"], pooled["switch"]], dim=0)
    other = pooled["other"]
    if rt.shape[0] == 0:
        raise ValueError("no check/switch (reflection∪transition) vectors to average")
    if other.shape[0] == 0:
        raise ValueError("no other (execution) vectors to average")
    return rt.mean(dim=0) - other.mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool multi-domain boundary vectors into S_general (Phase 1 or 2)."
    )
    parser.add_argument(
        "--domain",
        action="append",
        type=_parse_domain,
        required=True,
        metavar="NAME=HIDDEN.PT[,HIDDEN.PT...]",
        help="Repeatable. Example: --domain math=a/hidden.pt,b/hidden.pt "
             "--domain apps=c/hidden.pt,d/hidden.pt",
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Phase 2: equalize boundary-vector counts across domains "
             "(budget=min(n_domain), seeded subsample). Default = Phase 1 (pool all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for Phase 2 subsampling (ignored in Phase 1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/general/S_general_math_apps_phase1.pt",
        help="Output .pt path (sibling .meta.json is also written).",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="NAME=PATH.pt",
        help="Optional: cosine-compare the built vector against NAME=path.pt "
             "(repeatable). Logged to stdout and meta.",
    )
    args = parser.parse_args()

    domains: Dict[str, Dict[str, torch.Tensor]] = {}
    domain_paths: Dict[str, List[str]] = {}
    for name, paths in args.domain:
        if name in domains:
            raise ValueError(f"duplicate --domain name: {name}")
        print(f"[load] domain={name} paths={paths}")
        domains[name] = load_domain(name, paths, args.layer)
        domain_paths[name] = paths

    per_domain_counts = {name: counts(rows) for name, rows in domains.items()}
    for name, c in per_domain_counts.items():
        print(
            f"[counts] {name}: total={c['total']} "
            f"check={c['check']} switch={c['switch']} other={c['other']} "
            f"check+switch={c['check_switch']}"
        )

    phase = 2 if args.balance else 1
    budget = None
    if args.balance:
        # Phase 2: equalize boundary counts across domains.
        totals = {name: c["total"] for name, c in per_domain_counts.items()}
        budget = min(totals.values())
        print(f"[balance] budget=min(...)={budget} from {totals}")
        rng = random.Random(args.seed)
        domains = {
            name: subsample_domain(rows, budget, rng) for name, rows in domains.items()
        }
        per_domain_counts = {name: counts(rows) for name, rows in domains.items()}
        for name, c in per_domain_counts.items():
            print(
                f"[counts-after-balance] {name}: total={c['total']} "
                f"check={c['check']} switch={c['switch']} other={c['other']}"
            )
    else:
        print("[phase1] pooling ALL boundary vectors (no cross-domain balancing)")

    pooled = {
        "check": torch.cat([d["check"] for d in domains.values()], dim=0),
        "switch": torch.cat([d["switch"] for d in domains.values()], dim=0),
        "other": torch.cat([d["other"] for d in domains.values()], dim=0),
    }
    pooled_counts = counts(pooled)
    print(
        f"[pooled] total={pooled_counts['total']} "
        f"check+switch={pooled_counts['check_switch']} other={pooled_counts['other']}"
    )

    steer_vec = build_vector(pooled)
    norm = float(steer_vec.norm().item())
    print(f"[vector] dim={steer_vec.numel()} norm={norm:.6f} phase={phase}")

    comparisons = {}
    for spec in args.compare:
        if "=" not in spec:
            raise ValueError(f"--compare expected NAME=path.pt, got {spec!r}")
        cname, cpath = spec.split("=", 1)
        other = torch.load(cpath, weights_only=False, map_location="cpu")
        if isinstance(other, dict):
            # tolerate accidental meta/dict wraps
            raise ValueError(f"--compare {cname}: expected a 1-D tensor at {cpath}")
        sim = cosine(steer_vec, other.float())
        comparisons[cname] = {"path": cpath, "cosine": sim, "norm": float(other.float().norm().item())}
        print(f"[compare] cos(S_general, {cname}) = {sim:.6f}")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(steer_vec, out_path)
    meta = {
        "name": os.path.splitext(os.path.basename(out_path))[0],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": phase,
        "balance": bool(args.balance),
        "budget": budget,
        "seed": args.seed if args.balance else None,
        "layer": args.layer,
        "sign_convention": "vector = mean(check ∪ switch) − mean(other); "
                           "apply by ADDING coef * vector with coef = -1.0",
        "apply_coef": -1.0,
        "domains": {
            name: {
                "paths": domain_paths[name],
                "counts": per_domain_counts[name],
            }
            for name in domains
        },
        "pooled_counts": pooled_counts,
        "vector_dim": int(steer_vec.numel()),
        "vector_norm": norm,
        "comparisons": comparisons,
        "out": out_path,
    }
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[save] {meta_path}")


if __name__ == "__main__":
    main()
