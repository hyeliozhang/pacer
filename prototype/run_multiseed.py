#!/usr/bin/env python3
"""Robustness check over repeated random seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np

from trustvql import run_suite

METHODS = ["PostFilter-IVF", "PreFilter-IVF", "PACER-A", "PACER-C", "PACER-X"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root / "results" / "multiseed"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[11, 23, 37])
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--per-regime", type=int, default=20)
    ap.add_argument("--nlist", type=int, default=48)
    ap.add_argument("--budget", type=int, default=1600)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for seed in args.seeds:
        summary = run_suite(
            out / f"seed_{seed}",
            n=args.n,
            dim=args.dim,
            per_regime=args.per_regime,
            seed=seed,
            nlist=args.nlist,
            candidate_budget=args.budget,
            correlation="positive",
            delete_rate=0.12,
            tenant_cluster_correlation=0.82,
        )
        for m in METHODS:
            v = summary["overall"][m]
            rows.append({
                "seed": seed,
                "method": m,
                "queries": summary["queries"],
                "recall_at_k": v["recall_at_k"],
                "secure_topk_exact": v["secure_topk_exact"],
                "latency_ms": v["latency_ms"],
                "policy_violation_per_query": v["policy_violations_per_query"] + v["deleted_violations_per_query"] + v["tenant_violations_per_query"],
                "certified_fraction": v["certified_fraction"],
                "certificate_errors": v["certificate_errors"],
            })
    if not rows:
        raise SystemExit("no rows generated")
    with (out / "multiseed_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    agg = []
    for m in METHODS:
        xs = [r for r in rows if r["method"] == m]
        for metric in ["recall_at_k", "secure_topk_exact", "latency_ms", "policy_violation_per_query", "certified_fraction", "certificate_errors"]:
            vals = np.array([float(r[metric]) for r in xs], dtype=float)
            agg.append({"method": m, "metric": metric, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "min": float(vals.min()), "max": float(vals.max())})
    with (out / "multiseed_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    with (out / "multiseed_summary.json").open("w") as f:
        json.dump({"rows": rows, "summary": agg, "parameters": vars(args)}, f, indent=2)
    print(json.dumps({"rows": len(rows), "seeds": args.seeds, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
