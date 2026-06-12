#!/usr/bin/env python3
"""Run the frontier-scale PACER audit.

The default 15K workload is the main evidence path; the 60K gate checks normal
scaling.  This script adds a deliberately compact 120K/240K frontier audit over
a compact fixed-seed query set so users can inspect whether PACER's
raw-verification work, certification status, and latency counters remain
meaningful beyond the main
workload without waiting for the full 320-query suite at each scale.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from trustvql import (
    IVFIndex,
    contract_planner,
    exact_ordered,
    exact_secure,
    generate_dataset,
    generate_queries,
    pacer_sliced_search,
    post_filter_ivf,
    recall_at_k,
)


def _metric_row(ds, p, gold: Dict[str, Any], res: Dict[str, Any], method: str, qi: int, n: int, k: int) -> Dict[str, Any]:
    pv, dv, tv = ds.violations(res["ids"], p)
    return {
        "n": int(n),
        "query_id": int(qi),
        "method": method,
        "recall_at_k": float(recall_at_k(gold["ids"], res["ids"], k=k)),
        "ordered_exact_secure_topk": int(exact_ordered(gold["ids"], res["ids"], k=k)),
        "raw_candidates": int(res.get("raw_candidates", 0)),
        "latency_ms": float(res.get("latency_ms", 0.0)),
        "policy_violations": int(pv),
        "deleted_violations": int(dv),
        "tenant_violations": int(tv),
        "certified": int(bool(res.get("certified", False))),
    }


def run_config(n: int, nlist: int, budget: int, per_regime: int, seed_base: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t = time.perf_counter()
    ds = generate_dataset(
        n=n,
        dim=64,
        true_clusters=nlist,
        tenants=64,
        regions=8,
        dtypes=8,
        prov_tags=16,
        seed=seed_base + n,
        delete_rate=0.12,
        tenant_cluster_correlation=0.82,
    )
    data_generation_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    index = IVFIndex(ds, nlist=nlist, seed=seed_base + 100 + n, iters=2)
    index_build_ms = (time.perf_counter() - t) * 1000.0

    queries = generate_queries(ds, per_regime=per_regime, seed=seed_base + 200 + n, correlation="positive")
    funcs = {
        "PostFilter-IVF": lambda p: post_filter_ivf(ds, index, p, k=10, nprobe=12, candidate_budget=budget),
        "PACER-A": lambda p: pacer_sliced_search(ds, index, p, k=10, candidate_budget=budget, force_certify=False),
        "PACER-C": lambda p: contract_planner(ds, index, p, k=10, candidate_budget=budget, service_level="certify"),
        "PACER-X": lambda p: pacer_sliced_search(ds, index, p, k=10, candidate_budget=ds.n, force_certify=True),
    }
    rows: list[dict[str, Any]] = []
    for qi, policy in enumerate(queries):
        print(f"[scale_frontier] n={n} query {qi + 1}/{len(queries)}", flush=True)
        gold = exact_secure(ds, policy, k=10)
        for method, fn in funcs.items():
            rows.append(_metric_row(ds, policy, gold, fn(policy), method, qi, n, 10))

    methods_summary = {}
    for method in funcs:
        xs = [r for r in rows if r["method"] == method]
        methods_summary[method] = {
            "recall_at_k": float(np.mean([r["recall_at_k"] for r in xs])),
            "ordered_exact_secure_topk": float(np.mean([r["ordered_exact_secure_topk"] for r in xs])),
            "raw_candidates_mean": float(np.mean([r["raw_candidates"] for r in xs])),
            "latency_ms_mean": float(np.mean([r["latency_ms"] for r in xs])),
            "latency_ms_p95": float(np.percentile([r["latency_ms"] for r in xs], 95)),
            "certified_fraction": float(np.mean([r["certified"] for r in xs])),
            "violations_total": int(sum(r["policy_violations"] + r["deleted_violations"] + r["tenant_violations"] for r in xs)),
        }
    summary = {
        "n": int(n),
        "nlist": int(nlist),
        "queries": int(len(queries)),
        "method_query_rows": int(len(rows)),
        "budget": int(budget),
        "data_generation_ms": float(data_generation_ms),
        "index_build_ms": float(index_build_ms),
        "index_overhead": index.index_overhead(),
        "methods_summary": methods_summary,
        "claim_checks": {
            "pacer_x_exact_failures": int(sum(1 - r["ordered_exact_secure_topk"] for r in rows if r["method"] == "PACER-X")),
            "safe_violations_total": int(sum(r["policy_violations"] + r["deleted_violations"] + r["tenant_violations"] for r in rows)),
            "pacer_a_raw_less_than_postfilter": bool(methods_summary["PACER-A"]["raw_candidates_mean"] < methods_summary["PostFilter-IVF"]["raw_candidates_mean"]),
        },
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PACER 120K/240K frontier-scale audit")
    ap.add_argument("--out", default=None)
    ap.add_argument("--configs", nargs="*", default=["120000:128:2400:4", "240000:160:2400:2"],
                    help="configs as n:nlist:budget:per_regime")
    ap.add_argument("--seed-base", type=int, default=3100)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "results" / "scale_frontier"
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for item in args.configs:
        n, nlist, budget, per_regime = (int(x) for x in item.split(":"))
        rows, summary = run_config(n, nlist, budget, per_regime, args.seed_base)
        all_rows.extend(rows)
        summaries.append(summary)

    with (out / "scale_frontier.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (out / "scale_frontier_summary.json").open("w") as f:
        json.dump({"schema": "pacer-scale-frontier-v1", "summaries": summaries, "rows": len(all_rows)}, f, indent=2)
    print(json.dumps({"status": "ok", "configs": len(summaries), "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
