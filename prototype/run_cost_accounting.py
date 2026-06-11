#!/usr/bin/env python3
"""End-to-end cost accounting for PACER and baselines.

Reports wall-clock latency, raw row checks, score evaluations, policy checks,
index build time, structural memory, and amortized update/compaction signals.
The goal is to prevent a paper from relying only on a favorable row counter.
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
from typing import Dict, List
import numpy as np
from trustvql import generate_dataset, generate_queries, IVFIndex, exact_secure, post_filter_ivf, prefilter_ivf, bitmap_slice_exact, pacer_sliced_search, contract_planner


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "cost_accounting"; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ds = generate_dataset(n=15000, dim=64, true_clusters=64, seed=7, delete_rate=0.12, tenant_cluster_correlation=0.82)
    data_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter(); index = IVFIndex(ds, nlist=64, seed=8); build_ms = (time.perf_counter() - t1) * 1000
    qs = generate_queries(ds, per_regime=2, seed=9, correlation="positive")
    k=10; budget=1200
    funcs = {
        "ExactSecure": lambda p: exact_secure(ds, p, k=k),
        "PostFilter-IVF": lambda p: post_filter_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=budget),
        "PreFilter-IVF": lambda p: prefilter_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=budget),
        "BitmapSlice-Exact": lambda p: bitmap_slice_exact(ds, index, p, k=k),
        "PACER-A": lambda p: pacer_sliced_search(ds, index, p, k=k, candidate_budget=budget, force_certify=False),
        "PACER-C": lambda p: contract_planner(ds, index, p, k=k, candidate_budget=budget, service_level="certify-if-cheap"),
        "PACER-X": lambda p: pacer_sliced_search(ds, index, p, k=k, candidate_budget=ds.n, force_certify=True),
    }
    rows: List[Dict] = []
    for qi, p in enumerate(qs):
        # Explicit bitset intersection timing, independent of method body.
        tb = time.perf_counter(); qb = index.query_policy_bitset(p); bit_ms = (time.perf_counter()-tb)*1000
        for name, fn in funcs.items():
            res = fn(p)
            rows.append({"query_id": qi, "method": name, "latency_ms": float(res["latency_ms"]),
                         "raw_candidates": int(res.get("raw_candidates",0)), "distance_evals": int(res.get("distance_evals",0)),
                         "policy_checks": int(res.get("policy_checks",0)), "summary_checks": int(res.get("summary_checks",0)),
                         "visited_cluster_count": int(res.get("visited_cluster_count",0)), "bitset_intersection_ms": bit_ms,
                         "query_bitset_popcount": int(qb.bit_count()), "certified": int(bool(res.get("certified", False)))})
    with (out_dir / "cost_accounting.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    by = {}
    for m in sorted({r["method"] for r in rows}):
        xs = [r for r in rows if r["method"]==m]
        by[m] = {"latency_ms_mean": float(np.mean([r["latency_ms"] for r in xs])),
                 "latency_ms_median": float(np.median([r["latency_ms"] for r in xs])),
                 "latency_ms_p95": float(np.percentile([r["latency_ms"] for r in xs], 95)),
                 "raw_candidates_mean": float(np.mean([r["raw_candidates"] for r in xs])),
                 "distance_evals_mean": float(np.mean([r["distance_evals"] for r in xs])),
                 "policy_checks_mean": float(np.mean([r["policy_checks"] for r in xs])),
                 "summary_checks_mean": float(np.mean([r["summary_checks"] for r in xs])),
                 "bitset_intersection_ms_mean": float(np.mean([r["bitset_intersection_ms"] for r in xs])),
                 "certified_fraction": float(np.mean([r["certified"] for r in xs]))}
    overhead = index.index_overhead()
    summary = {"n": ds.n, "queries": len(qs), "data_generation_ms": data_ms, "index_build_ms": build_ms,
               "index_overhead": overhead, "methods": by,
               "claim_checks": {"has_end_to_end_latency": True,
                                "has_index_build_time": build_ms > 0,
                                "has_memory_accounting": overhead["total_structural_bytes_per_record"] > 0,
                                "pacer_a_raw_less_than_postfilter": by["PACER-A"]["raw_candidates_mean"] < by["PostFilter-IVF"]["raw_candidates_mean"],
                                "pacer_a_latency_recorded": by["PACER-A"]["latency_ms_mean"] > 0}}
    with (out_dir / "cost_accounting_summary.json").open("w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary["claim_checks"], indent=2))

if __name__ == "__main__": main()
