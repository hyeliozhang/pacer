#!/usr/bin/env python3
"""Frozen-parameter blind robustness audit for PACER.

This run is designed to address the reviewer concern that the default synthetic
workload could be over-tuned.  It fixes the executor parameters once, then runs a
small factorial grid over independent seeds, policy/vector correlation regimes,
and deletion rates.  The script never changes budgets or index parameters based
on observed outcomes; the configuration manifest is written before evaluation
and hashed into the summary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from trustvql import (
    IVFIndex,
    bitmap_slice_exact,
    contract_planner,
    exact_ordered,
    exact_secure,
    generate_dataset,
    generate_queries,
    pacer_sliced_search,
    post_filter_ivf,
    prefilter_ivf,
    recall_at_k,
)


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    a = sorted(float(x) for x in xs)
    idx = min(len(a) - 1, max(0, int(round(q * (len(a) - 1)))))
    return float(a[idx])


def manifest() -> dict:
    return {
        "purpose": "frozen_parameter_blind_robustness_audit",
        "n": 3500,
        "dim": 64,
        "nlist": 48,
        "candidate_budget": 700,
        "k": 10,
        "per_regime": 4,
        "seeds": [11, 23, 37],
        "correlations": ["positive", "independent", "negative"],
        "delete_rates": [0.0, 0.12, 0.24],
        "tenant_cluster_correlation": {"positive": 0.82, "independent": 0.72, "negative": 0.72},
        "frozen_methods": ["PostFilter-IVF", "PreFilter-IVF", "BitmapSlice-Exact", "PACER-A", "PACER-C", "PACER-X"],
        "no_tuning_rule": "Budgets, k, nlist, and method parameters are fixed before any configuration is evaluated.",
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "blind_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = manifest()
    manifest_text = json.dumps(cfg, sort_keys=True, indent=2)
    (out_dir / "blind_config_manifest.json").write_text(manifest_text)
    manifest_sha256 = hashlib.sha256(manifest_text.encode()).hexdigest()

    rows: List[dict] = []
    config_rows: List[dict] = []
    t0_all = time.perf_counter()

    for seed in cfg["seeds"]:
        for correlation in cfg["correlations"]:
            for delete_rate in cfg["delete_rates"]:
                ds = generate_dataset(
                    n=int(cfg["n"]),
                    dim=int(cfg["dim"]),
                    true_clusters=max(48, int(cfg["nlist"])),
                    seed=int(seed),
                    delete_rate=float(delete_rate),
                    tenant_cluster_correlation=float(cfg["tenant_cluster_correlation"][correlation]),
                )
                index = IVFIndex(ds, nlist=int(cfg["nlist"]), seed=int(seed) + 1)
                queries = generate_queries(ds, per_regime=int(cfg["per_regime"]), seed=int(seed) + 2, correlation=str(correlation))
                methods: Dict[str, Callable] = {
                    "PostFilter-IVF": lambda p, ds=ds, index=index: post_filter_ivf(ds, index, p, k=int(cfg["k"]), nprobe=8, candidate_budget=int(cfg["candidate_budget"])),
                    "PreFilter-IVF": lambda p, ds=ds, index=index: prefilter_ivf(ds, index, p, k=int(cfg["k"]), nprobe=8, candidate_budget=int(cfg["candidate_budget"])),
                    "BitmapSlice-Exact": lambda p, ds=ds, index=index: bitmap_slice_exact(ds, index, p, k=int(cfg["k"])),
                    "PACER-A": lambda p, ds=ds, index=index: pacer_sliced_search(ds, index, p, k=int(cfg["k"]), candidate_budget=int(cfg["candidate_budget"]), use_bounds=True, force_certify=False, bound_mode="cone"),
                    "PACER-C": lambda p, ds=ds, index=index: contract_planner(ds, index, p, k=int(cfg["k"]), candidate_budget=int(cfg["candidate_budget"]), service_level="certify-if-cheap"),
                    "PACER-X": lambda p, ds=ds, index=index: pacer_sliced_search(ds, index, p, k=int(cfg["k"]), candidate_budget=ds.n, use_bounds=True, force_certify=True, bound_mode="cone"),
                }
                per_method = {m: {"recall": [], "exact": [], "raw": [], "lat": [], "viol": [], "cert": []} for m in methods}
                for qid, p in enumerate(queries):
                    gold = exact_secure(ds, p, k=int(cfg["k"]))
                    for name, fn in methods.items():
                        res = fn(p)
                        viol = sum(ds.violations(res["ids"], p))
                        rec = recall_at_k(gold["ids"], res["ids"], k=int(cfg["k"]))
                        exact = float(exact_ordered(gold["ids"], res["ids"], k=int(cfg["k"])))
                        raw = float(res.get("raw_candidates", res.get("candidates", 0)))
                        lat = float(res.get("latency_ms", 0.0))
                        per_method[name]["recall"].append(rec)
                        per_method[name]["exact"].append(exact)
                        per_method[name]["raw"].append(raw)
                        per_method[name]["lat"].append(lat)
                        per_method[name]["viol"].append(float(viol))
                        per_method[name]["cert"].append(float(bool(res.get("certified", False))))
                        rows.append({
                            "seed": seed,
                            "correlation": correlation,
                            "delete_rate": delete_rate,
                            "query_id": qid,
                            "method": name,
                            "recall_at_k": f"{rec:.6f}",
                            "secure_topk_exact": f"{exact:.6f}",
                            "raw_candidates": f"{raw:.3f}",
                            "latency_ms": f"{lat:.6f}",
                            "violations": int(viol),
                            "certified": int(bool(res.get("certified", False))),
                        })
                cfg_row = {"seed": seed, "correlation": correlation, "delete_rate": delete_rate, "queries": len(queries)}
                for name, vals in per_method.items():
                    cfg_row[f"{name}_recall"] = mean(vals["recall"])
                    cfg_row[f"{name}_exact"] = mean(vals["exact"])
                    cfg_row[f"{name}_raw"] = mean(vals["raw"])
                    cfg_row[f"{name}_latency_ms"] = mean(vals["lat"])
                    cfg_row[f"{name}_violations"] = int(sum(vals["viol"]))
                    cfg_row[f"{name}_certified_fraction"] = mean(vals["cert"])
                config_rows.append(cfg_row)

    csv_fields = ["seed", "correlation", "delete_rate", "query_id", "method", "recall_at_k", "secure_topk_exact", "raw_candidates", "latency_ms", "violations", "certified"]
    with (out_dir / "blind_robustness_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader(); w.writerows(rows)
    cfg_fields = list(config_rows[0].keys())
    with (out_dir / "blind_robustness_by_config.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfg_fields)
        w.writeheader(); w.writerows(config_rows)

    methods = cfg["frozen_methods"]
    summary_methods = {}
    for m in methods:
        exacts = [r[f"{m}_exact"] for r in config_rows]
        recalls = [r[f"{m}_recall"] for r in config_rows]
        raws = [r[f"{m}_raw"] for r in config_rows]
        lats = [r[f"{m}_latency_ms"] for r in config_rows]
        summary_methods[m] = {
            "mean_recall": mean(recalls),
            "min_recall_by_config": min(recalls),
            "p05_recall_by_config": quantile(recalls, 0.05),
            "mean_exact": mean(exacts),
            "min_exact_by_config": min(exacts),
            "p05_exact_by_config": quantile(exacts, 0.05),
            "mean_raw_candidates": mean(raws),
            "mean_latency_ms": mean(lats),
            "total_violations": int(sum(r[f"{m}_violations"] for r in config_rows)),
        }

    claim_checks = {
        "config_count": len(config_rows),
        "query_method_rows": len(rows),
        "manifest_sha256": manifest_sha256,
        "all_safe_method_violations": int(sum(int(r["violations"]) for r in rows if r["method"] in {"PostFilter-IVF", "PreFilter-IVF", "BitmapSlice-Exact", "PACER-A", "PACER-C", "PACER-X"})),
        "pacer_a_beats_postfilter_exact_configs": int(sum(r["PACER-A_exact"] > r["PostFilter-IVF_exact"] for r in config_rows)),
        "pacer_a_beats_postfilter_raw_configs": int(sum(r["PACER-A_raw"] < r["PostFilter-IVF_raw"] for r in config_rows)),
        "pacer_c_exact_configs": int(sum(r["PACER-C_exact"] == 1.0 for r in config_rows)),
        "pacer_x_exact_configs": int(sum(r["PACER-X_exact"] == 1.0 for r in config_rows)),
        "pacer_c_certificate_errors": int(sum(1 for r in rows if r["method"] == "PACER-C" and int(r["certified"]) == 1 and float(r["secure_topk_exact"]) < 1.0)),
        "pacer_x_certificate_errors": int(sum(1 for r in rows if r["method"] == "PACER-X" and int(r["certified"]) == 1 and float(r["secure_topk_exact"]) < 1.0)),
    }
    summary = {
        "manifest": cfg,
        "manifest_sha256": manifest_sha256,
        "configs": len(config_rows),
        "queries_per_config": int(cfg["per_regime"]) * 4,
        "query_method_rows": len(rows),
        "methods": summary_methods,
        "claim_checks": claim_checks,
        "runtime_seconds": round(time.perf_counter() - t0_all, 3),
    }
    (out_dir / "blind_robustness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["claim_checks"], indent=2))


if __name__ == "__main__":
    main()
