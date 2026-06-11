"""Snapshot-insertion/deletion stream checks for PACER.

The physical index is intentionally built over all rows, including rows whose
insert epoch is in the future and rows that are tombstoned at some snapshots.
The stream checker asks whether query execution leaks future or deleted rows at
each epoch.  This is a storage-freshness workload, not a static filtering toy.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Dict, List

import numpy as np

from trustvql import (
    IVFIndex,
    QueryPolicy,
    caps_search, pacer_sliced_search,
    exact_ordered,
    exact_secure,
    generate_dataset,
    generate_queries,
    post_filter_ivf,
    stale_no_recheck,
    naive_no_policy_ivf,
    recall_at_k,
)


def split_epoch_violations(ds, ids, p: QueryPolicy) -> Dict[str, int]:
    ids = np.asarray(list(ids), dtype=np.int64)
    if ids.size == 0:
        return {"future": 0, "deleted": 0, "policy": 0, "tenant": 0}
    ins = ds.insert_epochs()[ids]
    dele = ds.delete_epoch[ids]
    e = int(p.epoch)
    future = int(np.sum(ins > e))
    deleted = int(np.sum(dele <= e))
    policy, _, tenant = ds.violations(ids, p)
    return {"future": future, "deleted": deleted, "policy": int(policy), "tenant": int(tenant)}


def run_snapshot_stream(out_dir: Path, n: int = 12000, dim: int = 64, k: int = 10, seed: int = 1777, nlist: int = 64, candidate_budget: int = 1200) -> Dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = generate_dataset(n=n, dim=dim, true_clusters=64, tenants=24, regions=6, dtypes=5, prov_tags=8, delete_rate=0.18, tenant_cluster_correlation=0.78, seed=seed)
    index = IVFIndex(ds, nlist=nlist, seed=seed + 1, iters=4)
    base_queries = generate_queries(ds, per_regime=8, seed=seed + 2, correlation="positive")
    epochs = list(range(0, 12))
    rows: List[Dict] = []
    methods = {
        "PostFilter-IVF": lambda p: post_filter_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
        "StaleNoRecheck": lambda p: stale_no_recheck(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
        "NaiveNoPolicy": lambda p: naive_no_policy_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
        "PACER-A": lambda p: pacer_sliced_search(ds, index, p, k=k, candidate_budget=candidate_budget, use_bounds=True, force_certify=False),
        "PACER-X": lambda p: pacer_sliced_search(ds, index, p, k=k, candidate_budget=ds.n, use_bounds=True, force_certify=True),
    }
    for epoch in epochs:
        for qi, p0 in enumerate(base_queries):
            p = QueryPolicy(p0.qvec, p0.tenant_mask, p0.max_sensitivity, p0.region_mask, p0.dtype_mask, p0.provenance_mask, epoch, p0.regime, p0.source_id, p0.policy_anchor_id, p0.correlation)
            gold = exact_secure(ds, p, k=k)
            visible_count = int(ds.policy_mask(p, include_deletions=False).sum())
            future_in_index = int(np.sum(ds.insert_epochs() > epoch))
            stale_in_index = int(np.sum(ds.delete_epoch <= epoch))
            for name, fn in methods.items():
                res = fn(p)
                parts = split_epoch_violations(ds, res["ids"], p)
                rows.append({
                    "epoch": int(epoch),
                    "query_id": int(qi),
                    "regime": p.regime,
                    "method": name,
                    "visible_count": visible_count,
                    "future_postings_in_index": future_in_index,
                    "stale_postings_in_index": stale_in_index,
                    "returned": int(len(res["ids"])),
                    "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=k),
                    "secure_topk_exact": int(exact_ordered(gold["ids"], res["ids"], k=k)),
                    "future_output_violations": parts["future"],
                    "deleted_output_violations": parts["deleted"],
                    "policy_output_violations": parts["policy"],
                    "tenant_output_violations": parts["tenant"],
                    "certified": int(bool(res.get("certified", False))),
                    "raw_candidates": int(res.get("raw_candidates", 0)),
                    "candidates": int(res.get("candidates", 0)),
                    "policy_checks": int(res.get("policy_checks", 0)),
                })
    with (out_dir / "snapshot_stream.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    methods_seen = sorted(set(r["method"] for r in rows))
    summary = {"n": int(n), "queries_per_epoch": int(len(base_queries)), "epochs": epochs, "rows": int(len(rows)), "methods": {}, "claim_checks": {}}
    for m in methods_seen:
        xs = [r for r in rows if r["method"] == m]
        summary["methods"][m] = {
            "recall_at_k": float(np.mean([r["recall_at_k"] for r in xs])),
            "secure_topk_exact": float(np.mean([r["secure_topk_exact"] for r in xs])),
            "future_output_violations_total": int(sum(r["future_output_violations"] for r in xs)),
            "deleted_output_violations_total": int(sum(r["deleted_output_violations"] for r in xs)),
            "policy_output_violations_total": int(sum(r["policy_output_violations"] for r in xs)),
            "tenant_output_violations_total": int(sum(r["tenant_output_violations"] for r in xs)),
            "certified_fraction": float(np.mean([r["certified"] for r in xs])),
            "raw_candidates": float(np.mean([r["raw_candidates"] for r in xs])),
        }
    safe = ["PostFilter-IVF", "PACER-A", "PACER-X"]
    summary["claim_checks"] = {
        "safe_future_deleted_policy_tenant_violations": int(sum(summary["methods"][m]["future_output_violations_total"] + summary["methods"][m]["deleted_output_violations_total"] + summary["methods"][m]["policy_output_violations_total"] + summary["methods"][m]["tenant_output_violations_total"] for m in safe)),
        "stale_control_epoch_violations": int(summary["methods"]["StaleNoRecheck"]["future_output_violations_total"] + summary["methods"]["StaleNoRecheck"]["deleted_output_violations_total"]),
        "naive_policy_violations": int(summary["methods"]["NaiveNoPolicy"]["policy_output_violations_total"] + summary["methods"]["NaiveNoPolicy"]["tenant_output_violations_total"]),
        "pacer_x_exact_failures": int(sum(1 - r["secure_topk_exact"] for r in rows if r["method"] == "PACER-X")),
    }
    with (out_dir / "snapshot_stream_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/snapshot_stream"))
    args = ap.parse_args()
    s = run_snapshot_stream(args.out)
    print(json.dumps(s, indent=2))
