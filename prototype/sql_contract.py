#!/usr/bin/env python3
"""SQL-fragment visibility workload for PACER.

The PACER logical contract is not tied to one hard-coded conjunction. A relational
policy plan can evaluate a SQL-style visibility predicate into a snapshot bitmap;
PACER then uses no-false-negative per-cluster visibility summaries and the same
row verifier/certificate boundary. This script exercises DNF, semi-join,
anti-join, and mixed OR predicates without requiring an external SQL server.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import time
from typing import Dict, List

import numpy as np

from trustvql import (
    IVFIndex,
    _normalize,
    _topk_from_scores,
    exact_ordered,
    failure_mode,
    generate_dataset,
    recall_at_k,
)


def exact_sql(ds, qvec: np.ndarray, visible: np.ndarray, k: int = 10) -> Dict:
    t0 = time.perf_counter()
    ids = np.flatnonzero(visible).astype(np.int64)
    scores = ds.vectors[ids] @ qvec if ids.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(ids, scores, k)
    return {
        "ids": top_ids,
        "scores": top_scores,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "candidates": int(ids.size),
        "raw_candidates": int(ds.n),
        "policy_checks": int(ds.n),
        "certified": True,
        "visited_cluster_count": 0,
        "certificate_gap": 0.0,
    }


def post_sql(ds, index: IVFIndex, qvec: np.ndarray, visible: np.ndarray, k: int = 10, nprobe: int = 8, budget: int = 1200) -> Dict:
    """Global IVF post-filter baseline with score-ordered truncation.

    Earlier drafts accidentally truncated each posting list in physical row order.
    A strict baseline must first score the probed candidate pool and only then
    apply the global row budget before SQL verification.
    """
    t0 = time.perf_counter()
    order = np.argsort(-(index.centroids @ qvec))[: min(nprobe, index.nlist)]
    raw_pool = np.concatenate([index.cluster_ids[int(c0)] for c0 in order]).astype(np.int64) if len(order) else np.empty(0, dtype=np.int64)
    if raw_pool.size > int(budget):
        raw_scores = ds.vectors[raw_pool] @ qvec
        sel = np.argpartition(-raw_scores, int(budget) - 1)[: int(budget)]
        raw = raw_pool[sel]
    else:
        raw = raw_pool
    legal = raw[visible[raw]] if raw.size else np.empty(0, dtype=np.int64)
    scores = ds.vectors[legal] @ qvec if legal.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(legal, scores, k)
    return {
        "ids": top_ids,
        "scores": top_scores,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "candidates": int(legal.size),
        "raw_candidates": int(raw.size),
        "policy_checks": int(raw.size),
        "certified": False,
        "visited_cluster_count": int(len(order)),
        "certificate_gap": 0.0,
    }


def pacer_sql(ds, index: IVFIndex, qvec: np.ndarray, visible: np.ndarray, k: int = 10, budget: int = 1200, force_certify: bool = False) -> Dict:
    """PACER over a query-specific SQL visibility bitmap.

    The per-cluster visible-count summary is a no-false-negative unit summary
    for the already-evaluated SQL predicate. This tests the paper's compositional
    claim: arbitrary row-verifier predicates are safe if only conservative
    summaries are used before final verification.
    """
    t0 = time.perf_counter()
    upper = index.cluster_upper_bounds(qvec)
    order = np.argsort(-upper)
    visible_ids = np.flatnonzero(visible)
    visible_counts = np.bincount(index.assignments[visible_ids], minlength=index.nlist) if visible_ids.size else np.zeros(index.nlist, dtype=np.int32)
    may = visible_counts > 0
    processed = np.zeros(index.nlist, dtype=bool)
    top_ids = np.empty(0, dtype=np.int64)
    top_scores = np.empty(0, dtype=np.float32)
    raw = 0
    legal_total = 0
    checks = 0
    visited = 0
    certified = False
    gap = 0.0
    row_budget = ds.n if force_certify else int(budget)
    partial_budget_stop = False
    for c0 in order:
        c = int(c0)
        if not may[c]:
            processed[c] = True
            continue
        ids_full = index.cluster_ids[c]
        if not force_certify and raw >= row_budget:
            break
        if not force_certify and raw + int(ids_full.size) > row_budget:
            room = max(0, row_budget - raw)
            if room <= 0:
                break
            local_scores = ds.vectors[ids_full] @ qvec
            take_pos = np.argpartition(-local_scores, room - 1)[:room]
            ids = ids_full[take_pos]
            partial_budget_stop = True
        else:
            ids = ids_full
            processed[c] = True
        visited += 1
        raw += int(ids.size)
        checks += int(ids.size)
        legal = ids[visible[ids]]
        if legal.size:
            scores = ds.vectors[legal] @ qvec
            merge_ids = np.concatenate([top_ids, legal.astype(np.int64)]) if top_ids.size else legal.astype(np.int64)
            merge_scores = np.concatenate([top_scores, scores.astype(np.float32)]) if top_scores.size else scores.astype(np.float32)
            top_ids, top_scores = _topk_from_scores(merge_ids, merge_scores, k)
            legal_total += int(legal.size)
        if partial_budget_stop:
            break
        rem = np.flatnonzero((~processed) & may)
        maxub = float(np.max(upper[rem])) if rem.size else -math.inf
        if top_scores.size >= k:
            kth = float(top_scores[-1])
            gap = max(0.0, maxub - kth) if math.isfinite(maxub) else 0.0
            if kth >= maxub:
                certified = True
                break
        elif rem.size == 0:
            certified = True
            break
    if not certified and not np.any((~processed) & may):
        certified = True
        gap = 0.0
    return {
        "ids": top_ids,
        "scores": top_scores,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "candidates": int(legal_total),
        "raw_candidates": int(raw),
        "policy_checks": int(checks),
        "certified": bool(certified),
        "visited_cluster_count": int(visited),
        "certificate_gap": float(gap),
    }


def build_sql_queries(ds, nqueries: int = 128, seed: int = 803) -> List[Dict]:
    rng = np.random.default_rng(seed)
    auth_groups = rng.integers(0, 32, size=ds.n)
    deny_groups = rng.integers(0, 19, size=ds.n)
    queries: List[Dict] = []
    kinds = ["dnf", "semijoin", "antijoin", "mixed"]
    attempts = 0
    while len(queries) < nqueries:
        attempts += 1
        if attempts > nqueries * 2500:
            raise RuntimeError("not enough SQL policy queries")
        src = int(rng.integers(0, ds.n))
        qvec = _normalize(ds.vectors[src] + rng.normal(0, 0.15, ds.dim).astype(np.float32))[0]
        kind = kinds[len(queries) % len(kinds)]
        epoch = int(rng.integers(4, 9))
        live = ds.delete_epoch > epoch
        if kind == "dnf":
            tenants = np.unique(ds.tenant)
            regions = np.unique(ds.region)
            dtypes = np.unique(ds.dtype)
            tset = set(int(x) for x in rng.choice(tenants, size=min(4, len(tenants)), replace=False)); tset.add(int(ds.tenant[src]))
            rset = set(int(x) for x in rng.choice(regions, size=min(3, len(regions)), replace=False)); rset.add(int(ds.region[src]))
            dset = set(int(x) for x in rng.choice(dtypes, size=min(2, len(dtypes)), replace=False)); dset.add(int(ds.dtype[src]))
            ptag = int(rng.integers(0, int(ds.metadata["prov_tags"])))
            visible = (((np.isin(ds.tenant, list(tset)) & np.isin(ds.region, list(rset))) |
                        (np.isin(ds.dtype, list(dset)) & (ds.sensitivity <= int(ds.sensitivity[src])))) &
                       ((ds.provenance & (1 << ptag)) != 0) & live)
        elif kind == "semijoin":
            user_groups = set(int(x) for x in rng.choice(np.arange(32), size=5, replace=False)); user_groups.add(int(auth_groups[src]))
            ptag = int(rng.integers(0, int(ds.metadata["prov_tags"])))
            visible = (np.isin(auth_groups, list(user_groups)) & ((ds.provenance & (1 << ptag)) != 0) & (ds.sensitivity <= 3) & live)
        elif kind == "antijoin":
            deny = set(int(x) for x in rng.choice(np.arange(19), size=3, replace=False))
            tenants = np.unique(ds.tenant)
            tset = set(int(x) for x in rng.choice(tenants, size=min(8, len(tenants)), replace=False)); tset.add(int(ds.tenant[src]))
            visible = (np.isin(ds.tenant, list(tset)) & ~np.isin(deny_groups, list(deny)) & (ds.sensitivity <= 2) & live)
        else:
            t = int(ds.tenant[src]); r = int(ds.region[src]); d = int(ds.dtype[src])
            visible = (((ds.tenant == t) & (ds.sensitivity <= 2)) |
                       ((ds.region == r) & (ds.dtype == d)) |
                       ((ds.provenance & int(ds.provenance[src])) != 0)) & live
        if int(visible.sum()) >= 10:
            queries.append({"qvec": qvec.astype(np.float32), "visible": visible.astype(bool), "kind": kind, "epoch": epoch})
    return queries


def run_sql_contract(out_dir: Path, n: int = 9000, dim: int = 64, nqueries: int = 128, seed: int = 801, k: int = 10, budget: int = 1200) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = generate_dataset(n=n, dim=dim, true_clusters=64, tenants=24, regions=6, dtypes=5, prov_tags=8,
                          delete_rate=0.15, tenant_cluster_correlation=0.70, seed=seed)
    index = IVFIndex(ds, nlist=64, seed=seed + 1, iters=4)
    queries = build_sql_queries(ds, nqueries=nqueries, seed=seed + 2)
    methods = {
        "ExactSQL": lambda q: exact_sql(ds, q["qvec"], q["visible"], k),
        "PostFilter-SQL": lambda q: post_sql(ds, index, q["qvec"], q["visible"], k, nprobe=8, budget=budget),
        "PACER-A-SQL": lambda q: pacer_sql(ds, index, q["qvec"], q["visible"], k, budget=budget, force_certify=False),
        "PACER-X-SQL": lambda q: pacer_sql(ds, index, q["qvec"], q["visible"], k, budget=ds.n, force_certify=True),
    }
    rows: List[Dict] = []
    for qi, q in enumerate(queries):
        gold = methods["ExactSQL"](q)
        for m, fn in methods.items():
            res = fn(q)
            exact = exact_ordered(gold["ids"], res["ids"], k)
            cert = bool(res.get("certified", False))
            rows.append({
                "query_id": qi,
                "kind": q["kind"],
                "method": m,
                "n": ds.n,
                "visible_count": int(q["visible"].sum()),
                "recall_at_k": recall_at_k(gold["ids"], res["ids"], k),
                "secure_topk_exact": int(exact),
                "failure_mode": failure_mode(gold["ids"], res["ids"], k),
                "latency_ms": float(res["latency_ms"]),
                "candidates": int(res["candidates"]),
                "raw_candidates": int(res["raw_candidates"]),
                "policy_checks": int(res["policy_checks"]),
                "certified": int(cert),
                "certificate_false_positive": int(cert and not exact),
            })
    with (out_dir / "sql_policy_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary: Dict[str, Dict] = {"n": ds.n, "queries": len(queries), "methods": {}, "by_kind": {}}
    for m in methods:
        xs = [r for r in rows if r["method"] == m]
        cert = [r for r in xs if r["certified"] == 1]
        summary["methods"][m] = {
            "recall_at_k": float(np.mean([r["recall_at_k"] for r in xs])),
            "secure_topk_exact": float(np.mean([r["secure_topk_exact"] for r in xs])),
            "median_latency_ms": float(np.median([r["latency_ms"] for r in xs])),
            "candidates": float(np.mean([r["candidates"] for r in xs])),
            "raw_candidates": float(np.mean([r["raw_candidates"] for r in xs])),
            "certified_fraction": float(np.mean([r["certified"] for r in xs])),
            "certificate_errors": int(sum(r["certificate_false_positive"] for r in xs)),
            "certified_sound_fraction": float(np.mean([r["secure_topk_exact"] for r in cert])) if cert else 1.0,
        }
    for kind in sorted(set(r["kind"] for r in rows)):
        summary["by_kind"][kind] = {}
        for m in methods:
            xs = [r for r in rows if r["kind"] == kind and r["method"] == m]
            summary["by_kind"][kind][m] = {
                "recall_at_k": float(np.mean([r["recall_at_k"] for r in xs])),
                "secure_topk_exact": float(np.mean([r["secure_topk_exact"] for r in xs])),
            }
    with (out_dir / "sql_policy_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = {"PostFilter-SQL": "Post-SQL", "PACER-A-SQL": "PACER-A", "PACER-X-SQL": "PACER-X"}
    with (fig_dir / "table_sql_policy.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Rec. & Exact & Cert. & Cand.\\\\\n")
        f.write("\\midrule\n")
        for m in ["PostFilter-SQL", "PACER-A-SQL", "PACER-X-SQL"]:
            v = summary["methods"][m]
            f.write(f"{labels[m]} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['certified_fraction']:.2f} & {v['candidates']:.0f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run SQL-fragment visibility contract workload")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--queries", type=int, default=128)
    ap.add_argument("--budget", type=int, default=4800)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "sql_policy"
    summary = run_sql_contract(out, n=args.n, nqueries=args.queries, budget=args.budget)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
