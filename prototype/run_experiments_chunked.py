#!/usr/bin/env python3
"""Chunked default PACER benchmark runner.

The default workload has 320 queries x 16 methods.  Running all of it in one
long Python process stresses BLAS/thread pools on some shared CI machines.  This
script is semantically equivalent to run_experiments.py for the default paper
configuration, but executes independent query slices in fresh worker processes
and then combines the CSV/JSON outputs.  It keeps all outputs inside the
repository's results/ directory by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from trustvql import (
    IVFIndex,
    QueryPolicy,
    _method_dict,
    adaptive_post_filter_ivf,
    exact_ordered,
    exact_secure,
    failure_mode,
    generate_dataset,
    generate_queries,
    make_audit_transcript,
    naive_no_policy_ivf,
    pacer_sliced_search,
    post_filter_ivf,
    recall_at_k,
    stale_no_recheck,
    summarize_metrics,
)


def _build(args):
    ds = generate_dataset(
        n=args.n,
        dim=args.dim,
        true_clusters=max(48, args.nlist),
        seed=args.seed,
        delete_rate=args.delete_rate,
        tenant_cluster_correlation=args.tenant_cluster_correlation,
    )
    index = IVFIndex(ds, nlist=args.nlist, seed=args.seed + 1)
    queries = generate_queries(ds, per_regime=args.per_regime, seed=args.seed + 2, correlation=args.correlation)
    golds = [exact_secure(ds, p, k=args.k) for p in queries]
    return ds, index, queries, golds


def worker(args) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds, index, queries, golds = _build(args)
    methods = _method_dict(ds, index, args.k, args.budget)
    rows: List[Dict] = []
    audit_written = False
    end = min(args.worker_end, len(queries))
    for qi in range(args.worker_start, end):
        if args.progress and (qi == args.worker_start or qi % 10 == 0):
            print(f"[chunk-worker] main {qi+1}/{len(queries)}", flush=True)
        p = queries[qi]; gold = golds[qi]
        allowed = int(ds.policy_mask(p, include_deletions=False).sum())
        selectivity = float(allowed / args.n)
        for name, fn in methods.items():
            res = fn(p)
            policy_v, del_v, tenant_v = ds.violations(res["ids"], p)
            ev, et = ds.explanations_valid(res["ids"], p)
            exact = int(exact_ordered(gold["ids"], res["ids"], k=args.k))
            certified = int(bool(res.get("certified", False)))
            cert_false = int(certified == 1 and exact == 0)
            rows.append({
                "query_id": qi,
                "regime": p.regime,
                "correlation": p.correlation,
                "method": name,
                "n": args.n,
                "dim": args.dim,
                "k": args.k,
                "allowed_count": allowed,
                "selectivity": selectivity,
                "returned": int(len(res["ids"])),
                "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=args.k),
                "secure_topk_exact": exact,
                "failure_mode": failure_mode(gold["ids"], res["ids"], k=args.k),
                "policy_violations": policy_v,
                "deleted_violations": del_v,
                "tenant_violations": tenant_v,
                "explanation_valid": ev,
                "explanation_total": et,
                "explanation_accuracy": float(ev / et) if et else 1.0,
                "latency_ms": float(res["latency_ms"]),
                "raw_candidates": int(res.get("raw_candidates", res.get("candidates", 0))),
                "candidates": int(res["candidates"]),
                "distance_evals": int(res.get("distance_evals", res.get("candidates", 0))),
                "policy_checks": int(res.get("policy_checks", res.get("raw_candidates", 0))),
                "summary_checks": int(res.get("summary_checks", 0)),
                "visited_cluster_count": int(res.get("visited_cluster_count", 0)),
                "certified": certified,
                "certified_and_exact": int((not certified) or bool(exact)),
                "certificate_false_positive": cert_false,
                "certificate_gap": float(res.get("certificate_gap", 0.0)),
                "truncated_visible_slice": int(res.get("truncated_visible_slice", 0)),
            })
            if name == "PACER-X" and qi == args.worker_start and not audit_written:
                with (out / f"audit_transcript_sample_{args.worker_start}_{end}.json").open("w") as f:
                    json.dump(make_audit_transcript(ds, index, p, res, gold, args.k), f, indent=2)
                audit_written = True
    part = out / f"metrics_part_{args.worker_start}_{end}.csv"
    with part.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"status":"ok","rows":len(rows),"path":str(part)}))


def read_csv(path: Path) -> List[Dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def budget_and_deletion(args, out: Path):
    ds, index, queries, golds = _build(args)
    budget_rows: List[Dict] = []
    for budget in [150, 300, 600, 1200, 2400, 4800]:
        methods = {
            "PostFilter-IVF": lambda p, b=budget: post_filter_ivf(ds, index, p, k=args.k, nprobe=8, candidate_budget=b),
            "AdaptivePostFilter-IVF": lambda p, b=budget: adaptive_post_filter_ivf(ds, index, p, k=args.k, candidate_budget=b),
            "PACER-A": lambda p, b=budget: pacer_sliced_search(ds, index, p, k=args.k, candidate_budget=b),
        }
        for qi, (p, gold) in enumerate(zip(queries[:48], golds[:48])):
            for name, fn in methods.items():
                res = fn(p)
                policy_v, del_v, tenant_v = ds.violations(res["ids"], p)
                exact = int(exact_ordered(gold["ids"], res["ids"], k=args.k))
                budget_rows.append({
                    "query_id": qi, "regime": p.regime, "correlation": p.correlation, "method": name, "budget": int(budget),
                    "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=args.k), "secure_topk_exact": exact,
                    "latency_ms": float(res["latency_ms"]), "candidates": int(res["candidates"]),
                    "raw_candidates": int(res.get("raw_candidates", 0)), "distance_evals": int(res.get("distance_evals", 0)),
                    "policy_checks": int(res.get("policy_checks", 0)), "policy_violations": policy_v,
                    "deleted_violations": del_v, "tenant_violations": tenant_v, "certified": int(bool(res.get("certified", False))),
                })
    deletion_rows: List[Dict] = []
    for epoch in range(1, 10):
        pcount = pacer_del = certify_del = stale_del = naive_del = 0
        for p0 in queries[:32]:
            p = QueryPolicy(p0.qvec, p0.tenant_mask, p0.max_sensitivity, p0.region_mask, p0.dtype_mask, p0.provenance_mask, epoch, p0.regime, p0.source_id, p0.policy_anchor_id, p0.correlation)
            for key, res in [
                ("pacer", pacer_sliced_search(ds, index, p, k=args.k, candidate_budget=args.budget)),
                ("cert", pacer_sliced_search(ds, index, p, k=args.k, candidate_budget=ds.n, force_certify=True)),
                ("stale", stale_no_recheck(ds, index, p, k=args.k, nprobe=8, candidate_budget=args.budget)),
                ("naive", naive_no_policy_ivf(ds, index, p, k=args.k, nprobe=8, candidate_budget=args.budget)),
            ]:
                _, del_v, _ = ds.violations(res["ids"], p)
                if key == "pacer": pacer_del += del_v
                elif key == "cert": certify_del += del_v
                elif key == "stale": stale_del += del_v
                else: naive_del += del_v
            pcount += 1
        deletion_rows.append({
            "epoch": epoch, "queries": pcount, "stale_residues_in_index": int(np.sum(ds.delete_epoch <= epoch)),
            "pacer_deleted_results": int(pacer_del), "pacer_certify_deleted_results": int(certify_del),
            "stale_no_recheck_deleted_results": int(stale_del), "naive_deleted_results": int(naive_del),
        })
    write_csv(out / f"budget_n{args.n}.csv", budget_rows)
    write_csv(out / f"deletion_n{args.n}.csv", deletion_rows)
    return ds, index, budget_rows, deletion_rows


def master(args) -> None:
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    step = max(1, args.chunk_size)
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    for start in range(0, args.per_regime * 4, step):
        end = min(args.per_regime * 4, start + step)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--worker-start", str(start), "--worker-end", str(end), "--out", str(out), "--n", str(args.n), "--dim", str(args.dim), "--per-regime", str(args.per_regime), "--seed", str(args.seed), "--nlist", str(args.nlist), "--budget", str(args.budget), "--correlation", args.correlation, "--delete-rate", str(args.delete_rate), "--tenant-cluster-correlation", str(args.tenant_cluster_correlation)]
        if args.progress:
            cmd.append("--progress")
        print(f"[chunk-master] worker {start}:{end}", flush=True)
        subprocess.run(cmd, check=True, env=env)
    rows: List[Dict] = []
    for part in sorted(out.glob("metrics_part_*.csv"), key=lambda p: int(p.stem.split("_")[2])):
        rows.extend(read_csv(part))
    write_csv(out / f"metrics_n{args.n}.csv", rows)
    failure_rows: List[Dict] = []
    for method in sorted(set(r["method"] for r in rows)):
        xs = [r for r in rows if r["method"] == method]
        for fm in ["none", "candidate_starvation", "ranking_truncation"]:
            failure_rows.append({"method": method, "failure_mode": fm, "fraction": float(np.mean([r["failure_mode"] == fm for r in xs]))})
    write_csv(out / f"failure_modes_n{args.n}.csv", failure_rows)
    ds, index, budget_rows, deletion_rows = budget_and_deletion(args, out)
    # csv reader returns strings; summarize_metrics expects numeric coercible values.
    summary = summarize_metrics(rows, budget_rows, deletion_rows, index.index_overhead())
    summary.update({
        "dataset": ds.metadata, "n": int(args.n), "dim": int(args.dim), "queries": int(args.per_regime * 4),
        "query_method_rows": int(len(rows)), "nlist": int(index.nlist), "k": int(args.k), "correlation": args.correlation,
        "candidate_budget": int(args.budget),
        "budget_semantics": "raw row checks for bounded ANN methods; PACER-A uses exact non-epoch bitmap slice candidates",
        "runner": "chunked fresh-process default benchmark",
    })
    with (out / f"summary_n{args.n}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    sample = out / f"audit_transcript_sample_0_{min(step,args.per_regime*4)}.json"
    if sample.exists():
        (out / "audit_transcript_sample.json").write_text(sample.read_text())
    print(json.dumps({"status":"ok","queries":summary["queries"],"query_method_rows":summary["query_method_rows"],"methods":len(summary["overall"])}))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunked CPU-only PACER paper benchmark")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-start", type=int, default=0)
    ap.add_argument("--worker-end", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--per-regime", type=int, default=80)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nlist", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--correlation", choices=["positive", "independent", "negative"], default="positive")
    ap.add_argument("--delete-rate", type=float, default=0.12)
    ap.add_argument("--tenant-cluster-correlation", type=float, default=0.82)
    ap.add_argument("--chunk-size", type=int, default=80)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if args.worker:
        worker(args)
    else:
        master(args)


if __name__ == "__main__":
    main()
