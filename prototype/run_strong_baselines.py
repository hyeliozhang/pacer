#!/usr/bin/env python3
"""Transparent frontier filtered-vector baselines for PACER.

This script implements CPU-only, dependency-free counterparts for the main
algorithmic families used in filtered vector search:

* global graph search followed by filtering,
* inline/pre-filtered graph traversal,
* ACORN-style predicate-aware multi-hop expansion,
* Filtered-DiskANN-style beam traversal with post-verification,
* SeRF-style range-first filtering for ordered scalar predicates,
* SIEVE-style collection/row-slice index selection,
* PACER checked/certifying modes.

These are not vendor or production reimplementations. They are transparent
controls over the same vectors, metadata, deletion epochs, verifier, and cost
counters. The paper names them as algorithm-family controls, not as measured
claims about external systems.
"""
from __future__ import annotations

import csv, heapq, json, math, time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np

from trustvql import (
    PolicyVectorDataset, QueryPolicy, IVFIndex, generate_dataset, generate_queries,
    exact_secure, post_filter_ivf, prefilter_ivf, bitmap_slice_exact, pacer_sliced_search,
    contract_planner, recall_at_k, exact_ordered, failure_mode
)


def _topk(ids: np.ndarray, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64); scores = np.asarray(scores, dtype=np.float32)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    kk = min(k, ids.size)
    if ids.size > kk:
        part = np.argpartition(-scores, kk - 1)[:kk]
        order = part[np.lexsort((ids[part], -scores[part]))]
    else:
        order = np.lexsort((ids, -scores))
    return ids[order].astype(np.int64), scores[order].astype(np.float32)


class FrontierGraphIndex:
    """Cluster-local approximate kNN graph with deterministic bridge edges.

    The graph is strong enough to test graph-family access paths but still
    auditable: each coarse list is connected by vector-nearest local edges, and
    representative rows bridge nearby coarse lists. Construction uses only NumPy.
    """
    def __init__(self, ds: PolicyVectorDataset, ivf: IVFIndex, degree: int = 16, bridges: int = 4):
        self.ds = ds; self.ivf = ivf; self.degree = int(degree); self.bridges = int(bridges)
        self.neigh: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(ds.n)]
        self.build_ms = 0.0; self.edge_count = 0
        self._build()

    def _build(self) -> None:
        t0 = time.perf_counter()
        ds, ivf, deg = self.ds, self.ivf, self.degree
        neigh_sets = [set() for _ in range(ds.n)]
        for ids in ivf.cluster_ids:
            ids = np.asarray(ids, dtype=np.int64)
            if ids.size <= 1:
                continue
            X = ds.vectors[ids]
            for start in range(0, ids.size, 192):
                end = min(ids.size, start + 192)
                sims = X[start:end] @ X.T
                for i in range(end - start):
                    rid = int(ids[start + i])
                    row = sims[i]
                    row[start + i] = -np.inf
                    kk = min(deg, ids.size - 1)
                    if kk <= 0:
                        continue
                    part = np.argpartition(-row, kk - 1)[:kk]
                    for nb in ids[part]:
                        neigh_sets[rid].add(int(nb)); neigh_sets[int(nb)].add(rid)
        cent_sims = ivf.centroids @ ivf.centroids.T
        for c in range(ivf.nlist):
            cent_sims[c, c] = -np.inf
        for c in range(ivf.nlist):
            ids = ivf.cluster_ids[c]
            if not ids.size:
                continue
            reps = ids[np.argsort(-(ds.vectors[ids] @ ivf.centroids[c]))[:min(10, ids.size)]]
            nbcs = np.argsort(-cent_sims[c])[:min(self.bridges, ivf.nlist - 1)]
            for nc in nbcs:
                nids = ivf.cluster_ids[int(nc)]
                if not nids.size:
                    continue
                nreps = nids[np.argsort(-(ds.vectors[nids] @ ivf.centroids[int(nc)]))[:min(10, nids.size)]]
                for a in reps:
                    ss = ds.vectors[nreps] @ ds.vectors[int(a)]
                    for b in nreps[np.argsort(-ss)[:min(3, nreps.size)]]:
                        neigh_sets[int(a)].add(int(b)); neigh_sets[int(b)].add(int(a))
        self.neigh = [np.asarray(sorted(s), dtype=np.int64) for s in neigh_sets]
        self.edge_count = int(sum(len(x) for x in self.neigh))
        self.build_ms = (time.perf_counter() - t0) * 1000.0

    def entry_points(self, q: np.ndarray, ncentroids: int = 10, per_list: int = 5, restrict_bits: int | None = None) -> np.ndarray:
        order = np.argsort(-(self.ivf.centroids @ q))[:min(ncentroids, self.ivf.nlist)]
        entries: List[int] = []
        for c in order:
            ids = self.ivf.cluster_ids[int(c)]
            if not ids.size:
                continue
            if restrict_bits is not None:
                ids = np.asarray([int(rid) for rid in ids if (int(restrict_bits) >> int(rid)) & 1], dtype=np.int64)
                if not ids.size:
                    continue
            scores = self.ds.vectors[ids] @ q
            local = ids[np.argsort(-scores)[:min(per_list, ids.size)]]
            entries.extend(int(x) for x in local)
        if not entries:
            ids = self.ivf.cluster_ids[int(order[0])] if len(order) else np.arange(min(8, self.ds.n))
            entries.extend(int(x) for x in ids[:min(8, ids.size)])
        return np.asarray(sorted(set(entries)), dtype=np.int64)


def _verified_topk(ds: PolicyVectorDataset, ids: np.ndarray, p: QueryPolicy, k: int) -> Tuple[np.ndarray, np.ndarray, int]:
    ids = np.asarray(sorted(set(int(x) for x in ids)), dtype=np.int64)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0
    legal = ids[ds.policy_mask_ids(ids, p, include_deletions=False)]
    scores = ds.vectors[legal] @ p.qvec if legal.size else np.empty(0, dtype=np.float32)
    out_ids, out_scores = _topk(legal, scores, k)
    return out_ids, out_scores, int(legal.size)


def graph_family_search(ds: PolicyVectorDataset, g: FrontierGraphIndex, p: QueryPolicy, k: int, budget: int, mode: str) -> Dict:
    t0 = time.perf_counter()
    qbits = g.ivf.query_policy_bitset(p)
    raw_checks = 0; distance_evals = 0; policy_checks = 0; summary_checks = 0
    visited_clusters: set[int] = set()

    if mode in {"SIEVE-Collection", "SeRF-Range"}:
        upper = g.ivf.cluster_upper_bounds(p.qvec, mode="cone")
        top_ids = np.empty(0, dtype=np.int64); top_scores = np.empty(0, dtype=np.float32)
        # SIEVE: rank row-slice collections by cluster upper bound; SeRF: first
        # constrains by the ordered sensitivity range through the same bitset.
        for c in np.argsort(-upper):
            ids, postings = g.ivf.slice_candidate_ids(int(c), p, query_bits=qbits)
            summary_checks += 1
            if ids.size == 0:
                continue
            visited_clusters.add(int(c))
            room = max(0, int(budget) - raw_checks)
            if room <= 0:
                break
            sc = ds.vectors[ids] @ p.qvec
            local = np.lexsort((ids, -sc))[:min(room, ids.size)]
            ids = ids[local]
            raw_checks += int(ids.size); distance_evals += int(ids.size); policy_checks += int(ids.size)
            legal = ids[ds.policy_mask_ids(ids, p, include_deletions=False)]
            if legal.size:
                scores = ds.vectors[legal] @ p.qvec
                mid = np.concatenate([top_ids, legal]) if top_ids.size else legal
                msc = np.concatenate([top_scores, scores]) if top_scores.size else scores
                top_ids, top_scores = _topk(mid, msc, k)
            if raw_checks >= int(budget):
                break
        return {"ids": top_ids, "scores": top_scores, "latency_ms": (time.perf_counter()-t0)*1000, "candidates": int(len(top_ids)), "raw_candidates": raw_checks, "distance_evals": distance_evals, "policy_checks": policy_checks, "summary_checks": summary_checks, "visited_cluster_count": len(visited_clusters), "certified": False, "visited_clusters": sorted(visited_clusters), "graph_build_ms": g.build_ms, "graph_edges": g.edge_count}

    restrict = qbits if mode in {"InlineGraph", "ACORN-2Hop"} else None
    entries = g.entry_points(p.qvec, restrict_bits=restrict if mode in {"PredicateGraph", "ACORN-2Hop"} else None)
    seen: set[int] = set(int(x) for x in entries)
    popped: List[int] = []
    frontier: List[Tuple[float, int]] = []
    for rid in entries:
        distance_evals += 1
        heapq.heappush(frontier, (-float(ds.vectors[int(rid)] @ p.qvec), int(rid)))
    beam_width = 256 if mode == "FDANN-Beam" else 10**9
    while frontier and raw_checks < int(budget):
        neg, rid = heapq.heappop(frontier)
        raw_checks += 1; popped.append(rid)
        visited_clusters.add(int(g.ivf.assignments[rid]))
        r_ok = bool((int(qbits) >> int(rid)) & 1)
        nbs = g.neigh[rid]
        if mode == "ACORN-2Hop" and r_ok:
            # ACORN-style bridge expansion: matching nodes expose their two-hop
            # neighborhood, which helps traverse filtered predicate subgraphs.
            extra: List[int] = []
            for nb in nbs[:min(len(nbs), 24)]:
                extra.extend(int(x) for x in g.neigh[int(nb)][:4])
            if extra:
                nbs = np.asarray(sorted(set([int(x) for x in nbs] + extra)), dtype=np.int64)
        for nb0 in nbs:
            nb = int(nb0)
            if nb in seen:
                continue
            if mode == "InlineGraph" and not ((int(qbits) >> nb) & 1):
                continue
            if mode == "PredicateGraph" and raw_checks > budget // 3 and not ((int(qbits) >> nb) & 1):
                # Predicate-aware traversal becomes stricter after it has found
                # a local graph neighborhood.
                continue
            seen.add(nb); distance_evals += 1
            heapq.heappush(frontier, (-float(ds.vectors[nb] @ p.qvec), nb))
        if mode == "FDANN-Beam" and len(frontier) > beam_width:
            frontier = heapq.nsmallest(beam_width, frontier)
            heapq.heapify(frontier)
    raw = np.asarray(popped, dtype=np.int64)
    policy_checks = int(raw.size)
    top_ids, top_scores, legal_count = _verified_topk(ds, raw, p, k)
    return {"ids": top_ids, "scores": top_scores, "latency_ms": (time.perf_counter()-t0)*1000, "candidates": legal_count, "raw_candidates": int(raw.size), "distance_evals": int(distance_evals), "policy_checks": policy_checks, "summary_checks": len(visited_clusters), "visited_cluster_count": len(visited_clusters), "certified": False, "visited_clusters": sorted(visited_clusters), "graph_build_ms": g.build_ms, "graph_edges": g.edge_count}


def eval_result(ds, p, gold, res, method, qi, k):
    pv, dv, tv = ds.violations(res["ids"], p)
    return {"query_id": qi, "method": method, "regime": p.regime,
            "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=k),
            "secure_topk_exact": int(exact_ordered(gold["ids"], res["ids"], k=k)),
            "failure_mode": failure_mode(gold["ids"], res["ids"], k=k),
            "policy_violations": pv, "deleted_violations": dv, "tenant_violations": tv,
            "latency_ms": float(res["latency_ms"]), "raw_candidates": int(res.get("raw_candidates",0)),
            "distance_evals": int(res.get("distance_evals",0)), "policy_checks": int(res.get("policy_checks",0)),
            "visited_cluster_count": int(res.get("visited_cluster_count",0)), "certified": int(bool(res.get("certified", False)))}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "strong_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[strong_baselines] building dataset/index", flush=True)
    ds = generate_dataset(n=8000, dim=64, true_clusters=48, seed=7, delete_rate=0.12, tenant_cluster_correlation=0.82)
    ivf = IVFIndex(ds, nlist=48, seed=8)
    print("[strong_baselines] building frontier graph", flush=True)
    g = FrontierGraphIndex(ds, ivf, degree=12, bridges=3)
    queries = generate_queries(ds, per_regime=10, seed=9, correlation="positive")
    k = 10; budget = 1200
    rows: List[Dict] = []
    for qi, p in enumerate(queries):
        if qi % 5 == 0:
            print(f"[strong_baselines] query {qi+1}/{len(queries)}", flush=True)
        gold = exact_secure(ds, p, k=k)
        methods = {
            "PostFilter-IVF": lambda: post_filter_ivf(ds, ivf, p, k=k, nprobe=8, candidate_budget=budget),
            "PreFilter-IVF": lambda: prefilter_ivf(ds, ivf, p, k=k, nprobe=8, candidate_budget=budget),
            "GraphPostFilter": lambda: graph_family_search(ds, g, p, k, budget, mode="GraphPostFilter"),
            "InlineGraph": lambda: graph_family_search(ds, g, p, k, budget, mode="InlineGraph"),
            "PredicateGraph": lambda: graph_family_search(ds, g, p, k, budget, mode="PredicateGraph"),
            "ACORN-2Hop": lambda: graph_family_search(ds, g, p, k, budget, mode="ACORN-2Hop"),
            "FDANN-Beam": lambda: graph_family_search(ds, g, p, k, budget, mode="FDANN-Beam"),
            "SeRF-Range": lambda: graph_family_search(ds, g, p, k, budget, mode="SeRF-Range"),
            "SIEVE-Collection": lambda: graph_family_search(ds, g, p, k, budget, mode="SIEVE-Collection"),
            "BitmapSlice-Exact": lambda: bitmap_slice_exact(ds, ivf, p, k=k),
            "PACER-A": lambda: pacer_sliced_search(ds, ivf, p, k=k, candidate_budget=1200, force_certify=False),
            "PACER-C": lambda: contract_planner(ds, ivf, p, k=k, candidate_budget=1200, service_level="certify-if-cheap"),
            "PACER-X": lambda: pacer_sliced_search(ds, ivf, p, k=k, candidate_budget=ds.n, force_certify=True),
        }
        for name, fn in methods.items():
            rows.append(eval_result(ds, p, gold, fn(), name, qi, k))
    with (out_dir / "strong_baselines.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    by = {}
    for m in sorted({r["method"] for r in rows}):
        xs = [r for r in rows if r["method"] == m]
        by[m] = {"recall_at_k": float(np.mean([r["recall_at_k"] for r in xs])),
                 "secure_topk_exact": float(np.mean([r["secure_topk_exact"] for r in xs])),
                 "latency_ms": float(np.mean([r["latency_ms"] for r in xs])),
                 "raw_candidates": float(np.mean([r["raw_candidates"] for r in xs])),
                 "distance_evals": float(np.mean([r["distance_evals"] for r in xs])),
                 "policy_checks": float(np.mean([r["policy_checks"] for r in xs])),
                 "violations_per_query": float(np.mean([r["policy_violations"]+r["deleted_violations"]+r["tenant_violations"] for r in xs])),
                 "candidate_starvation": float(np.mean([r["failure_mode"] == "candidate_starvation" for r in xs])),
                 "ranking_truncation": float(np.mean([r["failure_mode"] == "ranking_truncation" for r in xs]))}
    approx_methods = [m for m in by if m not in {"PACER-X", "BitmapSlice-Exact"}]
    summary = {"queries": len(queries), "rows": len(rows), "budget": budget, "graph_build_ms": g.build_ms, "graph_edges": g.edge_count, "methods": by,
               "claim_checks": {"safe_baseline_violations": int(sum(r["policy_violations"]+r["deleted_violations"]+r["tenant_violations"] for r in rows)),
                                "pacer_c_exact_failures": int(sum(1-r["secure_topk_exact"] for r in rows if r["method"] == "PACER-C")),
                                "pacer_x_exact_failures": int(sum(1-r["secure_topk_exact"] for r in rows if r["method"] == "PACER-X")),
                                "pacer_a_beats_all_graph_exactness": bool(by["PACER-A"]["secure_topk_exact"] >= max(by[m]["secure_topk_exact"] for m in by if "Graph" in m or m in {"ACORN-2Hop","FDANN-Beam"})),
                                "pacer_c_beats_all_approx_exactness": bool(by["PACER-C"]["secure_topk_exact"] >= max(by[m]["secure_topk_exact"] for m in approx_methods))}}
    with (out_dir / "strong_baselines_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["claim_checks"], indent=2))

if __name__ == "__main__":
    main()
