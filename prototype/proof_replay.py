"""Independent proof-object replay checks for PACER certificates.

The paper's certificate claim is not just a metric flag.  This module replays
proof obligations from the base relation and the IVF summaries: summary pruning
must have no false negatives, every returned row must satisfy authoritative
visibility, and a certified answer must either exhaust all possible visible
clusters or dominate every unprocessed possible cluster by a cluster upper bound.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List
import csv
import json
import math

import numpy as np

from trustvql import (
    IVFIndex,
    QueryPolicy,
    caps_search,
    pacer_sliced_search,
    exact_ordered,
    exact_secure,
    generate_dataset,
    generate_queries,
)


def summary_false_negatives(ds, index: IVFIndex, p: QueryPolicy) -> List[int]:
    """Clusters that contain a visible row but are pruned by the summary."""
    bad: List[int] = []
    for c, ids in enumerate(index.cluster_ids):
        if ids.size == 0:
            continue
        contains_visible = bool(np.any(ds.policy_mask_ids(ids, p, include_deletions=False)))
        if contains_visible and not index.may_satisfy(c, p, include_deletions=False):
            bad.append(int(c))
    return bad


def _is_sorted_by_score_then_id(ds, ids: Iterable[int], p: QueryPolicy) -> bool:
    ids = list(map(int, ids))
    if len(ids) <= 1:
        return True
    scores = ds.vectors[np.asarray(ids, dtype=np.int64)] @ p.qvec
    pairs = [(-float(s), int(i)) for s, i in zip(scores, ids)]
    return pairs == sorted(pairs)


def verify_certificate_without_oracle(ds, index: IVFIndex, p: QueryPolicy, res: Dict, k: int = 10) -> Dict:
    ids = np.asarray(res.get("ids", []), dtype=np.int64)
    visible_ok = bool(np.all(ds.policy_mask_ids(ids, p, include_deletions=False))) if ids.size else True
    order_ok = _is_sorted_by_score_then_id(ds, ids, p)
    trunc_ok = int(res.get("truncated_visible_slice", 0)) == 0
    certified = bool(res.get("certified", False))
    processed = set(map(int, res.get("processed_clusters", [])))
    possible = [c for c in range(index.nlist) if index.may_satisfy(c, p, include_deletions=False)]
    unprocessed_possible = [c for c in possible if c not in processed]
    upper = index.cluster_upper_bounds(p.qvec)
    if ids.size >= min(k, max(1, ids.size)) and ids.size:
        scores = ds.vectors[ids] @ p.qvec
        kth = float(scores[min(k, len(scores)) - 1])
    else:
        kth = -math.inf
    bound_ok = True
    if certified:
        if len(ids) >= k:
            bound_ok = all(float(upper[c]) <= kth + 1e-6 for c in unprocessed_possible)
        else:
            # If fewer than k rows are returned, certification requires that no
            # unprocessed summary-compatible cluster remains.
            bound_ok = len(unprocessed_possible) == 0
    return {
        "certified": int(certified),
        "visible_ok": int(visible_ok),
        "order_ok": int(order_ok),
        "truncation_ok": int(trunc_ok),
        "unprocessed_possible_clusters": int(len(unprocessed_possible)),
        "bound_ok": int(bound_ok),
        "proof_ok_without_oracle": int((not certified) or (visible_ok and order_ok and trunc_ok and bound_ok)),
    }


def run_proof_replay(out_dir: Path, n: int = 15000, dim: int = 64, per_regime: int = 20, k: int = 10, seed: int = 7, nlist: int = 64, candidate_budget: int = 1200) -> Dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = generate_dataset(n=n, dim=dim, true_clusters=max(48, nlist), seed=seed, delete_rate=0.12, tenant_cluster_correlation=0.82)
    index = IVFIndex(ds, nlist=nlist, seed=seed + 1)
    queries = generate_queries(ds, per_regime=per_regime, seed=seed + 2, correlation="positive")
    rows: List[Dict] = []
    for qi, p in enumerate(queries):
        bad = summary_false_negatives(ds, index, p)
        gold = exact_secure(ds, p, k=k)
        for method, res in {
            "PACER-A": pacer_sliced_search(ds, index, p, k=k, candidate_budget=candidate_budget, use_bounds=True, force_certify=False),
            "PACER-X": pacer_sliced_search(ds, index, p, k=k, candidate_budget=ds.n, use_bounds=True, force_certify=True),
            "PACER-NoSummary": caps_search(ds, index, p, k=k, candidate_budget=candidate_budget, use_policy_slices=False, use_bounds=True, force_certify=False),
        }.items():
            replay = verify_certificate_without_oracle(ds, index, p, res, k=k)
            exact = int(exact_ordered(gold["ids"], res["ids"], k=k))
            rows.append({
                "query_id": qi,
                "regime": p.regime,
                "method": method,
                "summary_false_negative_clusters": int(len(bad)),
                "certified": replay["certified"],
                "proof_ok_without_oracle": replay["proof_ok_without_oracle"],
                "bound_ok": replay["bound_ok"],
                "visible_ok": replay["visible_ok"],
                "order_ok": replay["order_ok"],
                "truncation_ok": replay["truncation_ok"],
                "unprocessed_possible_clusters": replay["unprocessed_possible_clusters"],
                "oracle_exact": exact,
                "returned": int(len(res["ids"])),
                "processed_clusters": int(len(res.get("processed_clusters", []))),
                "visited_clusters": int(len(res.get("visited_clusters", []))),
            })
    metrics_path = out_dir / "proof_replay.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "n": int(n),
        "queries": int(len(queries)),
        "rows": int(len(rows)),
        "claim_checks": {
            "summary_false_negative_clusters_total": int(sum(r["summary_false_negative_clusters"] for r in rows if r["method"] == "PACER-A")),
            "certified_proof_failures_without_oracle": int(sum((r["certified"] == 1 and r["proof_ok_without_oracle"] == 0) for r in rows)),
            "certified_oracle_exact_failures": int(sum((r["certified"] == 1 and r["oracle_exact"] == 0) for r in rows)),
            "pacer_x_oracle_exact_failures": int(sum((r["method"] == "PACER-X" and r["oracle_exact"] == 0) for r in rows)),
        },
        "certified_fraction": {
            m: float(np.mean([r["certified"] for r in rows if r["method"] == m])) for m in sorted(set(r["method"] for r in rows))
        },
        "mean_processed_clusters": {
            m: float(np.mean([r["processed_clusters"] for r in rows if r["method"] == m])) for m in sorted(set(r["method"] for r in rows))
        },
    }
    with (out_dir / "proof_replay_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/proof_replay"))
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--per-regime", type=int, default=20)
    args = ap.parse_args()
    s = run_proof_replay(args.out, n=args.n, per_regime=args.per_regime)
    print(json.dumps(s, indent=2))
