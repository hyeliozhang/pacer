#!/usr/bin/env python3
"""Check that candidate generation is not an artifact of physical row-id order.

The test shuffles the stored row order inside every IVF list after the index is
built. Methods that rank raw candidates within a list should return identical
outputs under the same query and budget. A mismatch would indicate that the
reported recall is partly driven by arbitrary storage order rather than vector
query processing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from trustvql import (
    IVFIndex,
    caps_search, pacer_sliced_search,
    generate_dataset,
    generate_queries,
    post_filter_ivf,
    exact_ordered,
    exact_secure,
)


def clone_with_shuffled_lists(index: IVFIndex, seed: int) -> IVFIndex:
    rng = np.random.default_rng(seed)
    # A shallow object copy is enough: all summaries, assignments, centroids,
    # and radii stay fixed; only the physical row order inside each posting list
    # is shuffled.
    new = IVFIndex.__new__(IVFIndex)
    new.__dict__.update(index.__dict__.copy())
    new.cluster_ids = [rng.permutation(ids).astype(np.int64) for ids in index.cluster_ids]
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--per-regime", type=int, default=20)
    ap.add_argument("--budget", type=int, default=1600)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "order_robustness"; out.mkdir(parents=True, exist_ok=True)
    ds = generate_dataset(n=args.n, dim=64, true_clusters=64, seed=771, delete_rate=0.12, tenant_cluster_correlation=0.82)
    index = IVFIndex(ds, nlist=64, seed=772)
    shuffled = clone_with_shuffled_lists(index, seed=773)
    queries = generate_queries(ds, per_regime=args.per_regime, seed=774, correlation="positive")
    methods = {
        "PostFilter-IVF": lambda idx, p: post_filter_ivf(ds, idx, p, k=10, nprobe=8, candidate_budget=args.budget),
        "PACER-A": lambda idx, p: pacer_sliced_search(ds, idx, p, k=10, candidate_budget=args.budget, use_bounds=True, force_certify=False, bound_mode="cone"),
        "PACER-X": lambda idx, p: pacer_sliced_search(ds, idx, p, k=10, candidate_budget=ds.n, use_bounds=True, force_certify=True, bound_mode="cone"),
    }
    rows = []
    for qi, p in enumerate(queries):
        gold = exact_secure(ds, p, k=10)
        for name, fn in methods.items():
            r1 = fn(index, p)
            r2 = fn(shuffled, p)
            same = [int(x) for x in r1["ids"]] == [int(x) for x in r2["ids"]]
            rows.append({
                "query_id": qi,
                "method": name,
                "same_ids_after_list_shuffle": int(same),
                "ordered_exact_original": int(exact_ordered(gold["ids"], r1["ids"], k=10)),
                "ordered_exact_shuffled": int(exact_ordered(gold["ids"], r2["ids"], k=10)),
                "raw_candidates_original": int(r1.get("raw_candidates", 0)),
                "raw_candidates_shuffled": int(r2.get("raw_candidates", 0)),
            })
    with (out / "order_robustness.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "queries": len(queries),
        "rows": len(rows),
        "claim_checks": {
            "row_order_mismatches": int(sum(1 for r in rows if int(r["same_ids_after_list_shuffle"]) == 0)),
            "pacer_x_exact_failures": int(sum(1 for r in rows if r["method"] == "PACER-X" and int(r["ordered_exact_original"]) == 0)),
        },
    }
    with (out / "order_robustness_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
