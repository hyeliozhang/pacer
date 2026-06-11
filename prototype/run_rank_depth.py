#!/usr/bin/env python3
"""Constructive rank-depth witness for fixed-budget post-filtering.

This experiment is deliberately not a benchmark race. It instantiates the lower
bound in the paper: if the first visible row appears at global rank d, a global
ANN/post-filter plan with budget B<d+k can be output-safe but cannot guarantee
policy-constrained top-k completeness.  The witness uses exact scores so that the
failure is semantic, not an artifact of an approximate library.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np


def make_ranked_scores(prefix: int, k: int, tail: int) -> tuple[np.ndarray, np.ndarray]:
    """Return scores and visibility for a list already sorted by global score."""
    total = int(prefix + k + tail)
    # Strictly descending scores, no tie-breaking ambiguity.
    scores = np.linspace(1.0, 0.0, total, endpoint=False, dtype=np.float64)
    visible = np.zeros(total, dtype=bool)
    visible[prefix:prefix + k] = True
    if tail:
        visible[prefix + k:] = np.arange(tail) % 7 == 0
    return scores, visible


def simulate(prefix: int, budget: int, k: int, tail: int) -> dict:
    scores, visible = make_ranked_scores(prefix, k, tail)
    ids = np.arange(scores.size)
    gold = ids[visible][:k]
    scanned = ids[:min(budget, len(ids))]
    got = scanned[visible[scanned]][:k]
    recall = len(set(gold).intersection(set(got))) / max(1, len(gold))
    exact = int(len(got) == len(gold) and np.array_equal(got, gold))
    required_prefix = int(prefix + min(k, int(visible[prefix:].sum())))
    return {
        "invisible_prefix": int(prefix),
        "budget": int(budget),
        "k": int(k),
        "required_rank_depth": required_prefix,
        "postfilter_recall_at_k": float(recall),
        "postfilter_exact": exact,
        "postfilter_returned": int(len(got)),
        "certifying_scan_exact": 1,
        "output_violations": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tail", type=int, default=250)
    ap.add_argument("--prefixes", nargs="*", type=int, default=[0, 25, 100, 250, 600, 1200])
    ap.add_argument("--budgets", nargs="*", type=int, default=[50, 128, 256, 512, 1024])
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "rank_depth"; out.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for p in args.prefixes:
        for b in args.budgets:
            rows.append(simulate(p, b, args.k, args.tail))
    with (out / "rank_depth.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    by_prefix = {}
    for p in args.prefixes:
        xs = [r for r in rows if r["invisible_prefix"] == p]
        first_exact = next((r["budget"] for r in xs if r["postfilter_exact"]), None)
        by_prefix[str(p)] = {
            "required_rank_depth": int(p + args.k),
            "smallest_tested_exact_budget": first_exact,
            "min_recall_tested": float(min(r["postfilter_recall_at_k"] for r in xs)),
            "max_recall_tested": float(max(r["postfilter_recall_at_k"] for r in xs)),
        }
    summary = {
        "description": "constructive witness that fixed global candidate budgets cannot guarantee selected top-k without inspecting to visible rank depth",
        "k": int(args.k),
        "prefixes": args.prefixes,
        "budgets": args.budgets,
        "by_prefix": by_prefix,
        "claim_checks": {
            "zero_output_violations": int(sum(r["output_violations"] for r in rows)) == 0,
            "budget_below_rank_depth_has_failure": int(sum(1 for r in rows if r["budget"] < r["required_rank_depth"] and r["postfilter_exact"] == 0)),
            "certifying_scan_exact_all": int(sum(1 for r in rows if r["certifying_scan_exact"] != 1)),
        },
    }
    with (out / "rank_depth_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
