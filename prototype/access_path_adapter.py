#!/usr/bin/env python3
"""Executable PACER access-path adapter contract.

This module is deliberately small.  It does not integrate a particular vendor
engine; instead it defines the four callbacks that any engine-side PACER access
path must expose and provides a conformance test over the transparent IVF
adapter used in the artifact:

  1. summary_possible(policy) for units that may contain visible rows;
  2. policy_relevant_row_slice(unit, policy) for row identifiers to verify;
  3. score_upper_bound(unit, qvec) for certification; and
  4. row_facts(ids, policy) for replayable visibility witnesses.

The checker recomputes the concrete row facts and fails if the adapter would
prune a visible row, omit a row from the exact non-epoch policy slice, advertise
an unsound score upper bound, or return incomplete witnesses.  It is an artifact
boundary for production integrations: graph blocks, vector pages, posting lists,
or relational partitions can replace the IVF unit implementation if they pass
the same tests.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from trustvql import IVFIndex, QueryPolicy, generate_dataset, generate_queries


_REQUIRED_WITNESS_KEYS = {
    "id", "tenant", "sensitivity", "region", "dtype", "provenance",
    "insert_epoch", "delete_epoch", "epoch", "visible", "tenant_ok",
    "region_ok", "dtype_ok", "provenance_ok", "sensitivity_ok",
    "insert_ok", "live_ok",
}


@dataclass
class IVFAdapter:
    """Reference adapter exposing the PACER callbacks over ``IVFIndex`` units."""

    index: IVFIndex

    def summary_possible(self, policy: QueryPolicy) -> List[int]:
        return [int(c) for c in range(self.index.nlist) if self.index.may_satisfy(c, policy, include_deletions=False)]

    def policy_relevant_row_slice(self, unit: int, policy: QueryPolicy) -> np.ndarray:
        ids, _ = self.index.slice_candidate_ids(int(unit), policy)
        return ids.astype(np.int64, copy=False)

    def score_upper_bound(self, unit: int, qvec: np.ndarray) -> float:
        return float(self.index.cluster_upper_bounds(qvec)[int(unit)])

    def row_facts(self, ids: Iterable[int], policy: QueryPolicy) -> List[Dict[str, Any]]:
        return self.index.ds.row_facts(ids, policy)


def _non_epoch_policy_ids(ds, ids: np.ndarray, p: QueryPolicy) -> np.ndarray:
    return ids[ds.policy_mask_ids(ids, p, include_deletions=True)] if ids.size else np.empty(0, dtype=np.int64)


def validate_adapter(adapter: IVFAdapter, policies: List[QueryPolicy], *, eps: float = 1e-5) -> Dict[str, Any]:
    ds = adapter.index.ds
    checked_units = 0
    pruned_units = 0
    visible_rows_in_pruned_units = 0
    slice_omissions = 0
    bound_violations = 0
    witness_failures = 0
    witness_rows = 0

    for p in policies:
        possible = set(adapter.summary_possible(p))
        all_units = set(range(adapter.index.nlist))
        for c in sorted(all_units - possible):
            ids = adapter.index.cluster_ids[c]
            if ids.size:
                visible_rows_in_pruned_units += int(np.sum(ds.policy_mask_ids(ids, p, include_deletions=False)))
            pruned_units += 1
        for c in sorted(possible):
            checked_units += 1
            cluster_ids = adapter.index.cluster_ids[c]
            expected = set(int(x) for x in _non_epoch_policy_ids(ds, cluster_ids, p))
            got = set(int(x) for x in adapter.policy_relevant_row_slice(c, p))
            slice_omissions += len(expected - got)
            if cluster_ids.size:
                ub = adapter.score_upper_bound(c, p.qvec)
                max_score = float(np.max(ds.vectors[cluster_ids] @ p.qvec))
                if ub + eps < max_score:
                    bound_violations += 1
            sample = list(sorted(got))[:5]
            facts = adapter.row_facts(sample, p)
            witness_rows += len(facts)
            for fact in facts:
                if not _REQUIRED_WITNESS_KEYS.issubset(fact.keys()):
                    witness_failures += 1
                    continue
                rid = int(fact["id"])
                expected_visible = bool(ds.policy_mask_ids(np.asarray([rid], dtype=np.int64), p, include_deletions=False)[0])
                if bool(fact["visible"]) != expected_visible:
                    witness_failures += 1

    failures = {
        "visible_rows_in_pruned_units": int(visible_rows_in_pruned_units),
        "slice_omissions": int(slice_omissions),
        "bound_violations": int(bound_violations),
        "witness_failures": int(witness_failures),
    }
    return {
        "schema": "pacer-access-path-adapter-contract-v1",
        "policies": int(len(policies)),
        "checked_units": int(checked_units),
        "pruned_units": int(pruned_units),
        "witness_rows_checked": int(witness_rows),
        "failures": failures,
        "passed": all(v == 0 for v in failures.values()),
    }


def run_default(seed: int = 404, n: int = 3000, per_regime: int = 3) -> Dict[str, Any]:
    ds = generate_dataset(n=n, dim=48, true_clusters=32, tenants=32, regions=6, dtypes=6, prov_tags=12,
                          delete_rate=0.17, tenant_cluster_correlation=0.73, seed=seed)
    index = IVFIndex(ds, nlist=32, seed=seed + 1, iters=2)
    policies = generate_queries(ds, per_regime=per_regime, seed=seed + 2, correlation="positive")
    result = validate_adapter(IVFAdapter(index), policies)
    result.update({"n": int(n), "nlist": 32, "seed": int(seed), "method_query_policies": int(len(policies))})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the executable PACER access-path adapter contract")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--per-regime", type=int, default=3)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "results" / "adapter_contract_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run_default(seed=args.seed, n=args.n, per_regime=args.per_regime)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
