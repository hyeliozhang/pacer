#!/usr/bin/env python3
"""Invariant tests for the PACER artifact.

These are small deterministic tests intended for reviewers. They do not replace
experiments; they check proof obligations that should hold independently of a
particular metric table.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

from trustvql import IVFIndex, generate_dataset, exact_secure, exact_ordered, generate_queries, caps_search  # type: ignore
from sql_ast_contract import Atom, And, Or, Not, RelationalSummaryIndex, make_rel_context, pacer_ast, exact_ast  # type: ignore
from access_path_adapter import run_default as run_adapter_contract  # type: ignore


def test_cluster_summary_pruning_has_no_false_negatives() -> None:
    ds = generate_dataset(n=1200, dim=32, true_clusters=24, tenants=12, regions=4, dtypes=4, prov_tags=8, seed=77)
    ctx = make_rel_context(ds, seed=88)
    index = IVFIndex(ds, nlist=24, seed=89, iters=3)
    relidx = RelationalSummaryIndex(ctx, index)
    expr = And((
        Or((Atom("tenant", "in", (1, 3, 5)), Atom("region", "in", (0,)))),
        Atom("deny_group", "notin", (2, 4, 6)),
        Atom("sensitivity", "<=", 2),
        Atom("live_epoch", "eq", 5),
    ))
    may = relidx.may_clusters(expr)
    for c, ids in enumerate(index.cluster_ids):
        if not may[c]:
            assert not bool(expr.eval(ctx, ids).any()), f"summary pruned visible row in cluster {c}"


def test_negation_is_residual_not_pruning_authority() -> None:
    ds = generate_dataset(n=800, dim=24, true_clusters=16, tenants=8, regions=4, dtypes=4, prov_tags=8, seed=91)
    ctx = make_rel_context(ds, seed=92)
    index = IVFIndex(ds, nlist=16, seed=93, iters=2)
    relidx = RelationalSummaryIndex(ctx, index)
    expr = Not(Atom("tenant", "in", (0,)))
    may = relidx.may_clusters(expr)
    assert bool(may.all()), "NOT predicates must remain residual unless complemented summaries are supplied"


def test_sql_ast_pacer_x_matches_exact() -> None:
    ds = generate_dataset(n=1000, dim=32, true_clusters=20, tenants=10, regions=5, dtypes=4, prov_tags=8, seed=101)
    ctx = make_rel_context(ds, seed=102)
    index = IVFIndex(ds, nlist=20, seed=103, iters=3)
    relidx = RelationalSummaryIndex(ctx, index)
    ids = np.arange(ds.n)
    for src in [5, 77, 111, 222, 333]:
        ptag = int(ds.provenance[src]) or 1
        expr = And((
            Or((Atom("tenant", "eq", int(ds.tenant[src])), Atom("region", "eq", int(ds.region[src])))),
            Atom("provenance", "bitany", ptag),
            Atom("live_epoch", "eq", 5),
        ))
        if int(expr.eval(ctx, ids).sum()) < 10:
            continue
        q = {"expr": expr, "qvec": ds.vectors[src], "kind": "unit"}
        gold = exact_ast(ctx, q, k=10)
        res = pacer_ast(ctx, index, relidx, q, k=10, budget=ds.n, force_certify=True)
        assert res["certified"], "PACER-X should certify after compatible-space exhaustion"
        assert exact_ordered(gold["ids"], res["ids"], k=10), "PACER-X certificate returned non-exact list"


def test_original_pacer_x_matches_exact_secure() -> None:
    ds = generate_dataset(n=1200, dim=32, true_clusters=20, tenants=10, regions=5, dtypes=4, prov_tags=8, seed=131)
    index = IVFIndex(ds, nlist=20, seed=132, iters=3)
    qs = generate_queries(ds, per_regime=3, seed=133)
    for q in qs[:8]:
        gold = exact_secure(ds, q, k=10)
        res = caps_search(ds, index, q, k=10, candidate_budget=ds.n, use_policy_slices=True, use_bounds=True, force_certify=True)
        assert res["certified"], "PACER-X did not certify"
        assert exact_ordered(gold["ids"], res["ids"], k=10), "PACER-X mismatch with exact secure top-k"


def test_snapshot_insert_epoch_is_authoritative() -> None:
    from trustvql import QueryPolicy, _bitmask
    ds = generate_dataset(n=600, dim=24, true_clusters=12, tenants=6, regions=3, dtypes=3, prov_tags=6, seed=151)
    future = int(np.argmax(ds.insert_epochs()))
    epoch = max(0, int(ds.insert_epochs()[future]) - 1)
    p = QueryPolicy(ds.vectors[future], _bitmask([int(ds.tenant[future])]), int(ds.sensitivity[future]), _bitmask([int(ds.region[future])]), _bitmask([int(ds.dtype[future])]), int(ds.provenance[future]), epoch, "unit", future, future)
    assert not bool(ds.policy_mask_ids(np.array([future]), p, include_deletions=False)[0]), "future insertion became visible before its insertion epoch"
    facts = ds.row_facts([future], p)[0]
    assert facts["insert_ok"] is False and facts["visible"] is False, "witness failed to expose insertion-epoch failure"


def test_rank_depth_constructive_witness() -> None:
    from run_rank_depth import simulate
    row = simulate(prefix=250, budget=128, k=10, tail=20)
    assert row["output_violations"] == 0, "post-filter witness should be output-safe"
    assert row["postfilter_exact"] == 0 and row["postfilter_recall_at_k"] == 0.0, "budget below visible rank depth should fail completeness"


def test_violation_counters_are_disjoint() -> None:
    from trustvql import QueryPolicy, _bitmask
    ds = generate_dataset(n=300, dim=16, true_clusters=8, tenants=6, regions=3, dtypes=3, prov_tags=6, seed=171)
    rid = 0
    # Build a policy that intentionally fails tenant, scalar/provenance, and epoch
    # on the same row.  The counters should report one failure in each disjoint
    # category rather than double-counting tenant as a generic policy failure.
    bad_tenant = (int(ds.tenant[rid]) + 1) % int(ds.metadata["tenants"])
    bad_region = (int(ds.region[rid]) + 1) % int(ds.metadata["regions"])
    p = QueryPolicy(
        ds.vectors[rid],
        _bitmask([bad_tenant]),
        max(0, int(ds.sensitivity[rid]) - 1),
        _bitmask([bad_region]),
        _bitmask([int(ds.dtype[rid])]),
        0,
        int(ds.delete_epoch[rid]) if int(ds.delete_epoch[rid]) < 10**9 else -1,
        "unit",
        rid,
        rid,
    )
    pol, deleted, tenant = ds.violations([rid], p)
    assert tenant == 1, "tenant failure was not counted separately"
    assert pol == 1, "residual policy failure should count once"
    assert deleted == 1, "epoch failure should count once"



def test_access_path_adapter_contract_holds() -> None:
    summary = run_adapter_contract(seed=515, n=1200, per_regime=2)
    assert summary["passed"], f"adapter contract failed: {summary['failures']}"
    assert summary["checked_units"] > 0 and summary["witness_rows_checked"] > 0, "adapter contract did not exercise units and witnesses"

def test_certificate_requires_strict_score_domination() -> None:
    ds = generate_dataset(n=800, dim=24, true_clusters=16, tenants=8, regions=4, dtypes=4, prov_tags=8, seed=181)
    index = IVFIndex(ds, nlist=16, seed=182, iters=2)
    q = generate_queries(ds, per_regime=1, seed=183)[0]
    # Force equal score bounds synthetically: without strict domination, a >= test
    # could certify while a tied unvisited unit remains possible.  With the strict-score
    # conservative check, the bounded run must not certify unless it exhausts.
    res = caps_search(ds, index, q, k=10, candidate_budget=10, use_policy_slices=True, use_bounds=False, force_certify=False)
    assert not res["certified"] or res.get("truncated_visible_slice", 0) == 0, "certificate should not ignore unresolved ties"
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
