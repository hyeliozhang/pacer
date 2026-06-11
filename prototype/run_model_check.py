"""Finite-domain exhaustive model checking for PACER semantics.

This runner is intentionally small but exhaustive over the database-policy axes
that are finite in the artifact: tenant masks, region masks, dtype masks,
provenance masks, sensitivity thresholds, and snapshot epochs.  It connects the
paper proofs to executable invariants: conservative summaries must not prune a
visible row, certified PACER answers must equal exact secure top-k, and row
verification must eliminate future/stale postings.
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
    caps_search,
    pacer_sliced_search,
    exact_ordered,
    exact_secure,
    generate_dataset,
    recall_at_k,
)
from proof_replay import summary_false_negatives, verify_certificate_without_oracle


def _masks(width: int) -> List[int]:
    return list(range(1, 1 << int(width)))


def run_model_check(out_dir: Path, n: int = 14, dim: int = 5, k: int = 5, seed: int = 2301) -> Dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tenants, regions, dtypes, prov_tags = 2, 2, 2, 2
    ds = generate_dataset(
        n=n,
        dim=dim,
        true_clusters=3,
        tenants=tenants,
        regions=regions,
        dtypes=dtypes,
        prov_tags=prov_tags,
        delete_rate=0.25,
        tenant_cluster_correlation=0.55,
        seed=seed,
    )
    index = IVFIndex(ds, nlist=3, seed=seed + 1, iters=3)
    qids = list(range(min(3, ds.n)))
    epochs = list(range(0, 6))
    rows: List[Dict] = []
    total = visible_cases = 0
    summary_bad = cert_bad = pacer_x_bad = safety_bad = 0
    pacer_a_certified = pacer_a_cert_bad = 0
    # Enumerate the finite policy surface.  To keep the artifact fast, q vectors
    # are a deterministic covering set of rows, while the policy space itself is
    # exhaustive for these finite domains.
    for qid in qids:
        qvec = ds.vectors[qid]
        for tmask in _masks(tenants):
            for rmask in _masks(regions):
                for dmask in _masks(dtypes):
                    for pmask in _masks(prov_tags):
                        for sens in range(4):
                            for epoch in epochs:
                                total += 1
                                p = QueryPolicy(
                                    qvec=qvec,
                                    tenant_mask=tmask,
                                    max_sensitivity=sens,
                                    region_mask=rmask,
                                    dtype_mask=dmask,
                                    provenance_mask=pmask,
                                    epoch=epoch,
                                    regime="exhaustive",
                                    source_id=qid,
                                    policy_anchor_id=qid,
                                    correlation="model_check",
                                )
                                gold = exact_secure(ds, p, k=k)
                                if len(gold["ids"]):
                                    visible_cases += 1
                                bad = summary_false_negatives(ds, index, p)
                                if bad:
                                    summary_bad += len(bad)
                                rx = pacer_sliced_search(ds, index, p, k=k, candidate_budget=ds.n, use_bounds=True, force_certify=True)
                                ra = pacer_sliced_search(ds, index, p, k=k, candidate_budget=6, use_bounds=True, force_certify=False)
                                for name, res in [("PACER-X", rx), ("PACER-A", ra)]:
                                    viol = sum(ds.violations(res["ids"], p))
                                    if viol:
                                        safety_bad += int(viol)
                                    replay = verify_certificate_without_oracle(ds, index, p, res, k=k)
                                    exact = int(exact_ordered(gold["ids"], res["ids"], k=k))
                                    if bool(res.get("certified", False)) and not replay["proof_ok_without_oracle"]:
                                        cert_bad += 1
                                    if name == "PACER-X" and exact != 1:
                                        pacer_x_bad += 1
                                    if name == "PACER-A" and bool(res.get("certified", False)):
                                        pacer_a_certified += 1
                                        if exact != 1:
                                            pacer_a_cert_bad += 1
                                # Save only failing rows and a deterministic sample to avoid a huge CSV.
                                if bad or (not exact_ordered(gold["ids"], rx["ids"], k=k)) or (total % 25000 == 0):
                                    rows.append({
                                        "case_id": total,
                                        "qid": qid,
                                        "tenant_mask": tmask,
                                        "region_mask": rmask,
                                        "dtype_mask": dmask,
                                        "provenance_mask": pmask,
                                        "sensitivity": sens,
                                        "epoch": epoch,
                                        "visible": int(len(gold["ids"])),
                                        "summary_false_negative_clusters": int(len(bad)),
                                        "pacer_x_exact": int(exact_ordered(gold["ids"], rx["ids"], k=k)),
                                        "pacer_a_recall": recall_at_k(gold["ids"], ra["ids"], k=k),
                                        "pacer_a_certified": int(bool(ra.get("certified", False))),
                                    })
    if not rows:
        rows.append({
            "case_id": 0,
            "qid": -1,
            "tenant_mask": 0,
            "region_mask": 0,
            "dtype_mask": 0,
            "provenance_mask": 0,
            "sensitivity": -1,
            "epoch": -1,
            "visible": 0,
            "summary_false_negative_clusters": 0,
            "pacer_x_exact": 1,
            "pacer_a_recall": 1.0,
            "pacer_a_certified": 0,
        })
    with (out_dir / "model_check_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "n": int(n),
        "dim": int(dim),
        "enumerated_policy_cases": int(total),
        "cases_with_nonempty_visible_answer": int(visible_cases),
        "claim_checks": {
            "summary_false_negative_clusters": int(summary_bad),
            "certified_proof_failures_without_oracle": int(cert_bad),
            "pacer_x_exact_failures": int(pacer_x_bad),
            "safe_output_violations": int(safety_bad),
            "pacer_a_certified_cases": int(pacer_a_certified),
            "pacer_a_certified_exact_failures": int(pacer_a_cert_bad),
        },
        "domains": {"tenants": tenants, "regions": regions, "dtypes": dtypes, "provenance_tags": prov_tags, "epochs": [min(epochs), max(epochs)], "query_vectors": len(qids)},
    }
    with (out_dir / "model_check_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/model_check"))
    args = ap.parse_args()
    s = run_model_check(args.out)
    print(json.dumps(s, indent=2))
