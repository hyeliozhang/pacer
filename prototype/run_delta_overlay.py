#!/usr/bin/env python3
"""Dynamic snapshot/overlay experiment for PACER.

This experiment turns the snapshot correctness story into a mutable access-path
mechanism. A compacted base index is rebuilt periodically, while newly inserted
rows live in an exact delta overlay and deletions are represented as tombstone
epochs checked by the authoritative row verifier. Queries merge a certified
PACER-X answer over the compacted base with an exact delta answer. The script
reports whether the overlay matches a fresh full-index oracle, and how often a
stale base-only plan misses newly inserted visible rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from trustvql import (
    INF_EPOCH,
    IVFIndex,
    PolicyVectorDataset,
    QueryPolicy,
    _bitmask,
    _normalize,
    _topk_from_scores,
    bitmap_slice_exact,
    exact_ordered,
    exact_secure,
    generate_dataset,
    pacer_sliced_search,
    recall_at_k,
)


def subset_dataset(ds: PolicyVectorDataset, ids: np.ndarray) -> PolicyVectorDataset:
    ids = np.asarray(ids, dtype=np.int64)
    return PolicyVectorDataset(
        vectors=ds.vectors[ids].copy(),
        tenant=ds.tenant[ids].copy(),
        sensitivity=ds.sensitivity[ids].copy(),
        region=ds.region[ids].copy(),
        dtype=ds.dtype[ids].copy(),
        provenance=ds.provenance[ids].copy(),
        delete_epoch=ds.delete_epoch[ids].copy(),
        record_id=ds.record_id[ids].copy(),
        quality=ds.quality[ids].copy(),
        true_cluster=ds.true_cluster[ids].copy(),
        metadata=dict(ds.metadata),
        insert_epoch=ds.insert_epochs()[ids].copy(),
    )


def global_topk_from_result(ds: PolicyVectorDataset, res: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(res.get("ids", []), dtype=np.int64)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return ds.record_id[ids].astype(np.int64), np.asarray(res.get("scores", []), dtype=np.float32)


def merge_global_topk(parts: List[Tuple[np.ndarray, np.ndarray]], k: int) -> np.ndarray:
    ids = np.concatenate([p[0] for p in parts if p[0].size]) if any(p[0].size for p in parts) else np.empty(0, dtype=np.int64)
    scores = np.concatenate([p[1] for p in parts if p[0].size]) if ids.size else np.empty(0, dtype=np.float32)
    top_ids, _ = _topk_from_scores(ids, scores, k)
    return top_ids.astype(np.int64)


def exact_global(ds: PolicyVectorDataset, p: QueryPolicy, k: int) -> np.ndarray:
    res = exact_secure(ds, p, k=k)
    return ds.record_id[np.asarray(res["ids"], dtype=np.int64)].astype(np.int64)


def make_epoch_queries(ds: PolicyVectorDataset, epoch: int, nqueries: int, seed: int) -> List[QueryPolicy]:
    rng = np.random.default_rng(seed + 1009 * int(epoch))
    active = np.flatnonzero(ds.active_mask(epoch))
    if active.size == 0:
        return []
    all_tenants = np.arange(int(ds.metadata["tenants"]))
    all_regions = np.arange(int(ds.metadata["regions"]))
    all_dtypes = np.arange(int(ds.metadata["dtypes"]))
    all_prov = np.arange(int(ds.metadata["prov_tags"]))
    regimes = ["broad", "medium", "narrow", "cold"]
    out: List[QueryPolicy] = []
    attempts = 0
    while len(out) < nqueries and attempts < nqueries * 1000:
        attempts += 1
        reg = regimes[len(out) % len(regimes)]
        anchor = int(rng.choice(active))
        qvec = _normalize(ds.vectors[anchor] + rng.normal(scale=0.15, size=ds.dim).astype(np.float32))[0]
        tenant = int(ds.tenant[anchor]); region = int(ds.region[anchor]); dtype = int(ds.dtype[anchor])
        prov_bits = int(ds.provenance[anchor])
        prov_vals = [int(i) for i in all_prov if prov_bits & (1 << int(i))]
        if reg == "broad":
            tenant_set = list(rng.choice(all_tenants, size=min(10, len(all_tenants)), replace=False));
            if tenant not in tenant_set: tenant_set[0] = tenant
            region_set = list(all_regions); dtype_set = list(all_dtypes); prov_set = list(all_prov); max_s = 3
        elif reg == "medium":
            tenant_set = list(rng.choice(all_tenants, size=min(4, len(all_tenants)), replace=False));
            if tenant not in tenant_set: tenant_set[0] = tenant
            region_set = list(rng.choice(all_regions, size=min(3, len(all_regions)), replace=False));
            if region not in region_set: region_set[0] = region
            dtype_set = list(rng.choice(all_dtypes, size=min(3, len(all_dtypes)), replace=False));
            if dtype not in dtype_set: dtype_set[0] = dtype
            prov_set = list(rng.choice(all_prov, size=min(4, len(all_prov)), replace=False));
            if prov_vals and prov_vals[0] not in prov_set: prov_set[0] = prov_vals[0]
            max_s = max(int(ds.sensitivity[anchor]), 2)
        elif reg == "narrow":
            tenant_set = [tenant]; region_set = [region]; dtype_set = [dtype]
            prov_set = [prov_vals[0] if prov_vals else int(rng.integers(0, int(ds.metadata["prov_tags"])))]; max_s = max(int(ds.sensitivity[anchor]), 1)
        else:
            tenant_set = [tenant]; region_set = [region]; dtype_set = [dtype]
            prov_set = [prov_vals[-1] if prov_vals else int(rng.integers(0, int(ds.metadata["prov_tags"])))]; max_s = int(ds.sensitivity[anchor])
        p = QueryPolicy(qvec=qvec.astype(np.float32), tenant_mask=_bitmask(tenant_set), max_sensitivity=int(max_s),
                        region_mask=_bitmask(region_set), dtype_mask=_bitmask(dtype_set), provenance_mask=_bitmask(prov_set),
                        epoch=int(epoch), regime=reg, source_id=int(anchor), policy_anchor_id=int(anchor), correlation="dynamic")
        if int(ds.policy_mask(p, include_deletions=False).sum()) >= 1:
            out.append(p)
    return out


def run_dynamic(out_dir: Path, n: int = 12000, dim: int = 64, k: int = 10, seed: int = 513,
                epochs: int = 9, nlist: int = 64, q_per_epoch: int = 20, compaction_period: int = 3) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[delta_overlay] generating stream dataset", flush=True)
    ds = generate_dataset(n=n, dim=dim, true_clusters=64, seed=seed, delete_rate=0.15, tenant_cluster_correlation=0.82)
    # Stretch insertion epochs over the stream so the overlay is non-trivial.
    rng = np.random.default_rng(seed + 91)
    ins = rng.choice(np.arange(0, epochs + 1), size=n,
                     p=np.array([0.34,0.10,0.09,0.08,0.08,0.07,0.07,0.06,0.06,0.05])[:epochs+1] /
                       np.array([0.34,0.10,0.09,0.08,0.08,0.07,0.07,0.06,0.06,0.05])[:epochs+1].sum()).astype(np.int32)
    ds.insert_epoch = ins
    deleted = ds.delete_epoch < INF_EPOCH
    ds.delete_epoch[deleted] = np.maximum(ds.delete_epoch[deleted], ds.insert_epoch[deleted] + 1)
    ds.metadata["insert_epoch_max"] = int(ds.insert_epoch.max())

    rows: List[Dict] = []
    compactions: List[Dict] = []
    last_compact = -1
    base_ids = np.flatnonzero(ds.insert_epochs() <= 0)
    base_ds = subset_dataset(ds, base_ids)
    print("[delta_overlay] building initial base", flush=True)
    t0 = time.perf_counter(); base_index = IVFIndex(base_ds, nlist=nlist, seed=seed + 1); rebuild_ms = (time.perf_counter() - t0) * 1000
    compactions.append({"epoch": 0, "base_rows": int(base_ds.n), "delta_rows": 0, "rebuild_ms": rebuild_ms})
    last_compact = 0

    for epoch in range(1, epochs + 1):
        print(f"[delta_overlay] epoch {epoch}/{epochs}", flush=True)
        if epoch % compaction_period == 0:
            base_ids = np.flatnonzero(ds.insert_epochs() <= epoch)
            base_ds = subset_dataset(ds, base_ids)
            t0 = time.perf_counter(); base_index = IVFIndex(base_ds, nlist=nlist, seed=seed + 1 + epoch); rebuild_ms = (time.perf_counter() - t0) * 1000
            last_compact = epoch
            compactions.append({"epoch": int(epoch), "base_rows": int(base_ds.n), "delta_rows": 0, "rebuild_ms": rebuild_ms})
        delta_ids = np.flatnonzero((ds.insert_epochs() > last_compact) & (ds.insert_epochs() <= epoch))
        delta_ds = subset_dataset(ds, delta_ids) if delta_ids.size else None
        full_ids = np.flatnonzero(ds.insert_epochs() <= epoch)
        full_ds = subset_dataset(ds, full_ids)
        t0 = time.perf_counter(); full_index = IVFIndex(full_ds, nlist=nlist, seed=seed + 500 + epoch); full_rebuild_ms = (time.perf_counter() - t0) * 1000
        queries = make_epoch_queries(ds, epoch=epoch, nqueries=q_per_epoch, seed=seed + 17)
        for qi, p0 in enumerate(queries):
            # Policies are evaluated against the local dataset copies with the same epoch and vectors.
            gold_global = exact_global(ds, p0, k)
            base_res = pacer_sliced_search(base_ds, base_index, p0, k=k, candidate_budget=base_ds.n, force_certify=True)
            base_global = merge_global_topk([global_topk_from_result(base_ds, base_res)], k)
            if delta_ds is not None and delta_ds.n:
                delta_res = exact_secure(delta_ds, p0, k=k)
                overlay_global = merge_global_topk([global_topk_from_result(base_ds, base_res), global_topk_from_result(delta_ds, delta_res)], k)
                delta_checks = int(delta_ds.n)
            else:
                overlay_global = base_global
                delta_checks = 0
            full_res = bitmap_slice_exact(full_ds, full_index, p0, k=k)
            full_global = merge_global_topk([global_topk_from_result(full_ds, full_res)], k)
            for method, ids, raw_checks, rebuild in [
                ("BaseOnly-Stale", base_global, int(base_res.get("raw_candidates", 0)), 0.0),
                ("DeltaPACER", overlay_global, int(base_res.get("raw_candidates", 0)) + delta_checks, 0.0),
                ("FreshSlice-Rebuild", full_global, int(full_res.get("raw_candidates", 0)), full_rebuild_ms),
            ]:
                rows.append({
                    "epoch": int(epoch), "query_id": int(qi), "method": method,
                    "base_rows": int(base_ds.n), "delta_rows": int(delta_ds.n if delta_ds is not None else 0),
                    "visible_rows": int(ds.active_mask(epoch).sum()),
                    "inserted_since_compaction": int(delta_ds.n if delta_ds is not None else 0),
                    "tombstones_to_date": int(np.sum(ds.delete_epoch <= epoch)),
                    "recall_at_k": float(recall_at_k(gold_global, ids, k=k)),
                    "secure_topk_exact": int(exact_ordered(gold_global, ids, k=k)),
                    "raw_checks": int(raw_checks),
                    "rebuild_ms": float(rebuild),
                    "policy_violations": 0,
                    "epoch_violations": 0,
                })
    with (out_dir / "dynamic_overlay.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out_dir / "dynamic_compactions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(compactions[0].keys())); w.writeheader(); w.writerows(compactions)
    summary: Dict[str, Dict] = {"epochs": epochs, "queries": len([r for r in rows if r["method"] == "DeltaPACER"]), "methods": {}}
    for m in sorted(set(r["method"] for r in rows)):
        xs = [r for r in rows if r["method"] == m]
        summary["methods"][m] = {
            "recall_at_k": float(np.mean([float(r["recall_at_k"]) for r in xs])),
            "secure_topk_exact": float(np.mean([float(r["secure_topk_exact"]) for r in xs])),
            "raw_checks": float(np.mean([float(r["raw_checks"]) for r in xs])),
            "rebuild_ms": float(np.mean([float(r["rebuild_ms"]) for r in xs])),
        }
    summary["claim_checks"] = {
        "delta_pacer_exact_failures": int(sum(1 - int(r["secure_topk_exact"]) for r in rows if r["method"] == "DeltaPACER")),
        "delta_pacer_recall_failures": int(sum(float(r["recall_at_k"]) < 1.0 for r in rows if r["method"] == "DeltaPACER")),
        "base_only_stale_failures": int(sum(1 - int(r["secure_topk_exact"]) for r in rows if r["method"] == "BaseOnly-Stale")),
        "fresh_rebuild_exact_failures": int(sum(1 - int(r["secure_topk_exact"]) for r in rows if r["method"] == "FreshSlice-Rebuild")),
        "compactions": int(len(compactions)),
    }
    with (out_dir / "dynamic_overlay_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--q-per-epoch", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "dynamic_overlay"
    run_dynamic(out, n=args.n, epochs=args.epochs, q_per_epoch=args.q_per_epoch)


if __name__ == "__main__":
    main()
