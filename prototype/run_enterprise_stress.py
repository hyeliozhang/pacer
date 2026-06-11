#!/usr/bin/env python3
"""Enterprise-style policy/deletion/provenance stress workload.

This workload is synthetic, but it stresses the dimensions that public feature
matrices do not contain: high-cardinality tenants, grouped ACL masks, bursty
logical deletion, insertion epochs, provenance lineages, and mixed selectivity
queries. It is deliberately separate from the default generator so that strong
claims do not depend on one benign data distribution.
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
from typing import Dict, List
import numpy as np

from trustvql import (
    INF_EPOCH, QueryPolicy, PolicyVectorDataset, IVFIndex, _normalize, _bitmask,
    exact_secure, post_filter_ivf, prefilter_ivf, bitmap_slice_exact, pacer_sliced_search,
    contract_planner, recall_at_k, exact_ordered, failure_mode
)


def make_enterprise_dataset(n=22000, dim=96, seed=41) -> PolicyVectorDataset:
    rng = np.random.default_rng(seed)
    clusters = 72
    centers = _normalize(rng.normal(size=(clusters, dim)).astype(np.float32))
    cid = rng.choice(clusters, size=n, p=np.linspace(1.7, 0.3, clusters) / np.linspace(1.7,0.3,clusters).sum()).astype(np.int16)
    vectors = _normalize(centers[cid] + rng.normal(scale=0.17, size=(n, dim)).astype(np.float32)).astype(np.float32)
    # Zipf-like tenant popularity with many cold tenants.
    tenants = 128
    ranks = np.arange(1, tenants + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** 1.08); probs = probs / probs.sum()
    tenant_base = rng.choice(tenants, size=n, p=probs)
    tenant_cluster = (cid * 3 + rng.integers(0, 7, size=n)) % tenants
    tenant = np.where(rng.random(n) < 0.72, tenant_cluster, tenant_base).astype(np.int16)
    region = ((cid + rng.integers(0, 4, size=n)) % 12).astype(np.int16)
    dtype = ((cid // 3 + rng.integers(0, 3, size=n)) % 9).astype(np.int16)
    sensitivity = np.minimum(5, (cid % 6) + (rng.random(n) < 0.18)).astype(np.int8)
    # Provenance is a small lineage: source system + transformation + optional parent.
    prov_tags = 18
    provenance = np.zeros(n, dtype=np.int64)
    for i in range(n):
        src = int(cid[i] % 9)
        transform = 9 + int((cid[i] // 8) % 6)
        mask = (1 << src) | (1 << transform)
        if rng.random() < 0.28:
            mask |= 1 << int(rng.integers(0, prov_tags))
        provenance[i] = mask
    # Insertions and bursty document-family deletions.
    insert_epoch = rng.choice(np.arange(0, 16), size=n, p=np.array([.22,.12,.10,.08,.08,.07,.06,.06,.05,.04,.04,.03,.02,.015,.01,.005])).astype(np.int32)
    delete_epoch = np.full(n, INF_EPOCH, dtype=np.int32)
    family = (tenant.astype(np.int32) * 31 + region.astype(np.int32) * 7 + dtype.astype(np.int32)) % 512
    families = np.unique(family)
    deleted_fams = set(rng.choice(families, size=int(0.18 * len(families)), replace=False).tolist())
    for fam in deleted_fams:
        ids = np.flatnonzero(family == fam)
        if ids.size == 0: continue
        # burst epoch after the median insertion in the family
        burst = int(max(np.median(insert_epoch[ids]) + rng.integers(2, 12), 2))
        sel = ids[rng.random(ids.size) < rng.uniform(0.35, 0.85)]
        delete_epoch[sel] = np.maximum(insert_epoch[sel] + 1, burst + rng.integers(0, 4, size=sel.size)).astype(np.int32)
    quality = rng.random(n).astype(np.float32)
    return PolicyVectorDataset(vectors=vectors, tenant=tenant, sensitivity=sensitivity, region=region, dtype=dtype,
        provenance=provenance, delete_epoch=delete_epoch, record_id=np.arange(n), quality=quality, true_cluster=cid,
        metadata={"workload":"enterprise_stress", "tenants":tenants, "regions":12, "dtypes":9, "prov_tags":prov_tags},
        insert_epoch=insert_epoch)


def make_enterprise_queries(ds: PolicyVectorDataset, per_regime=18, seed=43) -> List[QueryPolicy]:
    rng = np.random.default_rng(seed)
    queries: List[QueryPolicy] = []
    regimes = ["broad", "medium", "narrow", "cold"]
    active_ids_by_epoch = {e: np.flatnonzero(ds.active_mask(e)) for e in range(3, 20)}
    attempts = 0
    while len(queries) < per_regime * len(regimes) and attempts < 20000:
        attempts += 1
        reg = regimes[len(queries) % len(regimes)]
        epoch = int(rng.integers(5, 18))
        active = active_ids_by_epoch.get(epoch, np.flatnonzero(ds.active_mask(epoch)))
        if active.size == 0: continue
        anchor = int(rng.choice(active))
        qvec = _normalize(ds.vectors[anchor] + rng.normal(scale=0.055, size=ds.dim).astype(np.float32))[0]
        tenant = int(ds.tenant[anchor])
        # ACL group: own tenant plus adjacent tenants in an organization shard.
        shard = tenant // 8
        shard_tenants = list(range(shard * 8, min(128, shard * 8 + 8)))
        if reg == "broad":
            tenants = shard_tenants + [int(rng.integers(0, 128)) for _ in range(6)]
            regions = list(range(12)); dtypes = list(range(9)); max_s = 5
        elif reg == "medium":
            tenants = [tenant] + list(rng.choice(shard_tenants, size=min(3, len(shard_tenants)), replace=False))
            regions = [int(ds.region[anchor]), int((int(ds.region[anchor]) + 1) % 12)]
            dtypes = [int(ds.dtype[anchor]), int((int(ds.dtype[anchor]) + 1) % 9)]
            max_s = max(int(ds.sensitivity[anchor]), 2)
        elif reg == "narrow":
            tenants = [tenant]
            regions = [int(ds.region[anchor])]
            dtypes = [int(ds.dtype[anchor])]
            max_s = int(ds.sensitivity[anchor])
        else:
            tenants = [tenant]
            regions = [int(ds.region[anchor])]
            dtypes = [int(ds.dtype[anchor])]
            max_s = max(0, int(ds.sensitivity[anchor]) - 1)
        prov_bits = [b for b in range(18) if (int(ds.provenance[anchor]) >> b) & 1]
        if not prov_bits: prov_bits = [int(rng.integers(0, 18))]
        prov = [int(rng.choice(prov_bits))]
        p = QueryPolicy(qvec=qvec.astype(np.float32), tenant_mask=_bitmask(tenants), max_sensitivity=max_s,
                        region_mask=_bitmask(regions), dtype_mask=_bitmask(dtypes), provenance_mask=_bitmask(prov),
                        epoch=epoch, regime=reg, source_id=anchor, policy_anchor_id=anchor, correlation="enterprise")
        min_allowed = {"broad": 40, "medium": 8, "narrow": 2, "cold": 1}[reg]
        if int(ds.policy_mask(p, include_deletions=False).sum()) >= min_allowed:
            queries.append(p)
    return queries


def eval_result(ds, p, gold, res, method, qi, k):
    pv, dv, tv = ds.violations(res["ids"], p)
    return {"query_id": qi, "method": method, "regime": p.regime,
            "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=k),
            "secure_topk_exact": int(exact_ordered(gold["ids"], res["ids"], k=k)),
            "failure_mode": failure_mode(gold["ids"], res["ids"], k=k),
            "policy_violations": pv, "deleted_violations": dv, "tenant_violations": tv,
            "latency_ms": float(res["latency_ms"]), "raw_candidates": int(res.get("raw_candidates",0)),
            "distance_evals": int(res.get("distance_evals",0)), "policy_checks": int(res.get("policy_checks",0)),
            "certified": int(bool(res.get("certified", False)))}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "enterprise_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = make_enterprise_dataset()
    t0 = time.perf_counter(); ivf = IVFIndex(ds, nlist=72, seed=44); build_ms = (time.perf_counter() - t0) * 1000
    queries = make_enterprise_queries(ds)
    k=10; rows=[]
    for qi,p in enumerate(queries):
        gold = exact_secure(ds, p, k=k)
        methods = {
            "PostFilter-IVF": lambda: post_filter_ivf(ds, ivf, p, k=k, nprobe=10, candidate_budget=2500),
            "PreFilter-IVF": lambda: prefilter_ivf(ds, ivf, p, k=k, nprobe=10, candidate_budget=2500),
            "BitmapSlice-Exact": lambda: bitmap_slice_exact(ds, ivf, p, k=k),
            "PACER-A": lambda: pacer_sliced_search(ds, ivf, p, k=k, candidate_budget=1200, force_certify=False),
            "PACER-C": lambda: contract_planner(ds, ivf, p, k=k, candidate_budget=1200, service_level="certify-if-cheap"),
            "PACER-X": lambda: pacer_sliced_search(ds, ivf, p, k=k, candidate_budget=ds.n, force_certify=True),
        }
        for name,fn in methods.items(): rows.append(eval_result(ds,p,gold,fn(),name,qi,k))
    with (out_dir/"enterprise_stress.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={"n":ds.n,"dim":ds.dim,"queries":len(queries),"rows":len(rows),"index_build_ms":build_ms,"methods":{},"claim_checks":{}}
    for m in sorted({r['method'] for r in rows}):
        xs=[r for r in rows if r['method']==m]
        summary['methods'][m]={"recall_at_k":float(np.mean([r['recall_at_k'] for r in xs])),"secure_topk_exact":float(np.mean([r['secure_topk_exact'] for r in xs])),"raw_candidates":float(np.mean([r['raw_candidates'] for r in xs])),"latency_ms":float(np.mean([r['latency_ms'] for r in xs])),"violations":int(sum(r['policy_violations']+r['deleted_violations']+r['tenant_violations'] for r in xs)),"certified_fraction":float(np.mean([r['certified'] for r in xs]))}
    safe = {"PostFilter-IVF","PreFilter-IVF","BitmapSlice-Exact","PACER-A","PACER-C","PACER-X"}
    summary['claim_checks']={"safe_method_violations":int(sum(r['policy_violations']+r['deleted_violations']+r['tenant_violations'] for r in rows if r['method'] in safe)),"pacer_c_exact_failures":int(sum(1-r['secure_topk_exact'] for r in rows if r['method']=='PACER-C')),"pacer_x_exact_failures":int(sum(1-r['secure_topk_exact'] for r in rows if r['method']=='PACER-X')),"pacer_a_beats_postfilter_exactness":bool(summary['methods']['PACER-A']['secure_topk_exact']>summary['methods']['PostFilter-IVF']['secure_topk_exact'])}
    with (out_dir/"enterprise_stress_summary.json").open("w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary['claim_checks'], indent=2))

if __name__=='__main__':
    main()
