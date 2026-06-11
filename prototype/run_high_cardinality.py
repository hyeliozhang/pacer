#!/usr/bin/env python3
"""High-cardinality policy-domain stress test for PACER.

This test exercises tenants/regions/dtypes beyond 63 values to verify that the
row verifier and bitmap summaries do not depend on fixed-width int64 shifts.
"""
from pathlib import Path
import argparse, csv, json, time
import numpy as np
from trustvql import generate_dataset, QueryPolicy, _normalize, _bitmask, IVFIndex, exact_secure, post_filter_ivf, prefilter_ivf, pacer_sliced_search, bitmap_slice_exact, exact_ordered, recall_at_k


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--n', type=int, default=12000)
    ap.add_argument('--tenants', type=int, default=160)
    ap.add_argument('--regions', type=int, default=24)
    ap.add_argument('--dtypes', type=int, default=12)
    ap.add_argument('--queries', type=int, default=48)
    ap.add_argument('--seed', type=int, default=4401)
    ap.add_argument('--budget', type=int, default=1800)
    args=ap.parse_args()
    out=Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "high_cardinality"; out.mkdir(parents=True, exist_ok=True)
    per=max(1,args.queries//4)
    ds=generate_dataset(n=args.n, dim=64, true_clusters=96, tenants=args.tenants, regions=args.regions, dtypes=args.dtypes, prov_tags=16, delete_rate=0.12, tenant_cluster_correlation=0.72, seed=args.seed)
    t0=time.perf_counter(); index=IVFIndex(ds,nlist=96,seed=args.seed+1,iters=3); build=(time.perf_counter()-t0)*1000
    rng=np.random.default_rng(args.seed+2)
    live_ids=np.flatnonzero(ds.active_mask(6))
    qs=[]
    for qi in range(args.queries):
        src=int(rng.choice(live_ids))
        qvec=_normalize(ds.vectors[src]+rng.normal(0,0.16,ds.dim).astype(np.float32))[0]
        tenants=set(int(x) for x in rng.choice(np.arange(args.tenants),size=min(24,args.tenants),replace=False)); tenants.add(int(ds.tenant[src]))
        regions=set(int(x) for x in rng.choice(np.arange(args.regions),size=min(8,args.regions),replace=False)); regions.add(int(ds.region[src]))
        dtypes=set(int(x) for x in rng.choice(np.arange(args.dtypes),size=min(5,args.dtypes),replace=False)); dtypes.add(int(ds.dtype[src]))
        prov_bits=int(ds.provenance[src]) or 1
        prov_vals=[i for i in range(16) if prov_bits & (1<<i)] or [0]
        qs.append(QueryPolicy(qvec=qvec.astype(np.float32), tenant_mask=_bitmask(tenants), max_sensitivity=3, region_mask=_bitmask(regions), dtype_mask=_bitmask(dtypes), provenance_mask=_bitmask(prov_vals), epoch=6, regime='highcard', source_id=src, policy_anchor_id=src, correlation='positive'))
    methods={
        'ExactSecure':lambda p: exact_secure(ds,p,k=10),
        'BitmapSlice-Exact':lambda p: bitmap_slice_exact(ds,index,p,k=10),
        'PostFilter-IVF':lambda p: post_filter_ivf(ds,index,p,k=10,nprobe=12,candidate_budget=args.budget),
        'PreFilter-IVF':lambda p: prefilter_ivf(ds,index,p,k=10,nprobe=12,candidate_budget=args.budget),
        'PACER-A':lambda p: pacer_sliced_search(ds,index,p,k=10,candidate_budget=args.budget),
        'PACER-X':lambda p: pacer_sliced_search(ds,index,p,k=10,candidate_budget=ds.n,force_certify=True),
    }
    rows=[]
    for qi,p in enumerate(qs):
        gold=exact_secure(ds,p,k=10)
        for m,fn in methods.items():
            r=fn(p); pol,dele,ten=ds.violations(r['ids'],p)
            rows.append({'query_id':qi,'method':m,'recall_at_k':recall_at_k(gold['ids'],r['ids'],10),'secure_topk_exact':int(exact_ordered(gold['ids'],r['ids'],10)),'policy_violations':pol,'deleted_violations':dele,'tenant_violations':ten,'latency_ms':r['latency_ms'],'raw_candidates':r.get('raw_candidates',0),'policy_checks':r.get('policy_checks',0),'certified':int(bool(r.get('certified',False)))})
    with (out/'high_cardinality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={}
    for m in methods:
        xs=[r for r in rows if r['method']==m]
        summary[m]={'recall_at_k':float(np.mean([x['recall_at_k'] for x in xs])),'secure_topk_exact':float(np.mean([x['secure_topk_exact'] for x in xs])),'violations':int(sum(x['policy_violations']+x['deleted_violations']+x['tenant_violations'] for x in xs)),'latency_ms':float(np.mean([x['latency_ms'] for x in xs])),'raw_candidates':float(np.mean([x['raw_candidates'] for x in xs]))}
    final={'n':ds.n,'queries':len(qs),'tenants':args.tenants,'regions':args.regions,'dtypes':args.dtypes,'build_ms':build,'summary':summary,'max_tenant':int(ds.tenant.max()),'bitmask_width_ok': int(args.tenants > 63)}
    with (out/'high_cardinality_summary.json').open('w') as f: json.dump(final,f,indent=2)
    print(json.dumps(final,indent=2))
if __name__=='__main__': main()
