#!/usr/bin/env python3
from __future__ import annotations
import csv, json, time
from pathlib import Path
from typing import Dict, List
import numpy as np
from trustvql import generate_dataset, generate_queries, IVFIndex, exact_secure, post_filter_ivf, prefilter_ivf, bitmap_slice_exact, pacer_sliced_search, contract_planner, recall_at_k, exact_ordered, failure_mode

def row(ds,p,gold,res,method,qi,k,n):
    pv,dv,tv=ds.violations(res['ids'],p)
    return {'query_id':qi,'method':method,'n':n,'recall_at_k':recall_at_k(gold['ids'],res['ids'],k=k),'ordered_exact_secure_topk':int(exact_ordered(gold['ids'],res['ids'],k=k)),'failure_mode':failure_mode(gold['ids'],res['ids'],k=k),'policy_violations':pv,'deleted_violations':dv,'tenant_violations':tv,'latency_ms':float(res.get('latency_ms',0.0)),'raw_candidates':int(res.get('raw_candidates',0)),'distance_evals':int(res.get('distance_evals',0)),'policy_checks':int(res.get('policy_checks',0)),'summary_checks':int(res.get('summary_checks',0)),'visited_cluster_count':int(res.get('visited_cluster_count',0)),'certified':int(bool(res.get('certified',False))),'certificate_gap':float(res.get('certificate_gap',0.0))}

def main():
    root=Path(__file__).resolve().parents[1]; out=root/'results'/'large_scale'; out.mkdir(parents=True,exist_ok=True)
    n=60000; dim=64; nlist=96; k=10; budget=2400
    print('[large_scale] generating data', flush=True)
    t=time.perf_counter(); ds=generate_dataset(n=n,dim=dim,true_clusters=96,tenants=48,regions=8,dtypes=8,prov_tags=12,seed=1701,delete_rate=0.12,tenant_cluster_correlation=0.82); data_ms=(time.perf_counter()-t)*1000
    print('[large_scale] building index', flush=True)
    t=time.perf_counter(); index=IVFIndex(ds,nlist=nlist,seed=1702,iters=2); build_ms=(time.perf_counter()-t)*1000
    qs=generate_queries(ds,per_regime=3,seed=1703,correlation='positive')
    funcs={'PostFilter-IVF':lambda p:post_filter_ivf(ds,index,p,k=k,nprobe=12,candidate_budget=budget),'PreFilter-IVF':lambda p:prefilter_ivf(ds,index,p,k=k,nprobe=12,candidate_budget=budget),'BitmapSlice-Exact':lambda p:bitmap_slice_exact(ds,index,p,k=k),'PACER-A':lambda p:pacer_sliced_search(ds,index,p,k=k,candidate_budget=budget,force_certify=False),'PACER-C':lambda p:contract_planner(ds,index,p,k=k,candidate_budget=budget,service_level='certify'),'PACER-X':lambda p:pacer_sliced_search(ds,index,p,k=k,candidate_budget=ds.n,force_certify=True)}
    rows=[]
    for qi,p in enumerate(qs):
        print(f'[large_scale] query {qi+1}/{len(qs)}', flush=True)
        gold=exact_secure(ds,p,k=k); rows.append(row(ds,p,gold,gold,'ExactSecure',qi,k,n))
        for m,fn in funcs.items(): rows.append(row(ds,p,gold,fn(p),m,qi,k,n))
    with (out/'large_scale_60k.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    by={}
    for m in sorted({r['method'] for r in rows}):
        xs=[r for r in rows if r['method']==m]
        by[m]={'recall_at_k':float(np.mean([r['recall_at_k'] for r in xs])),'ordered_exact_secure_topk':float(np.mean([r['ordered_exact_secure_topk'] for r in xs])),'latency_ms_mean':float(np.mean([r['latency_ms'] for r in xs])),'latency_ms_median':float(np.median([r['latency_ms'] for r in xs])),'latency_ms_p95':float(np.percentile([r['latency_ms'] for r in xs],95)),'raw_candidates_mean':float(np.mean([r['raw_candidates'] for r in xs])),'policy_checks_mean':float(np.mean([r['policy_checks'] for r in xs])),'summary_checks_mean':float(np.mean([r['summary_checks'] for r in xs])),'certified_fraction':float(np.mean([r['certified'] for r in xs])),'violations_total':int(sum(r['policy_violations']+r['deleted_violations']+r['tenant_violations'] for r in xs))}
    summary={'n':n,'dim':dim,'queries':len(qs),'method_query_rows':len(rows),'nlist':nlist,'budget':budget,'data_generation_ms':data_ms,'index_build_ms':build_ms,'index_overhead':index.index_overhead(),'methods_summary':by,'claim_checks':{'safe_method_violations_total':int(sum(r['policy_violations']+r['deleted_violations']+r['tenant_violations'] for r in rows)),'pacer_x_exact_failures':int(sum(1-r['ordered_exact_secure_topk'] for r in rows if r['method']=='PACER-X')),'pacer_c_exact_failures':int(sum(1-r['ordered_exact_secure_topk'] for r in rows if r['method']=='PACER-C')),'pacer_a_raw_less_than_postfilter':bool(by['PACER-A']['raw_candidates_mean']<by['PostFilter-IVF']['raw_candidates_mean'])}}
    with (out/'large_scale_60k_summary.json').open('w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary['claim_checks'],indent=2))
if __name__=='__main__': main()
