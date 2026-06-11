#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Any, Set
import numpy as np
from trustvql import generate_dataset, IVFIndex

def deep_size(obj: Any, seen: Set[int] | None = None) -> int:
    if seen is None: seen=set()
    oid=id(obj)
    if oid in seen: return 0
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        return int(sys.getsizeof(obj) + obj.nbytes)
    size=sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_size(k,seen)+deep_size(v,seen) for k,v in obj.items())
    elif isinstance(obj, (list,tuple,set,frozenset)):
        size += sum(deep_size(x,seen) for x in obj)
    elif hasattr(obj,'__dict__'):
        size += deep_size(vars(obj),seen)
    return int(size)

def main():
    root=Path(__file__).resolve().parents[1]; out=root/'results'/'memory_audit'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n,nlist in [(15000,64),(60000,96)]:
        t=time.perf_counter(); ds=generate_dataset(n=n,dim=64,true_clusters=max(64,nlist),tenants=48 if n>20000 else 24,regions=8 if n>20000 else 6,dtypes=8 if n>20000 else 5,prov_tags=12 if n>20000 else 8,seed=1900+n,delete_rate=0.12,tenant_cluster_correlation=0.82); gen=(time.perf_counter()-t)*1000
        t=time.perf_counter(); idx=IVFIndex(ds,nlist=nlist,seed=1901+n,iters=2 if n>20000 else 3); build=(time.perf_counter()-t)*1000
        logical=idx.index_overhead()
        idx_bytes=deep_size({'assignments':idx.assignments,'centroids':idx.centroids,'radii':idx.radii,'angular_radii':idx.angular_radii,'signature_sets':idx.signature_sets,'slice_postings':idx.slice_postings,'cluster_bitsets':idx.cluster_bitsets,'tenant_bits':idx.tenant_bits,'region_bits':idx.region_bits,'dtype_bits':idx.dtype_bits,'prov_bits':idx.prov_bits,'min_sensitivity':idx.min_sensitivity,'min_insert_epoch':idx.min_insert_epoch,'max_delete_epoch':idx.max_delete_epoch,'tenant_value_bitsets':idx.tenant_value_bitsets,'region_value_bitsets':idx.region_value_bitsets,'dtype_value_bitsets':idx.dtype_value_bitsets,'prov_value_bitsets':idx.prov_value_bitsets,'sensitivity_leq_bitsets':idx.sensitivity_leq_bitsets})
        rows.append({'n':n,'nlist':nlist,'data_generation_ms':gen,'index_build_ms':build,'logical_structural_bytes_per_record':logical['total_structural_bytes_per_record'],'python_object_bytes_per_record':idx_bytes/float(n),'python_object_total_mb':idx_bytes/(1024*1024)})
    with (out/'memory_audit.json').open('w') as f: json.dump({'rows':rows,'claim_checks':{'has_empirical_python_object_accounting':all(r['python_object_bytes_per_record']>0 for r in rows),'has_60k_memory_audit':any(r['n']==60000 for r in rows)}},f,indent=2)
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
