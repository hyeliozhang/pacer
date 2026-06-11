from pathlib import Path
import csv, json, sys, time
import numpy as np
from trustvql import (
    generate_dataset, generate_queries, IVFIndex, exact_secure,
    post_filter_ivf, adaptive_post_filter_ivf, caps_search, pacer_sliced_search,
    exact_ordered, recall_at_k,
)

def mean(xs):
    return float(np.mean(xs)) if xs else 0.0

def run_one(out: Path, scenario: str, corr: str, delete_rate: float, seed: int, tenant_corr: float, n: int, per_regime: int, budget: int):
    print(f"build {scenario}", flush=True)
    ds = generate_dataset(n=n, dim=64, seed=seed, delete_rate=delete_rate, tenant_cluster_correlation=tenant_corr)
    index = IVFIndex(ds, nlist=64, seed=seed+1)
    qs = generate_queries(ds, per_regime=per_regime, seed=seed+2, correlation=corr)
    methods = {
        "PostFilter-IVF": lambda p: post_filter_ivf(ds, index, p, k=10, nprobe=8, candidate_budget=budget),
        "AdaptivePostFilter-IVF": lambda p: adaptive_post_filter_ivf(ds, index, p, k=10, candidate_budget=max(budget, 4800)),
        "PreFilter-Exact": lambda p: exact_secure(ds, p, k=10),
        "PACER-A": lambda p: pacer_sliced_search(ds, index, p, k=10, candidate_budget=budget),
        "PACER-X": lambda p: pacer_sliced_search(ds, index, p, k=10, candidate_budget=ds.n, force_certify=True),
    }
    golds = [exact_secure(ds, p, k=10) for p in qs]
    rows=[]
    for name, fn in methods.items():
        vals=[]
        for p,g in zip(qs,golds):
            r = fn(p)
            pv,dv,tv = ds.violations(r['ids'], p)
            vals.append({
                'recall_at_k': recall_at_k(g['ids'], r['ids'], k=10),
                'secure_topk_exact': int(exact_ordered(g['ids'], r['ids'], k=10)),
                'policy_violations_per_query': pv,
                'deleted_violations_per_query': dv,
                'tenant_violations_per_query': tv,
                'candidates': int(r['candidates']),
                'raw_candidates': int(r.get('raw_candidates', r['candidates'])),
                'latency_ms': float(r['latency_ms']),
                'certified_fraction': int(bool(r.get('certified', False))),
                'certificate_false_positive': int(bool(r.get('certified', False)) and not exact_ordered(g['ids'], r['ids'], k=10)),
            })
        rows.append({
            'scenario': scenario, 'correlation': corr, 'delete_rate': delete_rate,
            'tenant_cluster_correlation': tenant_corr, 'method': name,
            'queries': len(qs), 'n': n,
            'recall_at_k': mean([v['recall_at_k'] for v in vals]),
            'secure_topk_exact': mean([v['secure_topk_exact'] for v in vals]),
            'policy_violations_per_query': mean([v['policy_violations_per_query'] for v in vals]),
            'deleted_violations_per_query': mean([v['deleted_violations_per_query'] for v in vals]),
            'tenant_violations_per_query': mean([v['tenant_violations_per_query'] for v in vals]),
            'candidates': mean([v['candidates'] for v in vals]),
            'raw_candidates': mean([v['raw_candidates'] for v in vals]),
            'latency_ms': mean([v['latency_ms'] for v in vals]),
            'median_latency_ms': float(np.median([v['latency_ms'] for v in vals])),
            'certified_fraction': mean([v['certified_fraction'] for v in vals]),
            'certificate_false_positives': int(sum(v['certificate_false_positive'] for v in vals)),
        })
    return rows

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run lightweight PACER stress scenarios")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--per-regime", type=int, default=5)
    ap.add_argument("--budget", type=int, default=1200)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    configs = [
        ("positive/del=0.05", "positive", 0.05, 301, 0.82),
        ("positive/del=0.20", "positive", 0.20, 302, 0.82),
        ("independent/del=0.12", "independent", 0.12, 303, 0.20),
        ("negative/del=0.12", "negative", 0.12, 304, 0.82),
    ]
    rows=[]
    for c in configs:
        rows.extend(run_one(out,*c,n=args.n,per_regime=args.per_regime,budget=args.budget))
    path=out/'stress_summary.csv'
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out/'stress_summary.json').open('w') as f:
        json.dump(rows,f,indent=2)
    print(json.dumps(rows,indent=2))

if __name__ == '__main__':
    main()
