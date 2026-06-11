from pathlib import Path
import argparse
import csv
import json
from trustvql import run_suite


def main() -> None:
    ap = argparse.ArgumentParser(description='Run PACER stress suite over correlation and deletion regimes')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n', type=int, default=8000)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--per-regime', type=int, default=6)
    ap.add_argument('--budget', type=int, default=2400)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / 'results'; out.mkdir(parents=True, exist_ok=True)
    configs = [
        ('positive', 0.05, 301, 0.82),
        ('positive', 0.20, 302, 0.82),
        ('independent', 0.12, 303, 0.20),
        ('negative', 0.12, 304, 0.82),
    ]
    keep = ['PostFilter-IVF','AdaptivePostFilter-IVF','TenantPartition-IVF','PreFilter-Exact','PACER-A','PACER-C','PACER-X','StaleNoRecheck','NaiveNoPolicy']
    rows = []
    for corr, delete_rate, seed, tenant_corr in configs:
        sub = out / f"stress_{corr}_del{str(delete_rate).replace('.', '')}"
        s = run_suite(sub, n=args.n, dim=args.dim, per_regime=args.per_regime, seed=seed,
                      nlist=64, candidate_budget=args.budget, correlation=corr,
                      delete_rate=delete_rate, tenant_cluster_correlation=tenant_corr)
        for method in keep:
            v = s['overall'][method]
            rows.append({
                'scenario': f'{corr}/del={delete_rate}', 'correlation': corr, 'delete_rate': delete_rate,
                'tenant_cluster_correlation': tenant_corr, 'method': method,
                'recall_at_k': v['recall_at_k'], 'secure_topk_exact': v['secure_topk_exact'],
                'policy_violations_per_query': v['policy_violations_per_query'],
                'deleted_violations_per_query': v['deleted_violations_per_query'],
                'tenant_violations_per_query': v['tenant_violations_per_query'],
                'candidates': v['candidates'], 'raw_candidates': v['raw_candidates'],
                'latency_ms': v['latency_ms'], 'median_latency_ms': v['median_latency_ms'],
                'certified_fraction': v['certified_fraction'], 'certified_sound_fraction': v['certified_sound_fraction'],
                'certificate_errors': v['certificate_errors'],
            })
    with (out / 'stress_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    with (out / 'stress_summary.json').open('w') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
