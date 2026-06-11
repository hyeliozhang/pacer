from pathlib import Path
import argparse
import csv
import json
from trustvql import run_suite


def main() -> None:
    ap = argparse.ArgumentParser(description='Run compact PACER scalability sweep')
    ap.add_argument('--out', default=None)
    ap.add_argument('--sizes', nargs='+', type=int, default=[5000, 10000, 15000])
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--per-regime', type=int, default=8)
    ap.add_argument('--budget', type=int, default=2400)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / 'results'; out.mkdir(parents=True, exist_ok=True)
    keep = ['ExactSecure','PreFilter-Exact','PostFilter-IVF','AdaptivePostFilter-IVF','TenantPartition-IVF','PACER-A','PACER-C','PACER-X']
    rows = []
    for i, n in enumerate(args.sizes):
        summary = run_suite(out / f'scale_{n}', n=n, dim=args.dim, per_regime=args.per_regime,
                            seed=100+i, nlist=max(32, min(96, n//160)), candidate_budget=args.budget,
                            correlation='positive')
        for method in keep:
            v = summary['overall'][method]
            rows.append({
                'n': n, 'method': method,
                'recall_at_k': v['recall_at_k'], 'secure_topk_exact': v['secure_topk_exact'],
                'latency_ms': v['latency_ms'], 'median_latency_ms': v['median_latency_ms'],
                'p95_latency_ms': v['p95_latency_ms'], 'candidates': v['candidates'],
                'raw_candidates': v['raw_candidates'], 'policy_violations_per_query': v['policy_violations_per_query'],
                'deleted_violations_per_query': v['deleted_violations_per_query'],
                'tenant_violations_per_query': v['tenant_violations_per_query'],
                'certified_fraction': v['certified_fraction'], 'certified_sound_fraction': v['certified_sound_fraction'],
                'certificate_errors': v['certificate_errors'],
            })
    with (out / 'scalability.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    with (out / 'scalability.json').open('w') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
