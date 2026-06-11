from pathlib import Path
import argparse
import json
import os
from trustvql import run_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CPU-only PACER policy-vector experiments")
    ap.add_argument("--out", default=None, help="output directory (default: repository results/)")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--per-regime", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nlist", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--correlation", choices=["positive", "independent", "negative"], default="positive")
    ap.add_argument("--delete-rate", type=float, default=0.12)
    ap.add_argument("--tenant-cluster-correlation", type=float, default=0.82)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results"
    summary = run_suite(
        out_dir,
        n=args.n,
        dim=args.dim,
        per_regime=args.per_regime,
        seed=args.seed,
        nlist=args.nlist,
        candidate_budget=args.budget,
        correlation=args.correlation,
        delete_rate=args.delete_rate,
        tenant_cluster_correlation=args.tenant_cluster_correlation,
    )
    if os.environ.get("PACER_QUIET_JSON") != "1":
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps({"status": "ok", "n": summary["n"], "queries": summary["queries"], "query_method_rows": summary["query_method_rows"], "methods": len(summary["overall"])}, indent=2))


if __name__ == "__main__":
    main()
