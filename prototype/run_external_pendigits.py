"""Run PACER on a real public vector dataset (Pendigits sta16).

The Pendigits mirror used here is the public bschlief/pendigits GitHub mirror of
UCI/UCSD Pendigits. The code does not call an embedding API: each record is an
observed 16x16 bitmap-derived feature vector. Policies, deletion epochs, and
provenance bits are derived deterministically from labels and vector statistics
so that the benchmark remains a policy-constrained vector-query workload rather
than a classification benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from trustvql import (
    INF_EPOCH,
    IVFIndex,
    PolicyVectorDataset,
    _method_dict,
    exact_ordered,
    exact_secure,
    failure_mode,
    generate_queries,
    recall_at_k,
    summarize_metrics,
    _normalize,
)


def _load_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",", dtype=np.float32)


def _load_label_vector(path: Path) -> np.ndarray:
    labels = np.loadtxt(str(path), dtype=np.int32)
    # The mirror uses 1..10. Map 10 to digit 0 and 1..9 to themselves.
    labels = np.where(labels == 10, 0, labels)
    return labels.astype(np.int16)


def load_pendigits_dataset(data_dir: Path, delete_rate: float = 0.10, seed: int = 921) -> PolicyVectorDataset:
    data_dir = Path(data_dir)
    x_train = _load_csv_matrix(data_dir / "pendigits_sta16_train.csv")
    x_test = _load_csv_matrix(data_dir / "pendigits_sta16_test.csv")
    y_train = _load_label_vector(data_dir / "pendigits_label_train.csv")
    y_test = _load_label_vector(data_dir / "pendigits_label_test.csv")
    x = np.vstack([x_train, x_test]).astype(np.float32)
    labels = np.concatenate([y_train, y_test]).astype(np.int16)
    split = np.concatenate([np.zeros(len(y_train), dtype=np.int16), np.ones(len(y_test), dtype=np.int16)])
    vectors = _normalize(x)
    n, dim = vectors.shape
    rng = np.random.default_rng(seed)

    # Metadata is derived from real vector features and labels. It is not a new
    # supervised task: labels emulate tenant/application domains, while vector
    # statistics emulate scalar attributes often carried by vector rows.
    density = x.mean(axis=1)
    density_bin = np.digitize(density, np.quantile(density, [0.2, 0.4, 0.6, 0.8])).astype(np.int16)
    img = x.reshape(n, 16, 16)
    left_mass = img[:, :, :8].sum(axis=(1, 2))
    right_mass = img[:, :, 8:].sum(axis=(1, 2))
    top_mass = img[:, :8, :].sum(axis=(1, 2))
    bottom_mass = img[:, 8:, :].sum(axis=(1, 2))
    asym = ((right_mass > left_mass).astype(np.int16) + 2 * (bottom_mass > top_mass).astype(np.int16))

    tenants = 10
    regions = 6
    dtypes = 5
    prov_tags = 8
    tenant = labels.astype(np.int16)
    sensitivity = np.minimum(3, (density_bin + (labels % 3)) // 2).astype(np.int8)
    region = ((labels + asym + split) % regions).astype(np.int8)
    dtype = ((density_bin + split + (labels // 2)) % dtypes).astype(np.int8)

    provenance = np.zeros(n, dtype=np.int16)
    for i in range(n):
        m = 1 << int(split[i])  # source split bit
        m |= 1 << int(2 + (labels[i] % 3))
        m |= 1 << int(5 + (density_bin[i] % 3))
        provenance[i] = int(m)

    insert_epoch = ((labels + density_bin + 2 * split) % 6).astype(np.int32)
    delete_epoch = np.full(n, INF_EPOCH, dtype=np.int32)
    # Deterministic but feature-dependent deletes emulate stale vector postings.
    score = (density * 17.0 + labels * 0.07 + split * 0.13 + rng.random(n) * 0.01)
    cutoff = np.quantile(score, delete_rate)
    deleted = score <= cutoff
    delete_epoch[deleted] = (insert_epoch[deleted] + 1 + ((labels[deleted] + density_bin[deleted] + split[deleted]) % 7)).astype(np.int32)

    # true_cluster is not used by the index builder. It only records dataset structure.
    true_cluster = ((labels * 2 + split) % 20).astype(np.int16)
    return PolicyVectorDataset(
        vectors=vectors.astype(np.float32),
        tenant=tenant,
        sensitivity=sensitivity,
        region=region,
        dtype=dtype,
        provenance=provenance,
        delete_epoch=delete_epoch,
        insert_epoch=insert_epoch,
        record_id=np.arange(n, dtype=np.int64),
        quality=density.astype(np.float32),
        true_cluster=true_cluster,
        metadata={
            "dataset_name": "pendigits_sta16_public",
            "source": "UCI/UCSD Pendigits mirror bschlief/pendigits",
            "representation": "sta16 16x16 bitmap vectors",
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "true_clusters": int(20),
            "tenants": int(tenants),
            "regions": int(regions),
            "dtypes": int(dtypes),
            "prov_tags": int(prov_tags),
            "delete_rate": float(deleted.mean()),
            "tenant_cluster_correlation": 1.0,
            "seed": int(seed),
            "insert_epoch_min": int(insert_epoch.min()),
            "insert_epoch_max": int(insert_epoch.max()),
        },
    )


def run_external_suite(
    out_dir: Path,
    data_dir: Path,
    per_regime: int = 8,
    k: int = 10,
    seed: int = 921,
    nlist: int = 64,
    candidate_budget: int = 2400,
) -> Dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_pendigits_dataset(data_dir, seed=seed)
    index = IVFIndex(ds, nlist=nlist, seed=seed + 1, iters=4)
    queries = generate_queries(ds, per_regime=per_regime, seed=seed + 2, correlation="positive")
    methods = _method_dict(ds, index, k, candidate_budget)
    golds = [exact_secure(ds, p, k=k) for p in queries]

    rows: List[Dict] = []
    for qi, (p, gold) in enumerate(zip(queries, golds)):
        allowed = int(ds.policy_mask(p, include_deletions=False).sum())
        selectivity = float(allowed / ds.n)
        for name, fn in methods.items():
            res = fn(p)
            policy_v, del_v, tenant_v = ds.violations(res["ids"], p)
            ev, et = ds.explanations_valid(res["ids"], p)
            exact = int(exact_ordered(gold["ids"], res["ids"], k=k))
            certified = int(bool(res.get("certified", False)))
            rows.append({
                "query_id": qi,
                "regime": p.regime,
                "correlation": "real_pendigits",
                "method": name,
                "n": ds.n,
                "dim": ds.dim,
                "k": k,
                "allowed_count": allowed,
                "selectivity": selectivity,
                "returned": int(len(res["ids"])),
                "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=k),
                "secure_topk_exact": exact,
                "failure_mode": failure_mode(gold["ids"], res["ids"], k=k),
                "policy_violations": policy_v,
                "deleted_violations": del_v,
                "tenant_violations": tenant_v,
                "explanation_valid": ev,
                "explanation_total": et,
                "explanation_accuracy": float(ev / et) if et else 1.0,
                "latency_ms": float(res["latency_ms"]),
                "raw_candidates": int(res.get("raw_candidates", res.get("candidates", 0))),
                "candidates": int(res["candidates"]),
                "distance_evals": int(res.get("distance_evals", res.get("candidates", 0))),
                "policy_checks": int(res.get("policy_checks", res.get("raw_candidates", 0))),
                "summary_checks": int(res.get("summary_checks", 0)),
                "visited_cluster_count": int(res.get("visited_cluster_count", 0)),
                "certified": certified,
                "certified_and_exact": int((not certified) or bool(exact)),
                "certificate_false_positive": int(certified == 1 and exact == 0),
                "certificate_gap": float(res.get("certificate_gap", 0.0)),
                "truncated_visible_slice": int(res.get("truncated_visible_slice", 0)),
            })

    metrics_path = out_dir / "metrics_pendigits.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # compact budget and deletion checks for shared summarizer
    budget_rows: List[Dict] = []
    for budget in [100, 200, 400, 700]:
        for qi, (p, gold) in enumerate(zip(queries[:24], golds[:24])):
            for name, fn in {
                "PostFilter-IVF": lambda p, b=budget: _method_dict(ds, index, k, b)["PostFilter-IVF"](p),
                "PACER-A": lambda p, b=budget: _method_dict(ds, index, k, b)["PACER-A"](p),
            }.items():
                res = fn(p)
                policy_v, del_v, tenant_v = ds.violations(res["ids"], p)
                budget_rows.append({
                    "query_id": qi,
                    "regime": p.regime,
                    "correlation": "real_pendigits",
                    "method": name,
                    "budget": int(budget),
                    "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=k),
                    "secure_topk_exact": int(exact_ordered(gold["ids"], res["ids"], k=k)),
                    "latency_ms": float(res["latency_ms"]),
                    "candidates": int(res["candidates"]),
                    "raw_candidates": int(res.get("raw_candidates", 0)),
                    "distance_evals": int(res.get("distance_evals", 0)),
                    "policy_checks": int(res.get("policy_checks", 0)),
                    "policy_violations": policy_v,
                    "deleted_violations": del_v,
                    "tenant_violations": tenant_v,
                    "certified": int(bool(res.get("certified", False))),
                })
    with (out_dir / "budget_pendigits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        writer.writeheader(); writer.writerows(budget_rows)

    deletion_rows: List[Dict] = []
    for epoch in range(1, 10):
        stale_del = pacer_del = pacer_x_del = naive_del = 0
        for p0 in queries[:24]:
            p = type(p0)(p0.qvec, p0.tenant_mask, p0.max_sensitivity, p0.region_mask, p0.dtype_mask, p0.provenance_mask, epoch, p0.regime, p0.source_id, p0.policy_anchor_id, p0.correlation)
            r_pacer = _method_dict(ds, index, k, candidate_budget)["PACER-A"](p)
            r_x = _method_dict(ds, index, k, ds.n)["PACER-X"](p)
            r_stale = _method_dict(ds, index, k, candidate_budget)["StaleNoRecheck"](p)
            r_naive = _method_dict(ds, index, k, candidate_budget)["NaiveNoPolicy"](p)
            _, d1, _ = ds.violations(r_pacer["ids"], p)
            _, d2, _ = ds.violations(r_x["ids"], p)
            _, d3, _ = ds.violations(r_stale["ids"], p)
            _, d4, _ = ds.violations(r_naive["ids"], p)
            pacer_del += d1; pacer_x_del += d2; stale_del += d3; naive_del += d4
        deletion_rows.append({
            "epoch": epoch,
            "queries": 24,
            "stale_residues_in_index": int(np.sum(ds.delete_epoch <= epoch)),
            "pacer_deleted_results": int(pacer_del),
            "pacer_certify_deleted_results": int(pacer_x_del),
            "stale_no_recheck_deleted_results": int(stale_del),
            "naive_deleted_results": int(naive_del),
        })
    with (out_dir / "deletion_pendigits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(deletion_rows[0].keys()))
        writer.writeheader(); writer.writerows(deletion_rows)

    failure_rows: List[Dict] = []
    for method in sorted(set(r["method"] for r in rows)):
        xs = [r for r in rows if r["method"] == method]
        for fm in ["none", "candidate_starvation", "ranking_truncation"]:
            failure_rows.append({"method": method, "failure_mode": fm, "fraction": float(np.mean([r["failure_mode"] == fm for r in xs]))})
    with (out_dir / "failure_modes_pendigits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
        writer.writeheader(); writer.writerows(failure_rows)

    summary = summarize_metrics(rows, budget_rows, deletion_rows, index.index_overhead())
    summary.update({
        "dataset": ds.metadata,
        "n": int(ds.n),
        "dim": int(ds.dim),
        "queries": int(len(queries)),
        "query_method_rows": int(len(rows)),
        "nlist": int(index.nlist),
        "slice_count": int(index.nlist),
        "slice_postings": int(ds.n),
        "k": int(k),
        "candidate_budget": int(candidate_budget),
    })
    with (out_dir / "summary_pendigits.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary



def write_combined_external_table(project_root: Path, pendigits_summary: Dict) -> None:
    figures = project_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    ext_path = project_root / "results" / "external_summary.json"
    if ext_path.exists():
        ext = json.load(ext_path.open())
        for ds_name, pretty in [("digits", "Digits"), ("breast-cancer", "Breast"), ("wine", "Wine"), ("diabetes", "Diabetes")]:
            if ds_name in ext.get("datasets", {}):
                d = ext["datasets"][ds_name]
                meta = d["metadata"]
                post = d["methods"]["PostFilter-IVF"]
                pacer = d["methods"]["PACER-A"]
                pc = d["methods"].get("PACER-C", pacer)
                px = d["methods"]["PACER-X"]
                rows.append((pretty, int(meta["n"]), post["secure_topk_exact"], pacer["secure_topk_exact"], pc["secure_topk_exact"], px["secure_topk_exact"]))
    post = pendigits_summary["overall"]["PostFilter-IVF"]
    pacer = pendigits_summary["overall"]["PACER-A"]
    pc = pendigits_summary["overall"].get("PACER-C", pacer)
    px = pendigits_summary["overall"]["PACER-X"]
    rows.append(("Pendigits", int(pendigits_summary["n"]), post["secure_topk_exact"], pacer["secure_topk_exact"], pc["secure_topk_exact"], px["secure_topk_exact"]))
    with (figures / "table_external.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Data & $n$ & Post ex. & A ex. & C ex. & X ex.\\\\\n")
        f.write("\\midrule\n")
        for name, n, pe, ae, ce, xe in rows:
            f.write(f"{name} & {n} & {pe:.3f} & {ae:.3f} & {ce:.3f} & {xe:.3f}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "data" / "external" / "pendigits"))
    ap.add_argument("--out", default=str(root / "results" / "pendigits"))
    ap.add_argument("--per-regime", type=int, default=8)
    ap.add_argument("--nlist", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2400)
    args = ap.parse_args()
    summary = run_external_suite(Path(args.out), Path(args.data), per_regime=args.per_regime, nlist=args.nlist, candidate_budget=args.budget)
    project_root = Path(args.out).resolve().parents[1] if Path(args.out).resolve().name == "pendigits" else root
    write_combined_external_table(project_root, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
