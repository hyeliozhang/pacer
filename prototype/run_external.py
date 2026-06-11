#!/usr/bin/env python3
"""Run PACER on small public, locally shipped sklearn datasets.

This script is intentionally a supplement to the synthetic benchmark. It does not
call network services, embedding APIs, FAISS, hnswlib, vector DB servers, or GPUs.
The vectors come from real public feature matrices bundled with scikit-learn; the
policy/deletion/provenance columns are generated deterministically from labels and
feature statistics so that the same PACER semantics can be exercised without
proprietary data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from trustvql import (
    INF_EPOCH,
    IVFIndex,
    PolicyVectorDataset,
    exact_secure,
    exact_ordered,
    failure_mode,
    generate_queries,
    recall_at_k,
    _method_dict,
    _normalize,
)


def _quantile_bins(x: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if bins <= 1:
        return np.zeros_like(x, dtype=np.int16)
    qs = np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(x, qs, right=True).astype(np.int16)


def _snapshot_epochs(n: int, seed: int, delete_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic insertion/deletion epochs for public feature matrices.

    The public datasets supply real vector geometry but no database lifecycle
    columns.  We therefore derive snapshot metadata deterministically so the
    same row-verifier contract is exercised outside the synthetic generator.
    Deleted rows always have delete_epoch > insert_epoch.
    """
    rng = np.random.default_rng(seed)
    insert_epoch = rng.choice(np.arange(0, 6), size=n, p=np.array([0.42, 0.20, 0.15, 0.10, 0.08, 0.05])).astype(np.int32)
    delete_epoch = np.full(n, INF_EPOCH, dtype=np.int32)
    deleted = rng.random(n) < float(delete_rate)
    if int(np.sum(deleted)):
        delete_epoch[deleted] = (insert_epoch[deleted] + rng.integers(1, 8, size=int(np.sum(deleted)))).astype(np.int32)
    return insert_epoch, delete_epoch


def _prov_from_fields(primary: np.ndarray, secondary: np.ndarray, prov_tags: int) -> np.ndarray:
    out = np.zeros(len(primary), dtype=np.int16)
    for i, (a, b) in enumerate(zip(primary, secondary)):
        m = 1 << (int(a) % prov_tags)
        m |= 1 << ((int(a) + int(b) + 1) % prov_tags)
        if (i + int(b)) % 7 == 0:
            m |= 1 << ((int(a) * 3 + int(b)) % prov_tags)
        out[i] = m
    return out


def make_digits(seed: int = 1001) -> PolicyVectorDataset:
    from sklearn.datasets import load_digits

    data = load_digits()
    X = np.asarray(data.data, dtype=np.float32)
    # Retain the real 64-dimensional handwritten-digit feature vectors and only
    # normalize for cosine similarity.
    vectors = _normalize(X + 1e-6).astype(np.float32)
    y = np.asarray(data.target, dtype=np.int16)
    n = X.shape[0]
    img = X.reshape(n, 8, 8)
    quadrant_mass = np.stack([
        img[:, :4, :4].sum(axis=(1, 2)),
        img[:, :4, 4:].sum(axis=(1, 2)),
        img[:, 4:, :4].sum(axis=(1, 2)),
        img[:, 4:, 4:].sum(axis=(1, 2)),
    ], axis=1)
    region = np.argmax(quadrant_mass, axis=1).astype(np.int16)
    stroke = (X > 3).sum(axis=1)
    dtype = _quantile_bins(stroke, 5)
    total = X.sum(axis=1)
    sensitivity = _quantile_bins(total, 4).astype(np.int8)
    provenance = _prov_from_fields(y, dtype, 8)
    insert_epoch, delete_epoch = _snapshot_epochs(n, seed + 11, 0.12)
    return PolicyVectorDataset(
        vectors=vectors,
        tenant=y.astype(np.int16),
        sensitivity=sensitivity,
        region=region.astype(np.int8),
        dtype=dtype.astype(np.int8),
        provenance=provenance,
        delete_epoch=delete_epoch,
        insert_epoch=insert_epoch,
        record_id=np.arange(n, dtype=np.int64),
        quality=np.asarray(total / max(float(total.max()), 1.0), dtype=np.float32),
        true_cluster=y.astype(np.int16),
        metadata={
            "dataset_name": "digits",
            "source": "sklearn.load_digits",
            "n": int(n),
            "dim": int(vectors.shape[1]),
            "tenants": 10,
            "regions": 4,
            "dtypes": 5,
            "prov_tags": 8,
            "delete_rate": 0.12,
            "tenant_cluster_correlation": 1.0,
            "seed": int(seed),
            "insert_epoch_min": int(insert_epoch.min()),
            "insert_epoch_max": int(insert_epoch.max()),
        },
    )


def make_breast(seed: int = 2001) -> PolicyVectorDataset:
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X0 = np.asarray(data.data, dtype=np.float32)
    # Standardize positive clinical features and normalize for cosine search.
    X = StandardScaler().fit_transform(X0).astype(np.float32)
    vectors = _normalize(X).astype(np.float32)
    y = np.asarray(data.target, dtype=np.int16)
    n = X.shape[0]
    radius = _quantile_bins(X0[:, 0], 6)
    texture = _quantile_bins(X0[:, 1], 5)
    perimeter = _quantile_bins(X0[:, 2], 4)
    region = (radius % 6).astype(np.int16)
    dtype = (texture % 5).astype(np.int16)
    sensitivity = np.minimum(3, perimeter).astype(np.int8)
    provenance = _prov_from_fields(y + 2 * region, dtype, 8)
    insert_epoch, delete_epoch = _snapshot_epochs(n, seed + 11, 0.10)
    return PolicyVectorDataset(
        vectors=vectors,
        tenant=y.astype(np.int16),
        sensitivity=sensitivity,
        region=region.astype(np.int8),
        dtype=dtype.astype(np.int8),
        provenance=provenance,
        delete_epoch=delete_epoch,
        insert_epoch=insert_epoch,
        record_id=np.arange(n, dtype=np.int64),
        quality=np.asarray((X0[:, 0] - X0[:, 0].min()) / max(float(np.ptp(X0[:, 0])), 1e-9), dtype=np.float32),
        true_cluster=y.astype(np.int16),
        metadata={
            "dataset_name": "breast-cancer",
            "source": "sklearn.load_breast_cancer",
            "n": int(n),
            "dim": int(vectors.shape[1]),
            "tenants": 2,
            "regions": 6,
            "dtypes": 5,
            "prov_tags": 8,
            "delete_rate": 0.10,
            "tenant_cluster_correlation": 1.0,
            "seed": int(seed),
            "insert_epoch_min": int(insert_epoch.min()),
            "insert_epoch_max": int(insert_epoch.max()),
        },
    )



def make_wine(seed: int = 3001) -> PolicyVectorDataset:
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler

    data = load_wine()
    X0 = np.asarray(data.data, dtype=np.float32)
    X = StandardScaler().fit_transform(X0).astype(np.float32)
    vectors = _normalize(X).astype(np.float32)
    y = np.asarray(data.target, dtype=np.int16)
    n = X.shape[0]
    alcohol = _quantile_bins(X0[:, 0], 5)
    flav = _quantile_bins(X0[:, 6], 6)
    color = _quantile_bins(X0[:, 9], 4)
    proline = _quantile_bins(X0[:, 12], 5)
    region = (flav % 6).astype(np.int16)
    dtype = (alcohol % 5).astype(np.int16)
    sensitivity = np.minimum(3, color).astype(np.int8)
    provenance = _prov_from_fields(y + region, proline, 8)
    insert_epoch, delete_epoch = _snapshot_epochs(n, seed + 11, 0.10)
    return PolicyVectorDataset(
        vectors=vectors,
        tenant=y.astype(np.int16),
        sensitivity=sensitivity,
        region=region.astype(np.int8),
        dtype=dtype.astype(np.int8),
        provenance=provenance,
        delete_epoch=delete_epoch,
        insert_epoch=insert_epoch,
        record_id=np.arange(n, dtype=np.int64),
        quality=np.asarray((X0[:, 12] - X0[:, 12].min()) / max(float(np.ptp(X0[:, 12])), 1e-9), dtype=np.float32),
        true_cluster=y.astype(np.int16),
        metadata={
            "dataset_name": "wine",
            "source": "sklearn.load_wine",
            "n": int(n),
            "dim": int(vectors.shape[1]),
            "tenants": 3,
            "regions": 6,
            "dtypes": 5,
            "prov_tags": 8,
            "delete_rate": 0.10,
            "tenant_cluster_correlation": 1.0,
            "seed": int(seed),
            "insert_epoch_min": int(insert_epoch.min()),
            "insert_epoch_max": int(insert_epoch.max()),
        },
    )


def make_diabetes(seed: int = 3501) -> PolicyVectorDataset:
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler

    data = load_diabetes()
    X0 = np.asarray(data.data, dtype=np.float32)
    X = StandardScaler().fit_transform(X0).astype(np.float32)
    vectors = _normalize(X).astype(np.float32)
    y0 = np.asarray(data.target, dtype=np.float32)
    y = _quantile_bins(y0, 6).astype(np.int16)
    n = X.shape[0]
    bmi = _quantile_bins(X0[:, 2], 6)
    bp = _quantile_bins(X0[:, 3], 5)
    s5 = _quantile_bins(X0[:, 8], 4)
    s6 = _quantile_bins(X0[:, 9], 5)
    region = (bmi % 6).astype(np.int16)
    dtype = (bp % 5).astype(np.int16)
    sensitivity = np.minimum(3, s5).astype(np.int8)
    provenance = _prov_from_fields(y + region, s6, 8)
    insert_epoch, delete_epoch = _snapshot_epochs(n, seed + 11, 0.10)
    return PolicyVectorDataset(
        vectors=vectors,
        tenant=y.astype(np.int16),
        sensitivity=sensitivity,
        region=region.astype(np.int8),
        dtype=dtype.astype(np.int8),
        provenance=provenance,
        delete_epoch=delete_epoch,
        insert_epoch=insert_epoch,
        record_id=np.arange(n, dtype=np.int64),
        quality=np.asarray((y0 - y0.min()) / max(float(np.ptp(y0)), 1e-9), dtype=np.float32),
        true_cluster=y.astype(np.int16),
        metadata={
            "dataset_name": "diabetes",
            "source": "sklearn.load_diabetes",
            "n": int(n),
            "dim": int(vectors.shape[1]),
            "tenants": 6,
            "regions": 6,
            "dtypes": 5,
            "prov_tags": 8,
            "delete_rate": 0.10,
            "tenant_cluster_correlation": 1.0,
            "seed": int(seed),
            "insert_epoch_min": int(insert_epoch.min()),
            "insert_epoch_max": int(insert_epoch.max()),
        },
    )

def evaluate_dataset(ds: PolicyVectorDataset, out: Path, name: str, nlist: int, budget: int, per_regime: int, seed: int, k: int = 10) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    out.mkdir(parents=True, exist_ok=True)
    index = IVFIndex(ds, nlist=nlist, seed=seed + 1)
    queries = generate_queries(ds, per_regime=per_regime, seed=seed + 2, correlation="positive")
    methods = _method_dict(ds, index, k, budget)
    keep = ["ExactSecure", "PostFilter-IVF", "AdaptivePostFilter-IVF", "TenantPartition-IVF", "PreFilter-IVF", "PACER-A", "PACER-C", "PACER-X", "StaleNoRecheck", "NaiveNoPolicy"]
    golds = [exact_secure(ds, p, k=k) for p in queries]
    rows: List[Dict] = []
    for qi, (p, gold) in enumerate(zip(queries, golds)):
        allowed = int(ds.policy_mask(p, include_deletions=False).sum())
        for method in keep:
            res = methods[method](p)
            policy_v, del_v, tenant_v = ds.violations(res["ids"], p)
            exact = int(exact_ordered(gold["ids"], res["ids"], k=k))
            rows.append({
                "dataset": name,
                "query_id": qi,
                "regime": p.regime,
                "method": method,
                "n": ds.n,
                "dim": ds.dim,
                "k": k,
                "allowed_count": allowed,
                "selectivity": allowed / max(ds.n, 1),
                "returned": int(len(res["ids"])),
                "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=k),
                "secure_topk_exact": exact,
                "failure_mode": failure_mode(gold["ids"], res["ids"], k=k),
                "policy_violations": policy_v,
                "deleted_violations": del_v,
                "tenant_violations": tenant_v,
                "latency_ms": float(res["latency_ms"]),
                "raw_candidates": int(res.get("raw_candidates", res.get("candidates", 0))),
                "candidates": int(res["candidates"]),
                "distance_evals": int(res.get("distance_evals", res.get("candidates", 0))),
                "policy_checks": int(res.get("policy_checks", res.get("raw_candidates", 0))),
                "summary_checks": int(res.get("summary_checks", 0)),
                "visited_cluster_count": int(res.get("visited_cluster_count", 0)),
                "certified": int(bool(res.get("certified", False))),
                "certificate_false_positive": int(bool(res.get("certified", False)) and not exact),
            })
    with (out / f"external_metrics_{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    agg: Dict[str, Dict[str, float]] = {}
    for method in keep:
        xs = [r for r in rows if r["method"] == method]
        if not xs:
            continue
        agg[method] = {
            "recall_at_k": float(np.mean([float(r["recall_at_k"]) for r in xs])),
            "secure_topk_exact": float(np.mean([float(r["secure_topk_exact"]) for r in xs])),
            "violations_per_query": float(np.mean([int(r["policy_violations"]) + int(r["deleted_violations"]) + int(r["tenant_violations"]) for r in xs])),
            "median_latency_ms": float(np.median([float(r["latency_ms"]) for r in xs])),
            "raw_candidates": float(np.mean([int(r["raw_candidates"]) for r in xs])),
            "certified_fraction": float(np.mean([int(r["certified"]) for r in xs])),
            "certificate_false_positive_total": int(np.sum([int(r["certificate_false_positive"]) for r in xs])),
        }
    return rows, agg


def write_external_table(figures: Path, summary: Dict) -> None:
    rows = []
    for ds_name, pretty in [("digits", "Digits"), ("breast-cancer", "Breast"), ("wine", "Wine"), ("diabetes", "Diabetes")]:
        d = summary["datasets"][ds_name]
        meta = d["metadata"]
        post = d["methods"]["PostFilter-IVF"]
        pacer = d["methods"]["PACER-A"]
        px = d["methods"]["PACER-X"]
        rows.append((pretty, int(meta["n"]), post["recall_at_k"], post["secure_topk_exact"], pacer["recall_at_k"], pacer["secure_topk_exact"], px["secure_topk_exact"]))
    with (figures / "table_external.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Data & $n$ & Post rec. & Post ex. & PACER rec. & PACER ex. & X ex.\\\\\n")
        f.write("\\midrule\n")
        for name, n, pr, pe, rr, re, xe in rows:
            f.write(f"{name} & {n} & {pr:.3f} & {pe:.3f} & {rr:.3f} & {re:.3f} & {xe:.3f}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run real-data PACER external-validity supplement")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None, help="output figure directory (default: repository figures/)")
    ap.add_argument("--per-regime", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    figs = Path(args.figures) if args.figures else Path(__file__).resolve().parents[1] / "figures"; figs.mkdir(parents=True, exist_ok=True)

    configs = [
        ("digits", make_digits(), 28, 720, 3201),
        ("breast-cancer", make_breast(), 18, 240, 4201),
        ("wine", make_wine(), 12, 180, 5201),
        ("diabetes", make_diabetes(), 16, 220, 6201),
    ]
    all_rows: List[Dict] = []
    summary = {"description": "public sklearn feature-matrix supplement; policy/deletion/provenance fields are deterministic database metadata derived from labels and feature statistics", "datasets": {}}
    for name, ds, nlist, budget, seed in configs:
        rows, agg = evaluate_dataset(ds, out, name, nlist=nlist, budget=budget, per_regime=args.per_regime, seed=seed)
        all_rows.extend(rows)
        summary["datasets"][name] = {"metadata": ds.metadata, "queries": int(args.per_regime * 4), "methods": agg, "nlist": nlist, "candidate_budget": budget}

    with (out / "external_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader(); writer.writerows(all_rows)
    with (out / "external_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_external_table(figs, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
