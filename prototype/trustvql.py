"""PACER: CPU-only prototype for policy-constrained vector top-k experiments.

The prototype deliberately avoids FAISS, hnswlib, commercial embedding APIs,
LLM calls, GPUs, and external vector-store services. It implements synthetic
vector data generation, policy/deletion/provenance predicates, exact secure
semantics, IVF-style candidate generation, adaptive post-filtering baselines,
conservative cluster summaries, bounded PACER search, certifying exact fallback,
audit transcripts, and stress workloads using only Python and NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import csv
import json
import math
import os
import sys
import time

import numpy as np

INF_EPOCH = 10**9


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _bitmask(vals: Iterable[int]) -> int:
    m = 0
    for v in vals:
        m |= 1 << int(v)
    return int(m)


def _mask_membership(vals: np.ndarray, mask: int) -> np.ndarray:
    """Return membership in an arbitrary-width Python integer bitmask.

    NumPy int64 left shifts silently overflow for values >= 63.  The policy
    model deliberately permits high-cardinality tenant/type/region domains, so
    all row-level membership tests use Python integer shifts.
    """
    arr = np.asarray(vals)
    m = int(mask)
    return np.fromiter((((m >> int(v)) & 1) != 0 for v in arr), dtype=bool, count=arr.size)


@dataclass(frozen=True)
class QueryPolicy:
    qvec: np.ndarray
    tenant_mask: int
    max_sensitivity: int
    region_mask: int
    dtype_mask: int
    provenance_mask: int
    epoch: int
    regime: str
    source_id: int
    policy_anchor_id: int = -1
    correlation: str = "positive"


@dataclass
class PolicyVectorDataset:
    vectors: np.ndarray
    tenant: np.ndarray
    sensitivity: np.ndarray
    region: np.ndarray
    dtype: np.ndarray
    provenance: np.ndarray
    delete_epoch: np.ndarray
    record_id: np.ndarray
    quality: np.ndarray
    true_cluster: np.ndarray
    metadata: Dict[str, float]
    insert_epoch: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def insert_epochs(self) -> np.ndarray:
        if self.insert_epoch is None:
            return np.zeros(self.n, dtype=np.int32)
        return self.insert_epoch.astype(np.int32, copy=False)

    def active_mask(self, epoch: int) -> np.ndarray:
        e = int(epoch)
        return (self.insert_epochs() <= e) & (self.delete_epoch > e)

    def policy_mask_ids(self, ids: np.ndarray, p: QueryPolicy, include_deletions: bool = False) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=bool)
        tenant_ok = _mask_membership(self.tenant[ids], p.tenant_mask)
        sens_ok = self.sensitivity[ids] <= int(p.max_sensitivity)
        region_ok = _mask_membership(self.region[ids], p.region_mask)
        dtype_ok = _mask_membership(self.dtype[ids], p.dtype_mask)
        prov_ok = (self.provenance[ids].astype(np.int64) & int(p.provenance_mask)) != 0
        mask = tenant_ok & sens_ok & region_ok & dtype_ok & prov_ok
        if not include_deletions:
            e = int(p.epoch)
            mask = mask & (self.insert_epochs()[ids] <= e) & (self.delete_epoch[ids] > e)
        return mask

    def policy_mask(self, p: QueryPolicy, include_deletions: bool = False) -> np.ndarray:
        ids = np.arange(self.n, dtype=np.int64)
        return self.policy_mask_ids(ids, p, include_deletions=include_deletions)

    def violations(self, ids: Iterable[int], p: QueryPolicy) -> Tuple[int, int, int]:
        """Return disjoint violation counters: scalar/provenance, epoch, tenant.

        Earlier internal versions reported ``policy_violations`` using the full
        visibility predicate except deletion, which already included tenant.
        The public tables also report tenant failures separately, so this method
        now makes the counters disjoint: ``policy_violations`` means non-tenant,
        non-epoch predicate failures (sensitivity, region, dtype, provenance),
        ``deleted_violations`` means insertion/deletion epoch failures, and
        ``tenant_violations`` means tenant-mask failures.  Safe methods remain
        zero on all three counters; unsafe controls are easier to diagnose.
        """
        ids = np.asarray(list(ids), dtype=np.int64)
        if ids.size == 0:
            return 0, 0, 0
        e = int(p.epoch)
        inserted = self.insert_epochs()[ids] <= e
        not_deleted = self.delete_epoch[ids] > e
        active = inserted & not_deleted
        tenant_ok = _mask_membership(self.tenant[ids], p.tenant_mask)
        sens_ok = self.sensitivity[ids] <= int(p.max_sensitivity)
        region_ok = _mask_membership(self.region[ids], p.region_mask)
        dtype_ok = _mask_membership(self.dtype[ids], p.dtype_mask)
        prov_ok = (self.provenance[ids].astype(np.int64) & int(p.provenance_mask)) != 0
        residual_policy_ok = sens_ok & region_ok & dtype_ok & prov_ok
        return int(np.sum(~residual_policy_ok)), int(np.sum(~active)), int(np.sum(~tenant_ok))

    def row_facts(self, ids: Iterable[int], p: QueryPolicy) -> List[Dict]:
        rows: List[Dict] = []
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        visible_flags = self.policy_mask_ids(ids_arr, p, include_deletions=False) if ids_arr.size else np.empty(0, dtype=bool)
        for pos, rid0 in enumerate(ids_arr):
            rid = int(rid0)
            rows.append({
                "id": rid,
                "tenant": int(self.tenant[rid]),
                "sensitivity": int(self.sensitivity[rid]),
                "region": int(self.region[rid]),
                "dtype": int(self.dtype[rid]),
                "provenance": int(self.provenance[rid]),
                "insert_epoch": int(self.insert_epochs()[rid]),
                "delete_epoch": int(self.delete_epoch[rid]),
                "epoch": int(p.epoch),
                "visible": bool(visible_flags[pos]),
                "tenant_ok": bool(((1 << int(self.tenant[rid])) & int(p.tenant_mask)) != 0),
                "region_ok": bool(((1 << int(self.region[rid])) & int(p.region_mask)) != 0),
                "dtype_ok": bool(((1 << int(self.dtype[rid])) & int(p.dtype_mask)) != 0),
                "provenance_ok": bool((int(self.provenance[rid]) & int(p.provenance_mask)) != 0),
                "sensitivity_ok": bool(int(self.sensitivity[rid]) <= int(p.max_sensitivity)),
                "insert_ok": bool(self.insert_epochs()[rid] <= int(p.epoch)),
                "live_ok": bool(self.delete_epoch[rid] > int(p.epoch)),
            })
        return rows

    def explanations_valid(self, ids: Iterable[int], p: QueryPolicy) -> Tuple[int, int]:
        facts = self.row_facts(ids, p)
        valid = [f for f in facts if f["visible"] and f["tenant_ok"] and f["region_ok"] and f["dtype_ok"] and f["provenance_ok"] and f["sensitivity_ok"] and f.get("insert_ok", True) and f["live_ok"]]
        return int(len(valid)), int(len(facts))


def generate_dataset(
    n: int = 15000,
    dim: int = 64,
    true_clusters: int = 48,
    tenants: int = 24,
    regions: int = 6,
    dtypes: int = 5,
    prov_tags: int = 8,
    delete_rate: float = 0.12,
    tenant_cluster_correlation: float = 0.82,
    seed: int = 7,
) -> PolicyVectorDataset:
    """Create an attributed vector relation with controllable policy/vector correlation."""
    rng = np.random.default_rng(seed)
    corr = float(np.clip(tenant_cluster_correlation, 0.0, 1.0))
    centers = _normalize(rng.normal(size=(true_clusters, dim)).astype(np.float32))
    probs = rng.power(2.0, size=true_clusters)
    probs = probs / probs.sum()
    cid = rng.choice(true_clusters, size=n, p=probs).astype(np.int16)
    vectors = _normalize(centers[cid] + rng.normal(scale=0.20, size=(n, dim)).astype(np.float32)).astype(np.float32)

    tenant_base = (cid + rng.integers(0, 5, size=n)) % tenants
    tenant_noise = rng.integers(0, tenants, size=n)
    tenant = np.where(rng.random(n) < corr, tenant_base, tenant_noise).astype(np.int16)

    sens_cluster = np.minimum(3, (cid % 6) // 2 + rng.choice([0, 1], size=n, p=[0.78, 0.22]))
    sens_noise = rng.choice(np.arange(4), size=n, p=np.array([0.36, 0.34, 0.22, 0.08]))
    sensitivity = np.where(rng.random(n) < corr, sens_cluster, sens_noise).astype(np.int8)

    region_cluster = ((cid + rng.integers(0, 2, size=n)) % regions)
    region_noise = rng.integers(0, regions, size=n)
    region = np.where(rng.random(n) < corr, region_cluster, region_noise).astype(np.int8)

    dtype_cluster = ((cid // 2 + rng.integers(0, 2, size=n)) % dtypes)
    dtype_noise = rng.integers(0, dtypes, size=n)
    dtype = np.where(rng.random(n) < corr, dtype_cluster, dtype_noise).astype(np.int8)

    provenance = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cluster_tag = int((cid[i] + rng.integers(0, 3)) % prov_tags)
        random_tag = int(rng.integers(0, prov_tags))
        tag = cluster_tag if rng.random() < corr else random_tag
        m = 1 << tag
        if rng.random() < 0.30:
            m |= 1 << int(rng.integers(0, prov_tags))
        if rng.random() < 0.07:
            m |= 1 << int(rng.integers(0, prov_tags))
        provenance[i] = m

    insert_epoch = rng.choice(np.arange(0, 6), size=n, p=np.array([0.46, 0.18, 0.14, 0.10, 0.07, 0.05])).astype(np.int32)
    delete_epoch = np.full(n, INF_EPOCH, dtype=np.int32)
    deleted = rng.random(n) < float(delete_rate)
    if int(np.sum(deleted)):
        # A deleted row must first exist.  The tombstone epoch is strictly after
        # its insertion epoch, which lets experiments exercise snapshot freshness
        # without manufacturing impossible temporal histories.
        delete_epoch[deleted] = (insert_epoch[deleted] + rng.integers(1, 8, size=int(np.sum(deleted)))).astype(np.int32)
    quality = rng.random(n).astype(np.float32)

    return PolicyVectorDataset(
        vectors=vectors,
        tenant=tenant,
        sensitivity=sensitivity,
        region=region,
        dtype=dtype,
        provenance=provenance,
        delete_epoch=delete_epoch,
        record_id=np.arange(n, dtype=np.int64),
        quality=quality,
        true_cluster=cid,
        metadata={
            "true_clusters": int(true_clusters),
            "tenants": int(tenants),
            "regions": int(regions),
            "dtypes": int(dtypes),
            "prov_tags": int(prov_tags),
            "delete_rate": float(delete_rate),
            "tenant_cluster_correlation": float(corr),
            "seed": int(seed),
            "insert_epoch_min": int(insert_epoch.min()),
            "insert_epoch_max": int(insert_epoch.max()),
        },
        insert_epoch=insert_epoch,
    )


def _anchor_for_correlation(ds: PolicyVectorDataset, qvec: np.ndarray, src: int, rng: np.random.Generator, correlation: str) -> int:
    if correlation == "positive":
        return int(src)
    if correlation == "independent":
        return int(rng.integers(0, ds.n))
    if correlation == "negative":
        pool = rng.choice(ds.n, size=min(768, ds.n), replace=False)
        scores = ds.vectors[pool] @ qvec
        return int(pool[int(np.argmin(scores))])
    raise ValueError(f"unknown correlation mode: {correlation}")


def generate_queries(ds: PolicyVectorDataset, per_regime: int = 40, seed: int = 99, correlation: str = "positive") -> List[QueryPolicy]:
    rng = np.random.default_rng(seed)
    regimes = ["broad", "medium", "narrow", "cold"]
    queries: List[QueryPolicy] = []
    all_tenants = np.arange(int(ds.metadata["tenants"]))
    all_regions = np.arange(int(ds.metadata["regions"]))
    all_dtypes = np.arange(int(ds.metadata["dtypes"]))
    all_prov = np.arange(int(ds.metadata["prov_tags"]))
    regime_counts = {r: 0 for r in regimes}
    attempts = 0
    while min(regime_counts.values()) < per_regime:
        attempts += 1
        if attempts > max(10000, per_regime * len(regimes) * 2000):
            raise RuntimeError(f"could not generate enough satisfiable policies for correlation={correlation}")
        regime = regimes[attempts % len(regimes)]
        if regime_counts[regime] >= per_regime:
            continue
        src = int(rng.integers(0, ds.n))
        qvec = _normalize(ds.vectors[src] + rng.normal(scale=0.16, size=ds.dim).astype(np.float32))[0]
        anchor = _anchor_for_correlation(ds, qvec, src, rng, correlation)
        tenant = int(ds.tenant[anchor])
        reg = int(ds.region[anchor])
        typ = int(ds.dtype[anchor])
        prov_bits = int(ds.provenance[anchor])
        prov_vals = [int(i) for i in all_prov if prov_bits & (1 << int(i))]
        epoch = int(rng.integers(4, 9))

        if regime == "broad":
            tenant_set = list(rng.choice(all_tenants, size=min(12, len(all_tenants)), replace=False))
            if tenant not in tenant_set:
                tenant_set[0] = tenant
            region_set = list(all_regions)
            dtype_set = list(all_dtypes)
            prov_set = list(all_prov)
            max_s = 3
        elif regime == "medium":
            tenant_set = list(rng.choice(all_tenants, size=min(4, len(all_tenants)), replace=False))
            if tenant not in tenant_set:
                tenant_set[0] = tenant
            region_set = list(rng.choice(all_regions, size=min(3, len(all_regions)), replace=False))
            if reg not in region_set:
                region_set[0] = reg
            dtype_set = list(rng.choice(all_dtypes, size=min(3, len(all_dtypes)), replace=False))
            if typ not in dtype_set:
                dtype_set[0] = typ
            prov_set = list(rng.choice(all_prov, size=min(4, len(all_prov)), replace=False))
            if prov_vals and prov_vals[0] not in prov_set:
                prov_set[0] = prov_vals[0]
            max_s = max(int(ds.sensitivity[anchor]), 2)
        elif regime == "narrow":
            tenant_set = [tenant]
            region_set = [reg]
            if rng.random() < 0.35:
                x = int(rng.integers(0, int(ds.metadata["regions"])))
                if x not in region_set:
                    region_set.append(x)
            dtype_set = [typ]
            prov_set = [prov_vals[0] if prov_vals else int(rng.integers(0, int(ds.metadata["prov_tags"])))]
            if rng.random() < 0.45:
                x = int(rng.integers(0, int(ds.metadata["prov_tags"])))
                if x not in prov_set:
                    prov_set.append(x)
            max_s = max(int(ds.sensitivity[anchor]), 1)
        else:  # cold
            tenant_set = [tenant]
            region_set = [reg]
            dtype_set = [typ]
            prov_set = [prov_vals[-1] if prov_vals else int(rng.integers(0, int(ds.metadata["prov_tags"])))]
            max_s = int(ds.sensitivity[anchor])

        p = QueryPolicy(
            qvec=qvec.astype(np.float32),
            tenant_mask=_bitmask(tenant_set),
            max_sensitivity=int(max_s),
            region_mask=_bitmask(region_set),
            dtype_mask=_bitmask(dtype_set),
            provenance_mask=_bitmask(prov_set),
            epoch=epoch,
            regime=regime,
            source_id=src,
            policy_anchor_id=anchor,
            correlation=correlation,
        )
        min_allowed = {"broad": 20, "medium": 10, "narrow": 3, "cold": 1}[regime]
        if int(ds.policy_mask(p, include_deletions=False).sum()) >= min_allowed:
            queries.append(p)
            regime_counts[regime] += 1
    out: List[QueryPolicy] = []
    for i in range(per_regime):
        for reg in regimes:
            out.append([q for q in queries if q.regime == reg][i])
    return out


class IVFIndex:
    def __init__(self, ds: PolicyVectorDataset, nlist: int = 64, seed: int = 13, iters: int = 3):
        self.ds = ds
        self.nlist = int(nlist)
        self.seed = int(seed)
        self.iters = int(iters)
        self.centroids: np.ndarray
        self.assignments: np.ndarray
        self.cluster_ids: List[np.ndarray]
        self.radii: np.ndarray
        self.angular_radii: np.ndarray
        self.signature_sets: List[set]
        self.slice_postings: List[Dict[Tuple[int, int, int, int, int], np.ndarray]]
        self.cluster_bitsets: List[int]
        self.tenant_value_bitsets: Dict[int, int]
        self.region_value_bitsets: Dict[int, int]
        self.dtype_value_bitsets: Dict[int, int]
        self.prov_value_bitsets: Dict[int, int]
        self.sensitivity_leq_bitsets: List[int]
        self.tenant_bits: np.ndarray
        self.region_bits: np.ndarray
        self.dtype_bits: np.ndarray
        self.prov_bits: np.ndarray
        self.min_sensitivity: np.ndarray
        self.max_delete_epoch: np.ndarray
        self.min_insert_epoch: np.ndarray
        self._build()

    def _build(self) -> None:
        """Build a CPU-only IVF coarse quantizer from vectors.

        Unlike the first draft, the hardened artifact does not reuse the
        generator's latent clusters. It trains a small spherical k-means coarse
        partition with NumPy only, then stores conservative per-cluster policy,
        tombstone, and provenance summaries. The summaries may over-approximate
        visible rows but are not allowed to false-negatively prune a cluster that
        contains a row satisfying the policy at the query epoch.
        """
        x = self.ds.vectors
        n, dim = x.shape
        nlist = min(self.nlist, n)
        rng = np.random.default_rng(self.seed)
        # Random distinct initialization is deterministic under the seed and
        # avoids depending on hidden synthetic labels. A few iterations are
        # enough for the query-processing study; the exact oracle remains the
        # ground truth for every metric.
        init = rng.choice(n, size=nlist, replace=False)
        c = x[init].copy().astype(np.float32)
        c = _normalize(c).astype(np.float32)
        assign = np.zeros(n, dtype=np.int32)
        chunk = 8192
        for _ in range(max(1, int(self.iters))):
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                assign[lo:hi] = np.argmax(x[lo:hi] @ c.T, axis=1).astype(np.int32)
            for j in range(nlist):
                ids = np.flatnonzero(assign == j)
                if ids.size:
                    c[j] = x[ids].mean(axis=0)
                else:
                    c[j] = x[int(rng.integers(0, n))]
            c = _normalize(c).astype(np.float32)
        # Final assignment to trained centroids.
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            assign[lo:hi] = np.argmax(x[lo:hi] @ c.T, axis=1).astype(np.int32)
        self.nlist = nlist
        self.assignments = assign
        self.cluster_ids = [np.flatnonzero(assign == j).astype(np.int64) for j in range(nlist)]
        self.centroids = c.astype(np.float32)
        self.radii = np.zeros(nlist, dtype=np.float32)
        self.angular_radii = np.zeros(nlist, dtype=np.float32)
        self.signature_sets = [set() for _ in range(nlist)]
        self.slice_postings = [dict() for _ in range(nlist)]
        self.tenant_bits = np.zeros(nlist, dtype=object)
        self.region_bits = np.zeros(nlist, dtype=object)
        self.dtype_bits = np.zeros(nlist, dtype=object)
        self.prov_bits = np.zeros(nlist, dtype=object)
        self.min_sensitivity = np.full(nlist, 1000, dtype=np.int16)
        self.max_delete_epoch = np.zeros(nlist, dtype=np.int32)
        self.min_insert_epoch = np.full(nlist, INF_EPOCH, dtype=np.int32)
        ins = self.ds.insert_epochs()
        for j, ids in enumerate(self.cluster_ids):
            if not ids.size:
                continue
            self.radii[j] = float(np.max(np.linalg.norm(x[ids] - self.centroids[j], axis=1)))
            dots_to_centroid = np.clip(x[ids] @ self.centroids[j], -1.0, 1.0)
            self.angular_radii[j] = float(np.arccos(float(np.min(dots_to_centroid))))
            tb = rb = db = pb = 0
            sigs = set()
            postings: Dict[Tuple[int, int, int, int, int], List[int]] = {}
            for rid in ids:
                tenant_i = int(self.ds.tenant[rid])
                region_i = int(self.ds.region[rid])
                dtype_i = int(self.ds.dtype[rid])
                sens_i = int(self.ds.sensitivity[rid])
                prov_i = int(self.ds.provenance[rid])
                bits = [b for b in range(64) if (prov_i & (1 << b)) != 0]
                for b in bits:
                    key = (tenant_i, region_i, dtype_i, b, sens_i)
                    sigs.add(key)
                    postings.setdefault(key, []).append(int(rid))
            self.signature_sets[j] = sigs
            self.slice_postings[j] = {key: np.asarray(vals, dtype=np.int64) for key, vals in postings.items()}
            for v in np.unique(self.ds.tenant[ids]):
                tb |= 1 << int(v)
            for v in np.unique(self.ds.region[ids]):
                rb |= 1 << int(v)
            for v in np.unique(self.ds.dtype[ids]):
                db |= 1 << int(v)
            for v in np.unique(self.ds.provenance[ids]):
                pb |= int(v)
            self.tenant_bits[j] = tb
            self.region_bits[j] = rb
            self.dtype_bits[j] = db
            self.prov_bits[j] = pb
            self.min_sensitivity[j] = int(np.min(self.ds.sensitivity[ids]))
            self.max_delete_epoch[j] = int(np.max(self.ds.delete_epoch[ids]))
            self.min_insert_epoch[j] = int(np.min(ins[ids]))

        self.cluster_bitsets = [self._ids_to_bitset(ids) for ids in self.cluster_ids]
        self.tenant_value_bitsets = self._value_bitsets(self.ds.tenant)
        self.region_value_bitsets = self._value_bitsets(self.ds.region)
        self.dtype_value_bitsets = self._value_bitsets(self.ds.dtype)
        self.prov_value_bitsets = self._provenance_bitsets(self.ds.provenance, int(self.ds.metadata.get("prov_tags", 64)))
        max_sens = int(np.max(self.ds.sensitivity)) if self.ds.n else 0
        self.sensitivity_leq_bitsets = []
        for t in range(max(4, max_sens + 1)):
            self.sensitivity_leq_bitsets.append(self._ids_to_bitset(np.flatnonzero(self.ds.sensitivity <= t)))

    @staticmethod
    def _ids_to_bitset(ids: Iterable[int]) -> int:
        bits = 0
        for rid in ids:
            bits |= 1 << int(rid)
        return int(bits)

    @staticmethod
    def _bitset_to_ids(bits: int) -> np.ndarray:
        bits = int(bits)
        out: List[int] = []
        while bits:
            lsb = bits & -bits
            out.append(int(lsb.bit_length() - 1))
            bits ^= lsb
        return np.asarray(out, dtype=np.int64)

    @classmethod
    def _value_bitsets(cls, arr: np.ndarray) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for v in np.unique(arr):
            out[int(v)] = cls._ids_to_bitset(np.flatnonzero(arr == v))
        return out

    @classmethod
    def _provenance_bitsets(cls, arr: np.ndarray, prov_tags: int) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for b in range(int(prov_tags)):
            out[b] = cls._ids_to_bitset(np.flatnonzero((arr.astype(np.int64) & (1 << b)) != 0))
        return out

    @staticmethod
    def _mask_values(mask: int, limit: int = 64) -> List[int]:
        m = int(mask)
        return [i for i in range(limit) if (m & (1 << i)) != 0]

    def cluster_upper_bounds(self, qvec: np.ndarray, mode: str = "cone") -> np.ndarray:
        dot = np.clip(self.centroids @ qvec, -1.0, 1.0)
        if mode == "cone":
            theta = np.arccos(dot)
            ub = np.cos(np.maximum(0.0, theta - self.angular_radii))
            return np.minimum(1.0, ub).astype(np.float32)
        return (dot + self.radii).astype(np.float32)

    def may_satisfy(self, c: int, p: QueryPolicy, include_deletions: bool = False, tenant_only: bool = False, use_signatures: bool = True) -> bool:
        if not self.cluster_ids[c].size:
            return False
        if (int(self.tenant_bits[c]) & int(p.tenant_mask)) == 0:
            return False
        if tenant_only:
            return True
        if (int(self.region_bits[c]) & int(p.region_mask)) == 0:
            return False
        if (int(self.dtype_bits[c]) & int(p.dtype_mask)) == 0:
            return False
        if (int(self.prov_bits[c]) & int(p.provenance_mask)) == 0:
            return False
        if int(self.min_sensitivity[c]) > int(p.max_sensitivity):
            return False
        if use_signatures and self.signature_sets[c]:
            tenants = self._mask_values(int(p.tenant_mask), int(self.ds.metadata.get("tenants", 64)))
            regions = self._mask_values(int(p.region_mask), int(self.ds.metadata.get("regions", 64)))
            dtypes = self._mask_values(int(p.dtype_mask), int(self.ds.metadata.get("dtypes", 64)))
            provs = self._mask_values(int(p.provenance_mask), int(self.ds.metadata.get("prov_tags", 64)))
            sigset = self.signature_sets[c]
            possible = False
            # This co-occurrence summary is still conservative: it only says that
            # at least one row-level combination of tenant, region, type,
            # provenance bit, and sensitivity bucket exists in the unit. Epochs
            # remain a separate conservative interval and final visibility is
            # decided by the authoritative row verifier.
            for tt in tenants:
                for rr in regions:
                    for dd in dtypes:
                        for pp in provs:
                            for ss in range(int(p.max_sensitivity) + 1):
                                if (tt, rr, dd, pp, ss) in sigset:
                                    possible = True
                                    break
                            if possible: break
                        if possible: break
                    if possible: break
                if possible: break
            if not possible:
                return False
        if not include_deletions:
            e = int(p.epoch)
            if int(self.min_insert_epoch[c]) > e:
                return False
            if int(self.max_delete_epoch[c]) <= e:
                return False
        return True

    @staticmethod
    def _or_bitsets(values: Iterable[int], mapping: Dict[int, int]) -> int:
        bits = 0
        for v in values:
            bits |= int(mapping.get(int(v), 0))
        return int(bits)

    def query_policy_bitset(self, p: QueryPolicy) -> int:
        """Exact non-epoch row-slice bitmap for tenant/scalar/provenance atoms."""
        tenants = self._mask_values(int(p.tenant_mask), int(self.ds.metadata.get("tenants", 64)))
        regions = self._mask_values(int(p.region_mask), int(self.ds.metadata.get("regions", 64)))
        dtypes = self._mask_values(int(p.dtype_mask), int(self.ds.metadata.get("dtypes", 64)))
        provs = self._mask_values(int(p.provenance_mask), int(self.ds.metadata.get("prov_tags", 64)))
        if not tenants or not regions or not dtypes or not provs:
            return 0
        bits = self._or_bitsets(tenants, self.tenant_value_bitsets)
        bits &= self._or_bitsets(regions, self.region_value_bitsets)
        bits &= self._or_bitsets(dtypes, self.dtype_value_bitsets)
        bits &= self._or_bitsets(provs, self.prov_value_bitsets)
        sens_idx = min(max(0, int(p.max_sensitivity)), len(self.sensitivity_leq_bitsets) - 1)
        bits &= int(self.sensitivity_leq_bitsets[sens_idx])
        return int(bits)

    def slice_candidate_ids(self, c: int, p: QueryPolicy, query_bits: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Return row identifiers from exact policy-cooccurrence bitmap slices.

        The slice index is a secondary physical structure over stored row facts,
        not an authorization decision. It enumerates rows whose tenant, region,
        dtype, provenance bit, and sensitivity bucket can satisfy the query's
        SQL-policy atoms. Snapshot epoch predicates are deliberately not pushed
        into this slice because insert/delete visibility must be checked at the
        authoritative row boundary.
        """
        if query_bits is None:
            query_bits = self.query_policy_bitset(p)
        bits = int(self.cluster_bitsets[int(c)]) & int(query_bits)
        ids = self._bitset_to_ids(bits)
        return ids, int(ids.size)

    def index_overhead(self) -> Dict[str, float]:
        n = self.ds.n
        assignments = n * 4
        centroids = self.nlist * self.ds.dim * 4
        radii = self.nlist * 4
        summaries = self.nlist * (8 * 4 + 2 + 4 + 4 + 4)  # masks, sensitivity, epoch bounds, angular radius
        # Logical co-occurrence signatures are represented as Python sets in the
        # prototype; a production engine would encode them as compact bitmaps or
        # counted sketches. We account them as 8-byte signature identifiers to
        # avoid hiding the extra metadata used for stronger pruning.
        signature_bytes = 8 * sum(len(s) for s in self.signature_sets)
        # Row-slice postings are counted as 32-bit row identifiers.  Some rows
        # appear in more than one provenance-bit posting; this cost is the price
        # of avoiding full cluster scans under mixed SQL/vector policies.
        posting_entries = sum(int(arr.size) for d in self.slice_postings for arr in d.values())
        posting_bytes = 4 * posting_entries
        bitset_bytes = (self.ds.n + 7) // 8 * (self.nlist + len(self.tenant_value_bitsets) + len(self.region_value_bitsets) + len(self.dtype_value_bitsets) + len(self.prov_value_bitsets) + len(self.sensitivity_leq_bitsets))
        return {
            "nlist": float(self.nlist),
            "assignment_bytes_per_record": float(assignments / n),
            "centroid_bytes_per_record": float(centroids / n),
            "radius_bytes_per_record": float(radii / n),
            "summary_bytes_per_record": float((summaries + signature_bytes) / n),
            "signature_bytes_per_record": float(signature_bytes / n),
            "slice_posting_bytes_per_record": float(posting_bytes / n),
            "bitmap_slice_bytes_per_record": float(bitset_bytes / n),
            "total_structural_bytes_per_record": float((assignments + centroids + radii + summaries + signature_bytes + posting_bytes + bitset_bytes) / n),
        }


def _topk_from_scores(ids: np.ndarray, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    kk = min(int(k), int(ids.size))
    if ids.size > kk:
        part = np.argpartition(-scores, kk - 1)[:kk]
        order = part[np.lexsort((ids[part], -scores[part]))]
    else:
        order = np.lexsort((ids, -scores))
    return ids[order].astype(np.int64), scores[order].astype(np.float32)


def exact_secure(ds: PolicyVectorDataset, p: QueryPolicy, k: int = 10) -> Dict:
    t0 = time.perf_counter()
    mask = ds.policy_mask(p, include_deletions=False)
    ids = np.flatnonzero(mask).astype(np.int64)
    scores = ds.vectors[ids] @ p.qvec if ids.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(ids, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(ids.size), "raw_candidates": int(ds.n), "distance_evals": int(ids.size), "policy_checks": int(ds.n), "summary_checks": 0, "visited_cluster_count": 0, "certified": True, "certificate_gap": 0.0, "visited_clusters": []}


def bitmap_slice_exact(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10) -> Dict:
    """Exact row-slice prefilter baseline using PACER's non-epoch bitmap access path.

    This baseline gives reviewers a strong non-ANN competitor: it uses the same
    exact tenant/scalar/provenance/sensitivity bitmap slice as PACER, then applies
    the authoritative insertion/deletion verifier and scores every visible row in
    the slice. It is exact but does not attempt early stopping or upper-bound
    certification. PACER-C/X must therefore justify their contract with less or
    equal physical work rather than by comparing only against weak post-filtering.
    """
    t0 = time.perf_counter()
    ids = index._bitset_to_ids(index.query_policy_bitset(p))
    legal = ids[ds.policy_mask_ids(ids, p, include_deletions=False)] if ids.size else np.empty(0, dtype=np.int64)
    scores = ds.vectors[legal] @ p.qvec if legal.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(legal, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(legal.size), "raw_candidates": int(ids.size), "distance_evals": int(legal.size), "policy_checks": int(ids.size), "summary_checks": 0, "visited_cluster_count": 0, "certified": True, "certificate_gap": 0.0, "visited_clusters": [], "exact_bitmap_slice": True}


def metadata_only(ds: PolicyVectorDataset, p: QueryPolicy, k: int = 10) -> Dict:
    t0 = time.perf_counter()
    ids = np.flatnonzero(ds.policy_mask(p, include_deletions=False)).astype(np.int64)
    scores = ds.quality[ids] if ids.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(ids, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(ids.size), "raw_candidates": int(ds.n), "distance_evals": 0, "policy_checks": int(ds.n), "summary_checks": 0, "visited_cluster_count": 0, "certified": False, "certificate_gap": 0.0, "visited_clusters": []}


def pre_filter_exact(ds: PolicyVectorDataset, p: QueryPolicy, k: int = 10) -> Dict:
    """Exact physical pre-filter baseline: score every row satisfying policy and tombstone predicate."""
    return exact_secure(ds, p, k=k)


def scalar_only(ds: PolicyVectorDataset, p: QueryPolicy, k: int = 10) -> Dict:
    """Negative control that applies policy but ranks by scalar quality, not vector similarity."""
    return metadata_only(ds, p, k=k)


def _ivf_candidate_ids(index: IVFIndex, qvec: np.ndarray, nprobe: int, candidate_budget: int) -> Tuple[np.ndarray, int]:
    """Return IVF candidates ranked by row score inside the probed lists.

    Earlier drafts used physical row-id order within a list, which is an
    unrealistic implementation detail and can bias both PACER and baselines.
    This function now mirrors a conventional IVF reranking stage: choose the
    nearest coarse lists, score rows in those lists, and keep the best raw
    candidate identifiers under the requested budget.
    """
    cs = index.centroids @ qvec
    order = np.argsort(-cs)[: min(int(nprobe), index.nlist)]
    chunks = [index.cluster_ids[int(c)] for c in order if index.cluster_ids[int(c)].size]
    if not chunks:
        return np.empty(0, dtype=np.int64), int(len(order))
    ids = np.concatenate(chunks).astype(np.int64)
    scores = index.ds.vectors[ids] @ qvec
    budget = min(int(candidate_budget), int(ids.size))
    if ids.size > budget:
        part = np.argpartition(-scores, budget - 1)[:budget]
        part = part[np.lexsort((ids[part], -scores[part]))]
        ids = ids[part]
    else:
        order2 = np.lexsort((ids, -scores))
        ids = ids[order2]
    return ids.astype(np.int64), int(len(order))


def post_filter_ivf(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10, nprobe: int = 8, candidate_budget: int = 1200) -> Dict:
    t0 = time.perf_counter()
    raw, visited = _ivf_candidate_ids(index, p.qvec, nprobe, candidate_budget)
    legal = raw[ds.policy_mask_ids(raw, p, include_deletions=False)] if raw.size else np.empty(0, dtype=np.int64)
    scores = ds.vectors[legal] @ p.qvec if legal.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(legal, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(legal.size), "raw_candidates": int(raw.size), "distance_evals": int(raw.size), "policy_checks": int(raw.size), "summary_checks": 0, "visited_cluster_count": int(visited), "certified": False, "certificate_gap": 0.0, "visited_clusters": []}


def exact_prefix_post_filter(ds: PolicyVectorDataset, p: QueryPolicy, k: int = 10, candidate_budget: int = 2400) -> Dict:
    """Strong post-filter control using the exact global score prefix.

    This method is not an implementable ANN shortcut because it scores every row
    to obtain the true global prefix before filtering.  It is included to answer
    a reviewer question: even a perfect global-prefix access path is still not a
    declarative policy-constrained top-k plan unless its prefix reaches the
    visible rank depth.
    """
    t0 = time.perf_counter()
    ids = np.arange(ds.n, dtype=np.int64)
    scores_all = ds.vectors @ p.qvec
    prefix = min(int(candidate_budget), int(ds.n))
    order = np.lexsort((ids, -scores_all))[:prefix]
    raw = ids[order]
    legal = raw[ds.policy_mask_ids(raw, p, include_deletions=False)]
    scores = ds.vectors[legal] @ p.qvec if legal.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(legal, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {
        "ids": top_ids,
        "scores": top_scores,
        "latency_ms": lat,
        "candidates": int(legal.size),
        "raw_candidates": int(raw.size),
        "distance_evals": int(ds.n),
        "policy_checks": int(raw.size),
        "summary_checks": 0,
        "visited_cluster_count": 0,
        "certified": False,
        "certificate_gap": 0.0,
        "visited_clusters": [],
        "global_prefix_exact": True,
    }


def adaptive_post_filter_ivf(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10, candidate_budget: int = 4800) -> Dict:
    t0 = time.perf_counter()
    cs = index.centroids @ p.qvec
    order = np.argsort(-cs)
    chunks = []
    raw_total = 0
    legal_total = 0
    visited = []
    for c in order:
        if raw_total >= int(candidate_budget):
            break
        ids = index.cluster_ids[int(c)]
        room = max(0, int(candidate_budget) - raw_total)
        if ids.size > room:
            local_scores = ds.vectors[ids] @ p.qvec
            local_order = np.lexsort((ids, -local_scores))
            take = ids[local_order[:room]]
        else:
            local_scores = ds.vectors[ids] @ p.qvec if ids.size else np.empty(0, dtype=np.float32)
            local_order = np.lexsort((ids, -local_scores)) if ids.size else np.empty(0, dtype=np.int64)
            take = ids[local_order] if ids.size else ids
        raw_total += int(take.size)
        visited.append(int(c))
        legal = take[ds.policy_mask_ids(take, p, include_deletions=False)] if take.size else np.empty(0, dtype=np.int64)
        if legal.size:
            chunks.append(legal.astype(np.int64))
            legal_total += int(legal.size)
    cand = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    scores = ds.vectors[cand] @ p.qvec if cand.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(cand, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(cand.size), "raw_candidates": int(raw_total), "distance_evals": int(raw_total), "policy_checks": int(raw_total), "summary_checks": 0, "visited_cluster_count": int(len(visited)), "certified": False, "certificate_gap": 0.0, "visited_clusters": visited}


def tenant_partition_ivf(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10, nprobe: int = 8, candidate_budget: int = 1200) -> Dict:
    t0 = time.perf_counter()
    raw, visited = _ivf_candidate_ids(index, p.qvec, nprobe, candidate_budget)
    tenant_ok = _mask_membership(ds.tenant[raw], p.tenant_mask) if raw.size else np.empty(0, dtype=bool)
    raw2 = raw[tenant_ok]
    legal = raw2[ds.policy_mask_ids(raw2, p, include_deletions=False)] if raw2.size else np.empty(0, dtype=np.int64)
    scores = ds.vectors[legal] @ p.qvec if legal.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(legal, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(legal.size), "raw_candidates": int(raw.size), "distance_evals": int(raw.size), "policy_checks": int(raw.size), "summary_checks": 0, "visited_cluster_count": int(visited), "certified": False, "certificate_gap": 0.0, "visited_clusters": []}



def prefilter_ivf(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10, nprobe: int = 8, candidate_budget: int = 1200) -> Dict:
    """Conventional inline/pre-filtered IVF baseline.

    Cluster summaries are used only to choose possibly relevant lists; final row
    visibility is checked before scoring. The method has no top-k certificate and
    can miss legal rows in summary-compatible but unvisited clusters.
    """
    t0 = time.perf_counter()
    upper = index.cluster_upper_bounds(p.qvec)
    order = np.argsort(-upper)
    chunks: List[np.ndarray] = []
    raw_total = 0
    visited = []
    summary_checks = 0
    for c0 in order:
        c = int(c0)
        summary_checks += 1
        if not index.may_satisfy(c, p, include_deletions=False):
            continue
        visited.append(c)
        ids = index.cluster_ids[c]
        raw_total += int(ids.size)
        if ids.size:
            legal = ids[ds.policy_mask_ids(ids, p, include_deletions=False)]
            if legal.size:
                chunks.append(legal.astype(np.int64))
        if len(visited) >= int(nprobe) or raw_total >= int(candidate_budget):
            break
    cand = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    scores = ds.vectors[cand] @ p.qvec if cand.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(cand, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(cand.size), "raw_candidates": int(raw_total), "distance_evals": int(cand.size), "policy_checks": int(raw_total), "summary_checks": int(summary_checks), "visited_cluster_count": int(len(visited)), "certified": False, "certificate_gap": 0.0, "visited_clusters": visited}

def stale_no_recheck(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10, nprobe: int = 8, candidate_budget: int = 1200) -> Dict:
    t0 = time.perf_counter()
    raw, visited = _ivf_candidate_ids(index, p.qvec, nprobe, candidate_budget)
    legal = raw[ds.policy_mask_ids(raw, p, include_deletions=True)] if raw.size else np.empty(0, dtype=np.int64)
    scores = ds.vectors[legal] @ p.qvec if legal.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(legal, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(legal.size), "raw_candidates": int(raw.size), "distance_evals": int(raw.size), "policy_checks": int(raw.size), "summary_checks": 0, "visited_cluster_count": int(visited), "certified": False, "certificate_gap": 0.0, "visited_clusters": []}


def naive_no_policy_ivf(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10, nprobe: int = 8, candidate_budget: int = 1200) -> Dict:
    t0 = time.perf_counter()
    raw, visited = _ivf_candidate_ids(index, p.qvec, nprobe, candidate_budget)
    scores = ds.vectors[raw] @ p.qvec if raw.size else np.empty(0, dtype=np.float32)
    top_ids, top_scores = _topk_from_scores(raw, scores, k)
    lat = (time.perf_counter() - t0) * 1000
    return {"ids": top_ids, "scores": top_scores, "latency_ms": lat, "candidates": int(raw.size), "raw_candidates": int(raw.size), "distance_evals": int(raw.size), "policy_checks": 0, "summary_checks": 0, "visited_cluster_count": int(visited), "certified": False, "certificate_gap": 0.0, "visited_clusters": []}


def caps_search(
    ds: PolicyVectorDataset,
    index: IVFIndex,
    p: QueryPolicy,
    k: int = 10,
    candidate_budget: int = 1200,
    use_policy_slices: bool = True,
    use_bounds: bool = True,
    force_certify: bool = False,
    budget_mode: str = "raw",
    bound_mode: str = "cone",
    use_cooccurrence: bool = True,
) -> Dict:
    """Policy-aware candidate enumeration with optional top-k certificate.

    PACER never prunes a cluster using row-level visibility computed over the full
    table. It only uses conservative cluster summaries. Authoritative policy and
    tombstone predicates are evaluated for rows in visited clusters. The running
    top-k is maintained incrementally, so certifying fallback does not repeatedly
    rescore all previously seen candidates.
    """
    t0 = time.perf_counter()
    upper = index.cluster_upper_bounds(p.qvec, mode=bound_mode) if use_bounds else (index.centroids @ p.qvec)
    order = np.argsort(-upper)
    may = np.ones(index.nlist, dtype=bool)
    if use_policy_slices:
        may = np.array([index.may_satisfy(c, p, include_deletions=False, use_signatures=use_cooccurrence) for c in range(index.nlist)], dtype=bool)
    processed = np.zeros(index.nlist, dtype=bool)
    raw_candidates = 0
    legal_candidates = 0
    policy_checks = 0
    summary_checks = 0
    visited_clusters: List[int] = []
    processed_clusters: List[int] = []
    skipped_clusters: List[int] = []
    certified = False
    truncated_visible_slice = False
    gap = 0.0
    top_ids = np.empty(0, dtype=np.int64)
    top_scores = np.empty(0, dtype=np.float32)
    budget = INF_EPOCH if force_certify else int(candidate_budget)

    def remaining_possible() -> np.ndarray:
        rem0 = np.flatnonzero(~processed)
        if use_policy_slices:
            rem0 = rem0[may[rem0]]
        return rem0.astype(np.int64)

    for c0 in order:
        c = int(c0)
        processed[c] = True
        processed_clusters.append(c)
        summary_checks += 1
        if use_policy_slices and not may[c]:
            skipped_clusters.append(c)
            continue
        ids = index.cluster_ids[c]
        if ids.size == 0:
            continue
        visited_clusters.append(c)
        ids_to_check = ids
        if not force_certify and budget_mode == "raw":
            room_raw = max(0, budget - raw_candidates)
            if int(ids_to_check.size) > room_raw:
                # Only the partially consumed unit needs local score ordering;
                # fully consumed units are independent of list order.
                local_scores = ds.vectors[ids_to_check] @ p.qvec
                local_order = np.lexsort((ids_to_check, -local_scores))
                ids_to_check = ids_to_check[local_order[:room_raw]]
                truncated_visible_slice = True
        raw_candidates += int(ids_to_check.size)
        policy_checks += int(ids_to_check.size)
        legal = ids_to_check[ds.policy_mask_ids(ids_to_check, p, include_deletions=False)]
        if legal.size:
            if not force_certify and budget_mode == "visible":
                room = max(0, budget - legal_candidates)
                if int(legal.size) > room:
                    truncated_visible_slice = True
                    legal = legal[:room]
            if legal.size:
                scores = ds.vectors[legal] @ p.qvec
                merge_ids = np.concatenate([top_ids, legal.astype(np.int64)]) if top_ids.size else legal.astype(np.int64)
                merge_scores = np.concatenate([top_scores, scores.astype(np.float32)]) if top_scores.size else scores.astype(np.float32)
                top_ids, top_scores = _topk_from_scores(merge_ids, merge_scores, k)
                legal_candidates += int(legal.size)
        rem = remaining_possible()
        max_remaining_ub = float(np.max(upper[rem])) if rem.size else -math.inf
        if not truncated_visible_slice:
            if top_scores.size >= k:
                kth = float(top_scores[-1])
                gap = max(0.0, max_remaining_ub - kth) if math.isfinite(max_remaining_ub) else 0.0
                if use_bounds and kth > max_remaining_ub + 1e-12:
                    certified = True
                    break
            elif rem.size == 0:
                gap = 0.0
                certified = True
                break
        if not force_certify:
            if budget_mode == "visible" and legal_candidates >= budget:
                break
            if budget_mode == "raw" and raw_candidates >= budget:
                break

    if not certified and not truncated_visible_slice:
        rem = remaining_possible()
        if rem.size == 0:
            certified = True
            gap = 0.0
    lat = (time.perf_counter() - t0) * 1000
    return {
        "ids": top_ids,
        "scores": top_scores,
        "latency_ms": lat,
        "candidates": int(legal_candidates),
        "raw_candidates": int(raw_candidates),
        "distance_evals": int(raw_candidates),
        "policy_checks": int(policy_checks),
        "summary_checks": int(summary_checks),
        "visited_cluster_count": int(len(visited_clusters)),
        "certified": bool(certified),
        "certificate_gap": float(0.0 if certified else gap),
        "visited_clusters": visited_clusters,
        "processed_clusters": processed_clusters,
        "skipped_clusters": skipped_clusters,
        "truncated_visible_slice": int(truncated_visible_slice),
        "budget_mode": budget_mode,
        "bound_mode": bound_mode if use_bounds else "centroid",
        "cooccurrence_summary": int(use_cooccurrence),
    }


def recall_at_k(gold: np.ndarray, got: np.ndarray, k: int = 10) -> float:
    gold = np.asarray(gold, dtype=np.int64)
    got = np.asarray(got, dtype=np.int64)
    if gold.size == 0:
        return 1.0
    denom = min(int(k), int(gold.size))
    return float(len(set(gold[:denom]).intersection(set(got[:int(k)]))) / denom)


def exact_ordered(gold: np.ndarray, got: np.ndarray, k: int = 10) -> bool:
    need = min(int(k), int(len(gold)))
    got_prefix = [int(x) for x in np.asarray(got, dtype=np.int64)[:int(k)]]
    gold_prefix = [int(x) for x in np.asarray(gold, dtype=np.int64)[:need]]
    return len(got_prefix) == need and got_prefix == gold_prefix


def failure_mode(gold: np.ndarray, got: np.ndarray, k: int = 10) -> str:
    need = min(int(k), int(len(gold)))
    if need == 0 or exact_ordered(gold, got, k):
        return "none"
    if len(got) < need:
        return "candidate_starvation"
    return "ranking_truncation"


def _cluster_visible_counts(ds: PolicyVectorDataset, index: IVFIndex, visible: np.ndarray) -> np.ndarray:
    counts = np.bincount(index.assignments[visible], minlength=index.nlist)
    return counts.astype(np.int32)


def make_audit_transcript(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, result: Dict, gold: Dict, k: int) -> Dict:
    visible = ds.policy_mask(p, include_deletions=False)
    counts = _cluster_visible_counts(ds, index, visible)
    ub = index.cluster_upper_bounds(p.qvec)
    visited = set(int(x) for x in result.get("visited_clusters", []))
    exact_fallback = bool(result.get("exact_fallback", False))
    unvisited_visible = [] if exact_fallback else [int(i) for i in range(index.nlist) if i not in visited and int(counts[i]) > 0]
    kth_score = float(result["scores"][-1]) if len(result.get("scores", [])) else None
    max_unvisited_bound = float(np.max(ub[unvisited_visible])) if unvisited_visible else None
    cert = bool(result.get("certified", False))
    return {
        "query": {
            "k": int(k),
            "epoch": int(p.epoch),
            "regime": p.regime,
            "correlation": p.correlation,
            "source_id": int(p.source_id),
            "policy_anchor_id": int(p.policy_anchor_id),
            "tenant_mask": int(p.tenant_mask),
            "max_sensitivity": int(p.max_sensitivity),
            "region_mask": int(p.region_mask),
            "dtype_mask": int(p.dtype_mask),
            "provenance_mask": int(p.provenance_mask),
        },
        "result_ids": [int(x) for x in result["ids"]],
        "gold_ids": [int(x) for x in gold["ids"]],
        "row_witnesses": ds.row_facts(result["ids"], p),
        "certificate": {
            "certified": cert,
            "basis": "visible_relation_enumeration" if exact_fallback else "cluster_upper_bound",
            "exact_fallback": exact_fallback,
            "visited_clusters": sorted(list(visited)),
            "unvisited_visible_clusters_first50": unvisited_visible[:50],
            "num_unvisited_visible_clusters": int(len(unvisited_visible)),
            "kth_score": kth_score,
            "max_unvisited_bound": max_unvisited_bound,
            "certificate_gap": float(result.get("certificate_gap", 0.0)),
            "truncated_visible_slice": int(result.get("truncated_visible_slice", 0)),
            "certified_implies_ordered_exact": bool((not cert) or exact_ordered(gold["ids"], result["ids"], k)),
        },
    }



def pacer_sliced_search(
    ds: PolicyVectorDataset,
    index: IVFIndex,
    p: QueryPolicy,
    k: int = 10,
    candidate_budget: int = 1200,
    use_bounds: bool = True,
    force_certify: bool = False,
    bound_mode: str = "cone",
) -> Dict:
    """Policy-sliced PACER search using exact row-slice postings.

    This is the main PACER-A/C/X physical operator in the final artifact.  The
    access path first prunes clusters using no-false-negative summaries and then
    enumerates only row identifiers contained in policy-cooccurrence postings
    for tenant, scalar, and provenance atoms.  The row verifier still checks the
    full visibility predicate, including insert/delete epochs, before any output
    is emitted.  Certification uses the same upper-bound proof as ``caps_search``.
    """
    t0 = time.perf_counter()
    upper = index.cluster_upper_bounds(p.qvec, mode=bound_mode) if use_bounds else (index.centroids @ p.qvec)
    order = np.argsort(-upper)
    query_bits = index.query_policy_bitset(p)
    e = int(p.epoch)
    may = np.array([
        ((int(index.cluster_bitsets[c]) & int(query_bits)) != 0)
        and int(index.min_insert_epoch[c]) <= e
        and int(index.max_delete_epoch[c]) > e
        for c in range(index.nlist)
    ], dtype=bool)
    processed = np.zeros(index.nlist, dtype=bool)
    raw_candidates = 0
    posting_entries = 0
    legal_candidates = 0
    policy_checks = 0
    summary_checks = 0
    visited_clusters: List[int] = []
    processed_clusters: List[int] = []
    skipped_clusters: List[int] = []
    certified = False
    truncated_visible_slice = False
    gap = 0.0
    top_ids = np.empty(0, dtype=np.int64)
    top_scores = np.empty(0, dtype=np.float32)
    budget = INF_EPOCH if force_certify else int(candidate_budget)

    def remaining_possible() -> np.ndarray:
        rem0 = np.flatnonzero(~processed)
        rem0 = rem0[may[rem0]]
        return rem0.astype(np.int64)

    for c0 in order:
        c = int(c0)
        processed[c] = True
        processed_clusters.append(c)
        summary_checks += 1
        if not may[c]:
            skipped_clusters.append(c)
            continue
        ids, postings = index.slice_candidate_ids(c, p, query_bits=query_bits)
        posting_entries += int(postings)
        if ids.size == 0:
            # The exact posting slice can be empty even if the coarse summary is
            # conservatively non-empty.  The cluster is nevertheless processed.
            continue
        visited_clusters.append(c)
        if not force_certify:
            room_raw = max(0, budget - raw_candidates)
            if int(ids.size) > room_raw:
                local_scores = ds.vectors[ids] @ p.qvec
                local_order = np.lexsort((ids, -local_scores))
                ids = ids[local_order[:room_raw]]
                truncated_visible_slice = True
        raw_candidates += int(ids.size)
        policy_checks += int(ids.size)
        legal = ids[ds.policy_mask_ids(ids, p, include_deletions=False)]
        if legal.size:
            scores = ds.vectors[legal] @ p.qvec
            merge_ids = np.concatenate([top_ids, legal.astype(np.int64)]) if top_ids.size else legal.astype(np.int64)
            merge_scores = np.concatenate([top_scores, scores.astype(np.float32)]) if top_scores.size else scores.astype(np.float32)
            top_ids, top_scores = _topk_from_scores(merge_ids, merge_scores, k)
            legal_candidates += int(legal.size)
        rem = remaining_possible()
        max_remaining_ub = float(np.max(upper[rem])) if rem.size else -math.inf
        if not truncated_visible_slice:
            if top_scores.size >= k:
                kth = float(top_scores[-1])
                gap = max(0.0, max_remaining_ub - kth) if math.isfinite(max_remaining_ub) else 0.0
                if use_bounds and kth > max_remaining_ub + 1e-12:
                    certified = True
                    break
            elif rem.size == 0:
                gap = 0.0
                certified = True
                break
        if not force_certify and raw_candidates >= budget:
            break
    if not certified and not truncated_visible_slice:
        rem = remaining_possible()
        if rem.size == 0:
            certified = True
            gap = 0.0
    lat = (time.perf_counter() - t0) * 1000
    return {
        "ids": top_ids,
        "scores": top_scores,
        "latency_ms": lat,
        "candidates": int(legal_candidates),
        "raw_candidates": int(raw_candidates),
        "distance_evals": int(raw_candidates),
        "policy_checks": int(policy_checks),
        "summary_checks": int(summary_checks),
        "visited_cluster_count": int(len(visited_clusters)),
        "certified": bool(certified),
        "certificate_gap": float(0.0 if certified else gap),
        "visited_clusters": visited_clusters,
        "processed_clusters": processed_clusters,
        "skipped_clusters": skipped_clusters,
        "truncated_visible_slice": int(truncated_visible_slice),
        "budget_mode": "raw_slice",
        "bound_mode": bound_mode if use_bounds else "centroid",
        "cooccurrence_summary": 1,
        "slice_posting_entries": int(posting_entries),
    }

def contract_planner(ds: PolicyVectorDataset, index: IVFIndex, p: QueryPolicy, k: int = 10,
                     candidate_budget: int = 2400, service_level: str = "certify-if-cheap") -> Dict:
    """Contract-aware service-level planner.

    The planner is deliberately simple and auditable. It first obtains a fair
    raw-work PACER-A answer. If the caller asks for certification, or if the
    approximate answer is close to certification after bounded work, it refines
    the same execution family with progressively larger raw budgets before
    falling back to PACER-X. This is not an oracle: exact comparison is used only
    by evaluation, not by the planner.
    """
    base = pacer_sliced_search(ds, index, p, k=k, candidate_budget=candidate_budget,
                               use_bounds=True, force_certify=False, bound_mode="cone")
    base["planner_path"] = "bounded"
    if service_level == "checked":
        return base
    if bool(base.get("certified", False)):
        base["planner_path"] = "bounded_certified"
        return base
    # Refine when the certificate gap is small or the answer is too short.  The
    # rule is based only on executor-visible certificate statistics.
    gap = float(base.get("certificate_gap", 1.0))
    too_short = len(base.get("ids", [])) < min(k, int(np.sum(ds.policy_mask(p, include_deletions=False))))
    if service_level in {"certify", "certify-if-cheap"} or gap <= 0.60 or too_short:
        last = base
        for mult in (2, 4):
            res = pacer_sliced_search(ds, index, p, k=k, candidate_budget=min(ds.n, candidate_budget * mult),
                                      use_bounds=True, force_certify=False, bound_mode="cone")
            res["planner_path"] = f"refine_{mult}x"
            last = res
            if bool(res.get("certified", False)):
                return res
        if service_level != "certify":
            return last
        res = pacer_sliced_search(ds, index, p, k=k, candidate_budget=ds.n,
                                  use_bounds=True, force_certify=True, bound_mode="cone")
        res["planner_path"] = "certifying_fallback"
        return res
    return base


def _method_dict(ds: PolicyVectorDataset, index: IVFIndex, k: int, candidate_budget: int) -> Dict[str, Callable[[QueryPolicy], Dict]]:
    """Return paper-facing method names with stable semantics.

    PACER-X is not a label for the exact oracle. It runs the same policy-aware
    cluster ordering as PACER-A and continues until either the upper-bound
    certificate succeeds or the visible relation is exhausted. In the latter
    case it has effectively performed a certified exact fallback under the same
    row verifier used by the approximate mode.
    """
    return {
        "ExactSecure": lambda p: exact_secure(ds, p, k=k),
        "PreFilter-Exact": lambda p: pre_filter_exact(ds, p, k=k),
        "BitmapSlice-Exact": lambda p: bitmap_slice_exact(ds, index, p, k=k),
        "ScalarOnly": lambda p: metadata_only(ds, p, k=k),
        "PostFilter-IVF": lambda p: post_filter_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
        "AdaptivePostFilter-IVF": lambda p: adaptive_post_filter_ivf(ds, index, p, k=k, candidate_budget=max(candidate_budget, 4800)),
        "ExactPrefix-PostFilter": lambda p: exact_prefix_post_filter(ds, p, k=k, candidate_budget=candidate_budget),
        "TenantPartition-IVF": lambda p: tenant_partition_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
        "PreFilter-IVF": lambda p: prefilter_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
        "PACER-A": lambda p: pacer_sliced_search(ds, index, p, k=k, candidate_budget=candidate_budget, use_bounds=True, force_certify=False, bound_mode="cone"),
        "PACER-C": lambda p: contract_planner(ds, index, p, k=k, candidate_budget=candidate_budget, service_level="certify-if-cheap"),
        "PACER-X": lambda p: pacer_sliced_search(ds, index, p, k=k, candidate_budget=ds.n, use_bounds=True, force_certify=True, bound_mode="cone"),
        "PACER-A-NoSlices": lambda p: caps_search(ds, index, p, k=k, candidate_budget=candidate_budget, use_policy_slices=False, use_bounds=True, force_certify=False, budget_mode="raw", bound_mode="cone", use_cooccurrence=False),
        "PACER-A-NoBounds": lambda p: pacer_sliced_search(ds, index, p, k=k, candidate_budget=candidate_budget, use_bounds=False, force_certify=False, bound_mode="cone"),
        "StaleNoRecheck": lambda p: stale_no_recheck(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
        "NaiveNoPolicy": lambda p: naive_no_policy_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget),
    }

def summarize_metrics(rows: List[Dict], budget_rows: List[Dict], deletion_rows: List[Dict], overhead: Dict[str, float]) -> Dict:
    def agg(filter_fn=lambda r: True):
        data = [r for r in rows if filter_fn(r)]
        if not data:
            return {}
        out = {}
        for m in sorted(set(r["method"] for r in data)):
            xs = [r for r in data if r["method"] == m]
            certified_xs = [r for r in xs if int(r["certified"]) == 1]
            out[m] = {
                "recall_at_k": float(np.mean([float(r["recall_at_k"]) for r in xs])),
                "secure_topk_exact": float(np.mean([float(r["secure_topk_exact"]) for r in xs])),
                "selectivity": float(np.mean([float(r["selectivity"]) for r in xs])),
                "candidate_starvation": float(np.mean([r["failure_mode"] == "candidate_starvation" for r in xs])),
                "ranking_truncation": float(np.mean([r["failure_mode"] == "ranking_truncation" for r in xs])),
                "policy_violations_per_query": float(np.mean([int(r["policy_violations"]) for r in xs])),
                "deleted_violations_per_query": float(np.mean([int(r["deleted_violations"]) for r in xs])),
                "tenant_violations_per_query": float(np.mean([int(r["tenant_violations"]) for r in xs])),
                "latency_ms": float(np.mean([float(r["latency_ms"]) for r in xs])),
                "median_latency_ms": float(np.median([float(r["latency_ms"]) for r in xs])),
                "p95_latency_ms": float(np.percentile([float(r["latency_ms"]) for r in xs], 95)),
                "candidates": float(np.mean([int(r["candidates"]) for r in xs])),
                "raw_candidates": float(np.mean([int(r["raw_candidates"]) for r in xs])),
                "distance_evals": float(np.mean([int(r["distance_evals"]) for r in xs])),
                "policy_checks": float(np.mean([int(r["policy_checks"]) for r in xs])),
                "summary_checks": float(np.mean([int(r.get("summary_checks", 0)) for r in xs])),
                "visited_cluster_count": float(np.mean([int(r["visited_cluster_count"]) for r in xs])),
                "certified_fraction": float(np.mean([int(r["certified"]) for r in xs])),
                "certificate_false_positive_per_query": float(np.mean([int(r["certificate_false_positive"]) for r in xs])),
                "certified_sound_fraction": float(np.mean([int(r["secure_topk_exact"]) for r in certified_xs])) if certified_xs else 1.0,
                "certificate_errors": float(np.mean([(1 - int(r["secure_topk_exact"])) for r in certified_xs])) if certified_xs else 0.0,
                "mean_certificate_gap": float(np.mean([float(r["certificate_gap"]) for r in xs])),
                "explanation_accuracy": float(np.mean([float(r["explanation_accuracy"]) for r in xs])),
            }
        return out

    budget_summary: Dict[str, Dict] = {}
    for m in sorted(set(r["method"] for r in budget_rows)):
        budget_summary[m] = {}
        for b in sorted(set(int(r["budget"]) for r in budget_rows)):
            xs = [r for r in budget_rows if r["method"] == m and int(r["budget"]) == b]
            budget_summary[m][str(b)] = {
                "recall_at_k": float(np.mean([float(r["recall_at_k"]) for r in xs])),
                "secure_topk_exact": float(np.mean([float(r["secure_topk_exact"]) for r in xs])),
                "latency_ms": float(np.mean([float(r["latency_ms"]) for r in xs])),
                "candidates": float(np.mean([int(r["candidates"]) for r in xs])),
                "raw_candidates": float(np.mean([int(r["raw_candidates"]) for r in xs])),
                "policy_checks": float(np.mean([int(r["policy_checks"]) for r in xs])),
                "policy_violations_per_query": float(np.mean([int(r["policy_violations"]) for r in xs])),
                "deleted_violations_per_query": float(np.mean([int(r["deleted_violations"]) for r in xs])),
                "certified_fraction": float(np.mean([int(r["certified"]) for r in xs])),
            }
    regimes = sorted(set(r["regime"] for r in rows))
    safe_methods = {"ExactSecure", "PreFilter-Exact", "BitmapSlice-Exact", "ScalarOnly", "PostFilter-IVF", "AdaptivePostFilter-IVF", "TenantPartition-IVF", "PreFilter-IVF", "PACER-A", "PACER-C", "PACER-X", "PACER-A-NoSlices", "PACER-A-NoBounds"}
    safe_rows = [r for r in rows if r["method"] in safe_methods]
    claim_checks = {
        "safe_policy_deletion_tenant_violations_total": int(sum(int(r["policy_violations"]) + int(r["deleted_violations"]) + int(r["tenant_violations"]) for r in safe_rows)),
        "pacer_certificate_false_positive_total": int(sum(int(r["certificate_false_positive"]) for r in rows if r["method"] in {"PACER-A", "PACER-C", "PACER-X"})),
        "pacer_certify_ordered_failures_total": int(sum(1 - int(r["secure_topk_exact"]) for r in rows if r["method"] == "PACER-X")),
        "stale_no_recheck_deleted_results_total": int(sum(int(r["deleted_violations"]) for r in rows if r["method"] == "StaleNoRecheck")),
        "naive_no_policy_policy_violations_total": int(sum(int(r["policy_violations"]) + int(r["tenant_violations"]) for r in rows if r["method"] == "NaiveNoPolicy")),
    }
    return {
        "overall": agg(),
        "by_regime": {reg: agg(lambda r, reg=reg: r["regime"] == reg) for reg in regimes},
        "budget": budget_summary,
        "deletion": deletion_rows,
        "index_overhead": overhead,
        "claim_checks": claim_checks,
    }


def run_suite(
    out_dir: Path,
    n: int = 15000,
    dim: int = 64,
    per_regime: int = 40,
    k: int = 10,
    seed: int = 7,
    nlist: int = 64,
    candidate_budget: int = 1200,
    correlation: str = "positive",
    delete_rate: float = 0.12,
    tenant_cluster_correlation: float = 0.82,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = generate_dataset(n=n, dim=dim, true_clusters=max(48, nlist), seed=seed, delete_rate=delete_rate, tenant_cluster_correlation=tenant_cluster_correlation)
    index = IVFIndex(ds, nlist=nlist, seed=seed + 1)
    queries = generate_queries(ds, per_regime=per_regime, seed=seed + 2, correlation=correlation)
    methods = _method_dict(ds, index, k, candidate_budget)
    progress = os.environ.get("PACER_PROGRESS") == "1"
    if progress:
        print(f"[run_suite] building {len(queries)} exact oracles", file=sys.stderr, flush=True)
    golds = [exact_secure(ds, p, k=k) for p in queries]

    rows: List[Dict] = []
    audit_written = False
    for qi, (p, gold) in enumerate(zip(queries, golds)):
        if progress and (qi % 20 == 0 or qi == len(queries) - 1):
            print(f"[run_suite] main {qi+1}/{len(queries)}", file=sys.stderr, flush=True)
        allowed = int(ds.policy_mask(p, include_deletions=False).sum())
        selectivity = float(allowed / n)
        for name, fn in methods.items():
            res = fn(p)
            policy_v, del_v, tenant_v = ds.violations(res["ids"], p)
            ev, et = ds.explanations_valid(res["ids"], p)
            exact = int(exact_ordered(gold["ids"], res["ids"], k=k))
            certified = int(bool(res.get("certified", False)))
            cert_false = int(certified == 1 and exact == 0)
            rows.append({
                "query_id": qi,
                "regime": p.regime,
                "correlation": p.correlation,
                "method": name,
                "n": n,
                "dim": dim,
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
                "certificate_false_positive": cert_false,
                "certificate_gap": float(res.get("certificate_gap", 0.0)),
                "truncated_visible_slice": int(res.get("truncated_visible_slice", 0)),
            })
            if name == "PACER-X" and not audit_written:
                with (out_dir / "audit_transcript_sample.json").open("w") as f:
                    json.dump(make_audit_transcript(ds, index, p, res, gold, k), f, indent=2)
                audit_written = True

    csv_path = out_dir / f"metrics_n{n}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    budgets = [150, 300, 600, 1200, 2400, 4800]
    if progress:
        print("[run_suite] budget sweep", file=sys.stderr, flush=True)
    budget_rows: List[Dict] = []
    budget_limit = min(48, len(queries))
    for budget in budgets:
        sweep_methods = {
            "PostFilter-IVF": lambda p, b=budget: post_filter_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=b),
            "AdaptivePostFilter-IVF": lambda p, b=budget: adaptive_post_filter_ivf(ds, index, p, k=k, candidate_budget=b),
            "PACER-A": lambda p, b=budget: pacer_sliced_search(ds, index, p, k=k, candidate_budget=b),
        }
        for qi, (p, gold) in enumerate(zip(queries[:budget_limit], golds[:budget_limit])):
            for name, fn in sweep_methods.items():
                res = fn(p)
                policy_v, del_v, tenant_v = ds.violations(res["ids"], p)
                exact = int(exact_ordered(gold["ids"], res["ids"], k=k))
                budget_rows.append({
                    "query_id": qi,
                    "regime": p.regime,
                    "correlation": p.correlation,
                    "method": name,
                    "budget": int(budget),
                    "recall_at_k": recall_at_k(gold["ids"], res["ids"], k=k),
                    "secure_topk_exact": exact,
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
    budget_path = out_dir / f"budget_n{n}.csv"
    with budget_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        writer.writeheader(); writer.writerows(budget_rows)

    deletion_rows: List[Dict] = []
    if progress:
        print("[run_suite] deletion sweep", file=sys.stderr, flush=True)
    for epoch in range(1, 10):
        pcount = pacer_del = certify_del = stale_del = naive_del = 0
        for p0 in queries[: min(32, len(queries))]:
            p = QueryPolicy(p0.qvec, p0.tenant_mask, p0.max_sensitivity, p0.region_mask, p0.dtype_mask, p0.provenance_mask, epoch, p0.regime, p0.source_id, p0.policy_anchor_id, p0.correlation)
            r_pacer = pacer_sliced_search(ds, index, p, k=k, candidate_budget=candidate_budget)
            r_certify = pacer_sliced_search(ds, index, p, k=k, candidate_budget=ds.n, force_certify=True)
            r_stale = stale_no_recheck(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget)
            r_naive = naive_no_policy_ivf(ds, index, p, k=k, nprobe=8, candidate_budget=candidate_budget)
            _, del_v_a, _ = ds.violations(r_pacer["ids"], p)
            _, del_v_x, _ = ds.violations(r_certify["ids"], p)
            _, del_v_s, _ = ds.violations(r_stale["ids"], p)
            _, del_v_n, _ = ds.violations(r_naive["ids"], p)
            pacer_del += del_v_a; certify_del += del_v_x; stale_del += del_v_s; naive_del += del_v_n; pcount += 1
        deletion_rows.append({
            "epoch": epoch,
            "queries": pcount,
            "stale_residues_in_index": int(np.sum(ds.delete_epoch <= epoch)),
            "pacer_deleted_results": int(pacer_del),
            "pacer_certify_deleted_results": int(certify_del),
            "stale_no_recheck_deleted_results": int(stale_del),
            "naive_deleted_results": int(naive_del),
        })
    deletion_path = out_dir / f"deletion_n{n}.csv"
    with deletion_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(deletion_rows[0].keys()))
        writer.writeheader(); writer.writerows(deletion_rows)

    failure_rows: List[Dict] = []
    for method in sorted(set(r["method"] for r in rows)):
        xs = [r for r in rows if r["method"] == method]
        for fm in ["none", "candidate_starvation", "ranking_truncation"]:
            failure_rows.append({"method": method, "failure_mode": fm, "fraction": float(np.mean([r["failure_mode"] == fm for r in xs]))})
    with (out_dir / f"failure_modes_n{n}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
        writer.writeheader(); writer.writerows(failure_rows)

    summary = summarize_metrics(rows, budget_rows, deletion_rows, index.index_overhead())
    summary.update({
        "dataset": ds.metadata,
        "n": int(n),
        "dim": int(dim),
        "queries": int(len(queries)),
        "query_method_rows": int(len(rows)),
        "nlist": int(index.nlist),
        "k": int(k),
        "correlation": correlation,
        "candidate_budget": int(candidate_budget),
        "budget_semantics": "raw row checks for bounded ANN methods; PACER-A uses exact non-epoch bitmap slice candidates",
    })
    with (out_dir / f"summary_n{n}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary



def run_correlation_sweep(out_dir: Path, n: int = 12000, dim: int = 64, per_regime: int = 20, k: int = 10, seed: int = 700, nlist: int = 64, candidate_budget: int = 1200, correlations: Optional[List[str]] = None) -> List[Dict]:
    """Run positive/independent/negative policy-vector correlation stress tests."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if correlations is None:
        correlations = ["positive", "independent", "negative"]
    keep = ["ExactSecure", "PreFilter-Exact", "ScalarOnly", "PostFilter-IVF", "AdaptivePostFilter-IVF", "PreFilter-IVF", "PACER-A", "PACER-X"]
    rows: List[Dict] = []
    for i, corr in enumerate(correlations):
        summary = run_suite(out_dir / f"correlation_{corr}", n=n, dim=dim, per_regime=per_regime, k=k, seed=seed + 17*i, nlist=nlist, candidate_budget=candidate_budget, correlation=corr)
        for method in keep:
            v = summary["overall"][method]
            rows.append({"correlation": corr, "method": method, "n": int(n), "queries": int(summary["queries"]), "recall_at_k": v["recall_at_k"], "secure_topk_exact": v["secure_topk_exact"], "latency_ms": v["latency_ms"], "candidates": v["candidates"], "raw_candidates": v["raw_candidates"], "certified_fraction": v["certified_fraction"], "certified_sound_fraction": v["certified_sound_fraction"]})
    with (out_dir / "correlation_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    with (out_dir / "correlation_sweep.json").open("w") as f:
        json.dump(rows, f, indent=2)
    return rows
