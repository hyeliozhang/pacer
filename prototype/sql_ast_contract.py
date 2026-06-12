"""SQL-AST visibility contract for PACER.

This executable supplement makes the paper's SQL claim concrete without
pretending to be a full SQL parser. A DBMS parser/optimizer can compile SQL
into a boolean residual predicate and safe physical summaries. This file
implements a deterministic predicate AST with AND/OR/NOT, IN/NOT IN, range
atoms, bit-intersection predicates, semijoin-like group membership,
antijoin-like deny sets, and live-at-epoch atoms. PACER uses summaries only for
no-false-negative pruning and always evaluates the AST at the row verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple
import argparse
import csv
import json
import math
import time

import numpy as np

from trustvql import (
    INF_EPOCH,
    IVFIndex,
    PolicyVectorDataset,
    generate_dataset,
    _normalize,
    _topk_from_scores,
    exact_ordered,
    failure_mode,
    recall_at_k,
)


def _insert_epoch(ds: PolicyVectorDataset) -> np.ndarray:
    return getattr(ds, "insert_epoch", np.zeros(ds.n, dtype=np.int32))


def bitmask(vals: Iterable[int]) -> int:
    m = 0
    for v in vals:
        m |= 1 << int(v)
    return int(m)


@dataclass(frozen=True)
class Atom:
    field: str
    op: str
    value: object

    def eval(self, ctx: "RelContext", ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=bool)
        if self.field == "live_epoch":
            e = int(self.value)
            return (_insert_epoch(ctx.ds)[ids] <= e) & (ctx.ds.delete_epoch[ids] > e)
        vals = ctx.field_array(self.field)[ids]
        if self.op == "in":
            return np.isin(vals, np.asarray(list(self.value), dtype=vals.dtype))
        if self.op == "notin":
            return ~np.isin(vals, np.asarray(list(self.value), dtype=vals.dtype))
        if self.op == "<=":
            return vals <= self.value
        if self.op == ">=":
            return vals >= self.value
        if self.op == "between":
            lo, hi = self.value
            return (vals >= lo) & (vals <= hi)
        if self.op == "bitany":
            return (vals.astype(np.int64) & int(self.value)) != 0
        if self.op == "eq":
            return vals == self.value
        if self.op == "neq":
            return vals != self.value
        raise ValueError(f"unsupported op {self.op}")

    def may_satisfy(self, summary: "UnitSummary") -> bool:
        if self.field == "live_epoch":
            e = int(self.value)
            return not (summary.min_insert_epoch > e or summary.max_delete_epoch <= e)
        if self.field in {"tenant", "region", "dtype", "auth_group", "deny_group"}:
            mask = summary.bitsets[self.field]
            if self.op == "in":
                return (mask & bitmask(self.value)) != 0
            if self.op == "eq":
                return (mask & (1 << int(self.value))) != 0
            if self.op == "notin":
                denied = bitmask(self.value)
                return (mask & ~denied) != 0
            if self.op == "neq":
                return (mask & ~(1 << int(self.value))) != 0
            return True
        if self.field == "sensitivity":
            if self.op == "<=":
                return summary.min_sensitivity <= int(self.value)
            if self.op == ">=":
                return summary.max_sensitivity >= int(self.value)
            if self.op == "between":
                lo, hi = self.value
                return not (summary.max_sensitivity < lo or summary.min_sensitivity > hi)
            if self.op == "eq":
                return summary.min_sensitivity <= int(self.value) <= summary.max_sensitivity
            return True
        if self.field == "quality_bin":
            if self.op == "<=":
                return summary.min_quality_bin <= int(self.value)
            if self.op == ">=":
                return summary.max_quality_bin >= int(self.value)
            if self.op == "between":
                lo, hi = self.value
                return not (summary.max_quality_bin < lo or summary.min_quality_bin > hi)
            return True
        if self.field == "provenance":
            if self.op == "bitany":
                return (summary.provenance_union & int(self.value)) != 0
            return True
        return True

    def sql(self) -> str:
        if self.field == "live_epoch":
            return f"delete_epoch > {int(self.value)}"
        if self.op in {"in", "notin"}:
            vals = ",".join(str(int(v)) for v in self.value)
            return f"{self.field} {'NOT IN' if self.op == 'notin' else 'IN'} ({vals})"
        if self.op == "bitany":
            return f"({self.field} & {int(self.value)}) <> 0"
        if self.op == "between":
            lo, hi = self.value
            return f"{self.field} BETWEEN {lo} AND {hi}"
        return f"{self.field} {self.op} {self.value}"


@dataclass(frozen=True)
class And:
    parts: Tuple[object, ...]
    def eval(self, ctx: "RelContext", ids: np.ndarray) -> np.ndarray:
        out = np.ones(len(ids), dtype=bool)
        for p in self.parts:
            out &= p.eval(ctx, ids)
            if not out.any(): break
        return out
    def may_satisfy(self, summary: "UnitSummary") -> bool:
        return all(p.may_satisfy(summary) for p in self.parts)
    def sql(self) -> str:
        return "(" + " AND ".join(p.sql() for p in self.parts) + ")"


@dataclass(frozen=True)
class Or:
    parts: Tuple[object, ...]
    def eval(self, ctx: "RelContext", ids: np.ndarray) -> np.ndarray:
        out = np.zeros(len(ids), dtype=bool)
        for p in self.parts:
            out |= p.eval(ctx, ids)
            if out.all(): break
        return out
    def may_satisfy(self, summary: "UnitSummary") -> bool:
        return any(p.may_satisfy(summary) for p in self.parts)
    def sql(self) -> str:
        return "(" + " OR ".join(p.sql() for p in self.parts) + ")"


@dataclass(frozen=True)
class Not:
    part: object
    def eval(self, ctx: "RelContext", ids: np.ndarray) -> np.ndarray:
        return ~self.part.eval(ctx, ids)
    def may_satisfy(self, summary: "UnitSummary") -> bool:
        # Safe residual behavior: negation does not drive pruning unless the DBMS
        # supplies an exact complement summary.  This prevents false negatives.
        return True
    def sql(self) -> str:
        return f"(NOT {self.part.sql()})"


@dataclass
class RelContext:
    ds: PolicyVectorDataset
    auth_group: np.ndarray
    deny_group: np.ndarray
    quality_bin: np.ndarray
    def field_array(self, field: str) -> np.ndarray:
        if field == "tenant": return self.ds.tenant
        if field == "region": return self.ds.region
        if field == "dtype": return self.ds.dtype
        if field == "sensitivity": return self.ds.sensitivity
        if field == "provenance": return self.ds.provenance
        if field == "auth_group": return self.auth_group
        if field == "deny_group": return self.deny_group
        if field == "quality_bin": return self.quality_bin
        raise KeyError(field)


@dataclass
class UnitSummary:
    bitsets: Dict[str, int]
    min_sensitivity: int
    max_sensitivity: int
    min_quality_bin: int
    max_quality_bin: int
    provenance_union: int
    min_insert_epoch: int
    max_delete_epoch: int


class RelationalSummaryIndex:
    def __init__(self, ctx: RelContext, index: IVFIndex):
        self.ctx = ctx
        self.index = index
        ins = _insert_epoch(ctx.ds)
        self.summaries: List[UnitSummary] = []
        for ids in index.cluster_ids:
            if ids.size == 0:
                self.summaries.append(UnitSummary({k: 0 for k in ["tenant","region","dtype","auth_group","deny_group"]}, 999, -1, 999, -1, 0, INF_EPOCH, -1)); continue
            self.summaries.append(UnitSummary(
                bitsets={
                    "tenant": bitmask(np.unique(ctx.ds.tenant[ids])),
                    "region": bitmask(np.unique(ctx.ds.region[ids])),
                    "dtype": bitmask(np.unique(ctx.ds.dtype[ids])),
                    "auth_group": bitmask(np.unique(ctx.auth_group[ids])),
                    "deny_group": bitmask(np.unique(ctx.deny_group[ids])),
                },
                min_sensitivity=int(np.min(ctx.ds.sensitivity[ids])),
                max_sensitivity=int(np.max(ctx.ds.sensitivity[ids])),
                min_quality_bin=int(np.min(ctx.quality_bin[ids])),
                max_quality_bin=int(np.max(ctx.quality_bin[ids])),
                provenance_union=int(np.bitwise_or.reduce(ctx.ds.provenance[ids].astype(np.int64))),
                min_insert_epoch=int(np.min(ins[ids])),
                max_delete_epoch=int(np.max(ctx.ds.delete_epoch[ids])),
            ))
    def may_clusters(self, expr: object) -> np.ndarray:
        return np.asarray([expr.may_satisfy(s) for s in self.summaries], dtype=bool)


def make_rel_context(ds: PolicyVectorDataset, seed: int = 1207) -> RelContext:
    rng = np.random.default_rng(seed)
    auth_group = ((ds.tenant.astype(np.int32) * 3 + ds.region.astype(np.int32) + rng.integers(0, 3, ds.n)) % 32).astype(np.int16)
    deny_group = ((ds.dtype.astype(np.int32) * 5 + ds.sensitivity.astype(np.int32) + rng.integers(0, 2, ds.n)) % 19).astype(np.int16)
    q = np.quantile(ds.quality, [0.2, 0.4, 0.6, 0.8])
    quality_bin = np.searchsorted(q, ds.quality, side="right").astype(np.int16)
    return RelContext(ds, auth_group, deny_group, quality_bin)


def build_queries(ctx: RelContext, nqueries: int = 160, seed: int = 811) -> List[Dict]:
    rng = np.random.default_rng(seed); ds = ctx.ds; out: List[Dict] = []
    kinds = ["dnf", "semijoin", "antijoin", "range", "mixed"]
    all_t = np.unique(ds.tenant); all_r = np.unique(ds.region); all_d = np.unique(ds.dtype)
    attempts = 0
    while len(out) < nqueries:
        attempts += 1
        if attempts > nqueries * 6000: raise RuntimeError("could not generate enough SQL-AST queries")
        src = int(rng.integers(0, ds.n)); epoch = int(rng.integers(3, 9))
        qvec = _normalize(ds.vectors[src] + rng.normal(0, 0.12, ds.dim).astype(np.float32))[0]
        kind = kinds[len(out) % len(kinds)]
        live = Atom("live_epoch", "eq", epoch)
        prov_bits = int(ds.provenance[src]) or 1
        possible = [i for i in range(16) if prov_bits & (1 << i)] or [0]
        ptag = 1 << int(rng.choice(possible))
        if kind == "dnf":
            tset=set(int(x) for x in rng.choice(all_t,size=min(5,len(all_t)),replace=False)); tset.add(int(ds.tenant[src]))
            rset=set(int(x) for x in rng.choice(all_r,size=min(3,len(all_r)),replace=False)); rset.add(int(ds.region[src]))
            dset=set(int(x) for x in rng.choice(all_d,size=min(2,len(all_d)),replace=False)); dset.add(int(ds.dtype[src]))
            expr=And((Or((And((Atom("tenant","in",tuple(sorted(tset))),Atom("region","in",tuple(sorted(rset))))),And((Atom("dtype","in",tuple(sorted(dset))),Atom("sensitivity","<=",int(max(1,ds.sensitivity[src]))))))),Atom("provenance","bitany",ptag),live))
        elif kind == "semijoin":
            groups=set(int(x) for x in rng.choice(np.arange(32),size=6,replace=False)); groups.add(int(ctx.auth_group[src]))
            expr=And((Atom("auth_group","in",tuple(sorted(groups))),Atom("provenance","bitany",ptag),Atom("sensitivity","<=",3),live))
        elif kind == "antijoin":
            denied=set(int(x) for x in rng.choice(np.arange(19),size=4,replace=False))
            tset=set(int(x) for x in rng.choice(all_t,size=min(8,len(all_t)),replace=False)); tset.add(int(ds.tenant[src]))
            expr=And((Atom("tenant","in",tuple(sorted(tset))),Atom("deny_group","notin",tuple(sorted(denied))),Atom("sensitivity","<=",2),live))
        elif kind == "range":
            qb=int(ctx.quality_bin[src])
            expr=And((Atom("quality_bin",">=",max(0,qb-1)),Atom("quality_bin","<=",min(4,qb+1)),Atom("region","in",(int(ds.region[src]),)),live))
        else:
            expr=And((Or((Atom("tenant","eq",int(ds.tenant[src])),Atom("region","eq",int(ds.region[src])),Atom("provenance","bitany",prov_bits))),Not(Atom("deny_group","in",(int(ctx.deny_group[src]),))),live))
        ids=np.arange(ds.n,dtype=np.int64); visible=expr.eval(ctx,ids)
        if int(visible.sum())>=10:
            out.append({"expr":expr,"qvec":qvec.astype(np.float32),"kind":kind,"epoch":epoch,"visible_count":int(visible.sum()),"sql":expr.sql()})
    return out


def exact_ast(ctx: RelContext, q: Mapping, k: int) -> Dict:
    t0=time.perf_counter(); ids_all=np.arange(ctx.ds.n,dtype=np.int64); visible=q["expr"].eval(ctx,ids_all); ids=ids_all[visible]
    scores=ctx.ds.vectors[ids] @ q["qvec"] if ids.size else np.empty(0,dtype=np.float32)
    top_ids,top_scores=_topk_from_scores(ids,scores,k)
    return {"ids":top_ids,"scores":top_scores,"latency_ms":(time.perf_counter()-t0)*1000,"candidates":int(ids.size),"raw_candidates":int(ctx.ds.n),"policy_checks":int(ctx.ds.n),"certified":True,"visited_cluster_count":0}


def postfilter_ast(ctx: RelContext, index: IVFIndex, q: Mapping, k: int, nprobe: int=8, budget: int=4800) -> Dict:
    t0=time.perf_counter(); order=np.argsort(-(index.centroids @ q["qvec"]))[:min(nprobe,index.nlist)]
    raw_pool=np.concatenate([index.cluster_ids[int(c0)] for c0 in order]).astype(np.int64) if len(order) else np.empty(0,dtype=np.int64)
    if raw_pool.size>int(budget):
        raw_scores=ctx.ds.vectors[raw_pool] @ q["qvec"]; sel=np.argpartition(-raw_scores,int(budget)-1)[:int(budget)]; raw=raw_pool[sel]
    else:
        raw=raw_pool
    ok=q["expr"].eval(ctx,raw) if raw.size else np.empty(0,dtype=bool); legal=raw[ok]
    scores=ctx.ds.vectors[legal] @ q["qvec"] if legal.size else np.empty(0,dtype=np.float32); top_ids,top_scores=_topk_from_scores(legal,scores,k)
    return {"ids":top_ids,"scores":top_scores,"latency_ms":(time.perf_counter()-t0)*1000,"candidates":int(legal.size),"raw_candidates":int(raw.size),"policy_checks":int(raw.size),"certified":False,"visited_cluster_count":int(len(order))}


def pacer_ast(ctx: RelContext, index: IVFIndex, relidx: RelationalSummaryIndex, q: Mapping, k: int, budget: int=4800, force_certify: bool=False) -> Dict:
    t0=time.perf_counter(); qvec=q["qvec"]; upper=index.cluster_upper_bounds(qvec); may=relidx.may_clusters(q["expr"]); order=np.argsort(-upper)
    processed=np.zeros(index.nlist,dtype=bool); top_ids=np.empty(0,dtype=np.int64); top_scores=np.empty(0,dtype=np.float32)
    raw=checks=legal_total=visited=0; certified=False; gap=0.0; row_budget=ctx.ds.n if force_certify else int(budget); partial=False
    for c0 in order:
        c=int(c0)
        if not may[c]:
            processed[c]=True; continue
        ids_full=index.cluster_ids[c]
        if not force_certify and raw>=row_budget: break
        if not force_certify and raw+int(ids_full.size)>row_budget:
            room=max(0,row_budget-raw)
            if room<=0: break
            local_scores=ctx.ds.vectors[ids_full] @ qvec; take=np.argpartition(-local_scores,room-1)[:room]; ids=ids_full[take]; partial=True
        else:
            ids=ids_full; processed[c]=True
        visited += 1; raw += int(ids.size); checks += int(ids.size)
        ok=q["expr"].eval(ctx,ids); legal=ids[ok]
        if legal.size:
            scores=ctx.ds.vectors[legal] @ qvec
            mid=np.concatenate([top_ids,legal.astype(np.int64)]) if top_ids.size else legal.astype(np.int64)
            ms=np.concatenate([top_scores,scores.astype(np.float32)]) if top_scores.size else scores.astype(np.float32)
            top_ids,top_scores=_topk_from_scores(mid,ms,k); legal_total += int(legal.size)
        if partial: break
        rem=np.flatnonzero((~processed)&may); maxub=float(np.max(upper[rem])) if rem.size else -math.inf
        if top_scores.size>=k:
            kth=float(top_scores[-1]); gap=max(0.0,maxub-kth) if math.isfinite(maxub) else 0.0
            if kth>=maxub: certified=True; break
        elif rem.size==0:
            certified=True; break
    if not certified and not np.any((~processed)&may): certified=True; gap=0.0
    return {"ids":top_ids,"scores":top_scores,"latency_ms":(time.perf_counter()-t0)*1000,"candidates":int(legal_total),"raw_candidates":int(raw),"policy_checks":int(checks),"certified":bool(certified),"visited_cluster_count":int(visited),"certificate_gap":float(gap),"summary_pruned_clusters":int(np.sum(~may))}


def run_sql_ast_suite(out_dir: Path, n: int=9000, dim: int=64, nqueries: int=160, seed: int=811, k: int=10, nlist: int=64, budget: int=4800) -> Dict:
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    ds=generate_dataset(n=n,dim=dim,true_clusters=64,tenants=24,regions=6,dtypes=5,prov_tags=12,delete_rate=0.14,tenant_cluster_correlation=0.70,seed=seed)
    ctx=make_rel_context(ds,seed=seed+1); index=IVFIndex(ds,nlist=nlist,seed=seed+2,iters=4); relidx=RelationalSummaryIndex(ctx,index); queries=build_queries(ctx,nqueries=nqueries,seed=seed+3)
    methods={"ExactSQL-AST":lambda q: exact_ast(ctx,q,k),"PostFilter-SQL-AST":lambda q: postfilter_ast(ctx,index,q,k,nprobe=8,budget=budget),"PACER-A-SQL-AST":lambda q: pacer_ast(ctx,index,relidx,q,k,budget=budget,force_certify=False),"PACER-X-SQL-AST":lambda q: pacer_ast(ctx,index,relidx,q,k,budget=ds.n,force_certify=True)}
    rows=[]; qrows=[]
    for qi,q in enumerate(queries):
        gold=methods["ExactSQL-AST"](q); qrows.append({"query_id":qi,"kind":q["kind"],"epoch":q["epoch"],"visible_count":q["visible_count"],"sql_shape":q["sql"][:240]})
        for name,fn in methods.items():
            res=fn(q); exact=int(exact_ordered(gold["ids"],res["ids"],k=k)); cert=int(bool(res.get("certified",False)))
            rows.append({"query_id":qi,"kind":q["kind"],"method":name,"n":int(ds.n),"dim":int(ds.dim),"visible_count":int(q["visible_count"]),"recall_at_k":recall_at_k(gold["ids"],res["ids"],k=k),"secure_topk_exact":exact,"failure_mode":failure_mode(gold["ids"],res["ids"],k=k),"latency_ms":float(res["latency_ms"]),"candidates":int(res["candidates"]),"raw_candidates":int(res["raw_candidates"]),"policy_checks":int(res["policy_checks"]),"visited_cluster_count":int(res.get("visited_cluster_count",0)),"certified":cert,"certificate_false_positive":int(cert and not exact),"summary_pruned_clusters":int(res.get("summary_pruned_clusters",0))})
    for path,data in [(out_dir/"sql_ast_metrics.csv",rows),(out_dir/"sql_ast_queries.csv",qrows)]:
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    summary={"n":int(ds.n),"dim":int(ds.dim),"queries":len(queries),"configured_raw_cap":int(budget),"methods":{},"by_kind":{},"claim_checks":{}}
    for m in methods:
        xs=[r for r in rows if r["method"]==m]
        summary["methods"][m]={"recall_at_k":float(np.mean([r["recall_at_k"] for r in xs])),"secure_topk_exact":float(np.mean([r["secure_topk_exact"] for r in xs])),"certified_fraction":float(np.mean([r["certified"] for r in xs])),"certificate_errors":int(sum(r["certificate_false_positive"] for r in xs)),"median_latency_ms":float(np.median([r["latency_ms"] for r in xs])),"raw_candidates":float(np.mean([r["raw_candidates"] for r in xs])),"policy_checks":float(np.mean([r["policy_checks"] for r in xs])),"visited_cluster_count":float(np.mean([r["visited_cluster_count"] for r in xs])),"summary_pruned_clusters":float(np.mean([r["summary_pruned_clusters"] for r in xs]))}
    for kind in sorted(set(q["kind"] for q in queries)):
        summary["by_kind"][kind]={}
        for m in methods:
            xs=[r for r in rows if r["method"]==m and r["kind"]==kind]
            summary["by_kind"][kind][m]={"recall_at_k":float(np.mean([r["recall_at_k"] for r in xs])),"secure_topk_exact":float(np.mean([r["secure_topk_exact"] for r in xs]))}
    summary["claim_checks"]={"pacer_x_sql_ast_exact_failures":int(sum(1 for r in rows if r["method"]=="PACER-X-SQL-AST" and r["secure_topk_exact"]!=1)),"pacer_sql_ast_certificate_false_positives":int(sum(r["certificate_false_positive"] for r in rows if r["method"].startswith("PACER"))),"pacer_a_improves_postfilter_recall":bool(summary["methods"]["PACER-A-SQL-AST"]["recall_at_k"]>summary["methods"]["PostFilter-SQL-AST"]["recall_at_k"]),"pacer_a_improves_postfilter_exactness":bool(summary["methods"]["PACER-A-SQL-AST"]["secure_topk_exact"]>summary["methods"]["PostFilter-SQL-AST"]["secure_topk_exact"])}
    with (out_dir/"sql_ast_summary.json").open("w") as f: json.dump(summary,f,indent=2)
    return summary


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=None); ap.add_argument("--n",type=int,default=9000); ap.add_argument("--queries",type=int,default=160); ap.add_argument("--budget",type=int,default=4800); args=ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "sql_ast"
    print(json.dumps(run_sql_ast_suite(out,n=args.n,nqueries=args.queries,budget=args.budget),indent=2))

if __name__=="__main__": main()
