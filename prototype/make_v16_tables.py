#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
RES = ROOT / 'results'

def esc(s): return s.replace('_','\\_')

def main():
    strong = json.load((RES/'strong_baselines'/'strong_baselines_summary.json').open())
    methods = ['PostFilter-IVF','PreFilter-IVF','GraphPostFilter','InlineGraph','PredicateGraph','ACORN-2Hop','FDANN-Beam','SeRF-Range','SIEVE-Collection','PACER-A','PACER-C','PACER-X']
    names = {'PostFilter-IVF':'Post-IVF','PreFilter-IVF':'Pre-IVF','GraphPostFilter':'Graph-post','InlineGraph':'Graph-in','PredicateGraph':'Pred-graph','ACORN-2Hop':'ACORN-2h','FDANN-Beam':'FDANN-beam','SeRF-Range':'SeRF-range','SIEVE-Collection':'SIEVE-coll.','PACER-A':'PACER-A','PACER-C':'PACER-C','PACER-X':'PACER-X'}
    lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Rec. & Exact & Raw & ms\\\\', '\\midrule']
    for m in methods:
        v = strong['methods'][m]
        lines.append(f"{names[m]} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['raw_candidates']:.0f} & {v['latency_ms']:.2f}\\\\")
    lines += ['\\bottomrule','\\end{tabular}']
    (FIG/'table_strong_baselines.tex').write_text('\n'.join(lines))

    ent_path = RES/'enterprise_stress'/'enterprise_stress_summary.json'
    if ent_path.exists():
        ent = json.load(ent_path.open())
        methods_e = ['PostFilter-IVF','PreFilter-IVF','BitmapSlice-Exact','PACER-A','PACER-C','PACER-X']
        names_e = {'PostFilter-IVF':'Post-IVF','PreFilter-IVF':'Pre-IVF','BitmapSlice-Exact':'Slice-exact','PACER-A':'PACER-A','PACER-C':'PACER-C','PACER-X':'PACER-X'}
        lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Rec. & Exact & Raw & Lat.\\\\', '\\midrule']
        for m in methods_e:
            v = ent['methods'][m]
            lines.append(f"{names_e[m]} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['raw_candidates']:.0f} & {v['latency_ms']:.2f}\\\\")
        lines += ['\\bottomrule','\\end{tabular}']
        (FIG/'table_enterprise_stress.tex').write_text('\n'.join(lines))

    cost = json.load((RES/'cost_accounting'/'cost_accounting_summary.json').open())
    methods2 = ['ExactSecure','PostFilter-IVF','BitmapSlice-Exact','PACER-A','PACER-C','PACER-X']
    names2 = {'ExactSecure':'Exact','PostFilter-IVF':'Post-IVF','BitmapSlice-Exact':'Slice-exact','PACER-A':'PACER-A','PACER-C':'PACER-C','PACER-X':'PACER-X'}
    lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Lat. & P95 & Raw & Checks\\\\', '\\midrule']
    for m in methods2:
        v = cost['methods'][m]
        lines.append(f"{names2[m]} & {v['latency_ms_mean']:.2f} & {v['latency_ms_p95']:.2f} & {v['raw_candidates_mean']:.0f} & {v['policy_checks_mean']:.0f}\\\\")
    lines += ['\\bottomrule','\\end{tabular}']
    (FIG/'table_cost_accounting.tex').write_text('\n'.join(lines))

    large_path = RES/'large_scale'/'large_scale_60k_summary.json'
    frontier_path = RES/'scale_frontier'/'scale_frontier_summary.json'
    mem_path_for_eff = RES/'memory_audit'/'memory_audit.json'
    summary_path_for_eff = RES/'summary_n15000.json'
    if large_path.exists() and mem_path_for_eff.exists() and summary_path_for_eff.exists():
        large = json.load(large_path.open())
        mem_eff = {int(r['n']): r for r in json.load(mem_path_for_eff.open())['rows']}
        summary_eff = json.load(summary_path_for_eff.open())['overall']
        frontier = json.load(frontier_path.open()) if frontier_path.exists() else {'summaries': []}
        frontier_by_n = {int(s['n']): s for s in frontier.get('summaries', [])}

        def scale_row(label, post, pacer_a, pacer_x, build_ms, bpr):
            return (f"{label} & {post['recall_at_k']:.3f}/{post['raw']:.0f} & "
                    f"{pacer_a['recall_at_k']:.3f}/{pacer_a['raw']:.0f} & {pacer_a['lat']:.1f} & "
                    f"{pacer_x['ordered_exact_secure_topk']:.3f}/{pacer_x['raw']:.0f} & {build_ms:.0f}/{bpr:.1f}" + r"\\")

        rows_eff = []
        rows_eff.append(scale_row(
            '15K',
            {'recall_at_k': summary_eff['PostFilter-IVF']['recall_at_k'], 'raw': summary_eff['PostFilter-IVF']['raw_candidates']},
            {'recall_at_k': summary_eff['PACER-A']['recall_at_k'], 'raw': summary_eff['PACER-A']['raw_candidates'], 'lat': summary_eff['PACER-A']['latency_ms']},
            {'ordered_exact_secure_topk': summary_eff['PACER-X']['secure_topk_exact'], 'raw': summary_eff['PACER-X']['raw_candidates']},
            mem_eff[15000]['index_build_ms'], mem_eff[15000]['logical_structural_bytes_per_record']))
        rows_eff.append(scale_row(
            '60K',
            {'recall_at_k': large['methods_summary']['PostFilter-IVF']['recall_at_k'], 'raw': large['methods_summary']['PostFilter-IVF']['raw_candidates_mean']},
            {'recall_at_k': large['methods_summary']['PACER-A']['recall_at_k'], 'raw': large['methods_summary']['PACER-A']['raw_candidates_mean'], 'lat': large['methods_summary']['PACER-A']['latency_ms_mean']},
            {'ordered_exact_secure_topk': large['methods_summary']['PACER-X']['ordered_exact_secure_topk'], 'raw': large['methods_summary']['PACER-X']['raw_candidates_mean']},
            mem_eff[60000]['index_build_ms'], mem_eff[60000]['logical_structural_bytes_per_record']))
        for n in [120000, 240000]:
            if n in frontier_by_n:
                fr = frontier_by_n[n]
                rows_eff.append(scale_row(
                    f'{n//1000}K',
                    {'recall_at_k': fr['methods_summary']['PostFilter-IVF']['recall_at_k'], 'raw': fr['methods_summary']['PostFilter-IVF']['raw_candidates_mean']},
                    {'recall_at_k': fr['methods_summary']['PACER-A']['recall_at_k'], 'raw': fr['methods_summary']['PACER-A']['raw_candidates_mean'], 'lat': fr['methods_summary']['PACER-A']['latency_ms_mean']},
                    {'ordered_exact_secure_topk': fr['methods_summary']['PACER-X']['ordered_exact_secure_topk'], 'raw': fr['methods_summary']['PACER-X']['raw_candidates_mean']},
                    fr['index_build_ms'], fr['index_overhead']['total_structural_bytes_per_record']))

        lines = [
            r'% Scale audit: columns combine recall/exactness with raw identifiers and build/metadata cost.',
            r'\begin{tabular}{@{}lccccc@{}}',
            r'\toprule',
            r'Scale &',
            r'\begin{tabular}[c]{@{}c@{}}PostFilter\\rec., raw\end{tabular} &',
            r'\begin{tabular}[c]{@{}c@{}}PACER-A\\rec., raw\end{tabular} &',
            r'\begin{tabular}[c]{@{}c@{}}A\\ms\end{tabular} &',
            r'\begin{tabular}[c]{@{}c@{}}PACER-X\\exact, raw\end{tabular} &',
            r'\begin{tabular}[c]{@{}c@{}}Build\\ms, B/row\end{tabular}\\',
            r'\midrule',
        ] + rows_eff + [r'\bottomrule', r'\end{tabular}']
        (FIG/'table_efficiency_scalability.tex').write_text('\n'.join(lines))

    # Also write a small metadata table for dynamic maintenance, using current dynamic summary.
    dyn = json.load((RES/'dynamic_overlay'/'dynamic_overlay_summary.json').open())
    lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Rec. & Exact & Raw & Rebuild\\\\', '\\midrule']
    for m in ['BaseOnly-Stale','DeltaPACER','FreshSlice-Rebuild']:
        v = dyn['methods'][m]
        lines.append(f"{m.replace('BaseOnly-Stale','Stale-base').replace('FreshSlice-Rebuild','Fresh-rebuild')} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['raw_checks']:.0f} & {v['rebuild_ms']:.1f}\\\\")
    lines += ['\\bottomrule','\\end{tabular}']
    (FIG/'table_dynamic_overlay.tex').write_text('\n'.join(lines))

    high_path = RES/'high_cardinality'/'high_cardinality_summary.json'
    if high_path.exists():
        high = json.load(high_path.open())
        methods3 = ['PostFilter-IVF','PreFilter-IVF','BitmapSlice-Exact','PACER-A','PACER-X']
        names3 = {'PostFilter-IVF':'Post-IVF','PreFilter-IVF':'Pre-IVF','BitmapSlice-Exact':'Slice-exact','PACER-A':'PACER-A','PACER-X':'PACER-X'}
        lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Rec. & Exact & Raw & Viol.\\\\', '\\midrule']
        for m in methods3:
            v = high['summary'][m]
            lines.append(f"{names3[m]} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['raw_candidates']:.0f} & {v['violations']}\\\\")
        lines += ['\\bottomrule','\\end{tabular}']
        (FIG/'table_high_cardinality.tex').write_text('\n'.join(lines))



    sqlast_path = RES/'sql_ast'/'sql_ast_summary.json'
    if sqlast_path.exists():
        ast = json.load(sqlast_path.open())
        methods4 = ['PostFilter-SQL-AST','PACER-A-SQL-AST','PACER-X-SQL-AST']
        names4 = {'PostFilter-SQL-AST':'Post-AST','PACER-A-SQL-AST':'PACER-A','PACER-X-SQL-AST':'PACER-X'}
        lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Rec. & Exact & Cert. & Raw\\\\', '\\midrule']
        for m in methods4:
            v = ast['methods'][m]
            lines.append(f"{names4[m]} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['certified_fraction']:.2f} & {v['raw_candidates']:.0f}\\\\")
        lines += ['\\bottomrule','\\end{tabular}']
        (FIG/'table_sql_ast.tex').write_text('\n'.join(lines))


        sql_path = RES/'sql_policy'/'sql_policy_summary.json'
        if sql_path.exists():
            frag = json.load(sql_path.open())
            combo = ['\\begin{tabular}{llrrrr}', '\\toprule', 'Workload & Method & Rec. & Exact & Cert. & Raw\\\\', '\\midrule']
            rows = [
                ('Frag.', 'Post-SQL', frag['methods']['PostFilter-SQL']),
                ('Frag.', 'PACER-A', frag['methods']['PACER-A-SQL']),
                ('Frag.', 'PACER-X', frag['methods']['PACER-X-SQL']),
                ('AST', 'Post-AST', ast['methods']['PostFilter-SQL-AST']),
                ('AST', 'PACER-A', ast['methods']['PACER-A-SQL-AST']),
                ('AST', 'PACER-X', ast['methods']['PACER-X-SQL-AST']),
            ]
            for workload, name, v in rows:
                combo.append(f"{workload} & {name} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['certified_fraction']:.2f} & {v['raw_candidates']:.0f}\\\\")
            combo += ['\\bottomrule','\\end{tabular}']
            (FIG/'table_sql_workloads.tex').write_text('\n'.join(combo))


    blind_path = RES/'blind_robustness'/'blind_robustness_summary.json'
    if blind_path.exists():
        blind = json.load(blind_path.open())
        methods_b = ['PostFilter-IVF','PreFilter-IVF','PACER-A','PACER-C','PACER-X']
        names_b = {'PostFilter-IVF':'Post-IVF','PreFilter-IVF':'Pre-IVF','PACER-A':'PACER-A','PACER-C':'PACER-C','PACER-X':'PACER-X'}
        lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Mean exact & Min exact & Mean raw & Viol.\\\\', '\\midrule']
        for m in methods_b:
            v = blind['methods'][m]
            lines.append(f"{names_b[m]} & {v['mean_exact']:.3f} & {v['min_exact_by_config']:.3f} & {v['mean_raw_candidates']:.0f} & {v['total_violations']}\\\\")
        lines += ['\\bottomrule','\\end{tabular}']
        (FIG/'table_blind_robustness.tex').write_text('\n'.join(lines))

# Scale and memory audit tables are appended by a second pass.
def make_v20_extra_tables():
    import json
    large_path = RES/'large_scale'/'large_scale_60k_summary.json'
    if large_path.exists():
        large = json.load(large_path.open())
        methods = ['PostFilter-IVF','PreFilter-IVF','PACER-A','PACER-C','PACER-X']
        names = {'PostFilter-IVF':'Post-IVF','PreFilter-IVF':'Pre-IVF','PACER-A':'PACER-A','PACER-C':'PACER-C','PACER-X':'PACER-X'}
        lines = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Method & Rec. & Exact & Raw & Lat.\\\\', '\\midrule']
        for m in methods:
            v = large['methods_summary'][m]
            lines.append(f"{names[m]} & {v['recall_at_k']:.3f} & {v['ordered_exact_secure_topk']:.3f} & {v['raw_candidates_mean']:.0f} & {v['latency_ms_mean']:.1f}\\\\")
        lines += ['\\bottomrule','\\end{tabular}']
        (FIG/'table_large_scale.tex').write_text('\n'.join(lines))
    mem_path = RES/'memory_audit'/'memory_audit.json'
    if mem_path.exists():
        mem = json.load(mem_path.open())['rows']
        lines = ['\\begin{tabular}{lrrr}', '\\toprule', 'Rows & Logical B/r & Python B/r & Build ms\\\\', '\\midrule']
        for r in mem:
            lines.append(f"{int(r['n'])} & {r['logical_structural_bytes_per_record']:.1f} & {r['python_object_bytes_per_record']:.1f} & {r['index_build_ms']:.0f}\\\\")
        lines += ['\\bottomrule','\\end{tabular}']
        (FIG/'table_memory_audit.tex').write_text('\n'.join(lines))
        # The combined efficiency/scalability table is written in main(), where both
        # the default instrumentation subset and the 60K scale gate are available.


if __name__ == '__main__':
    main()
    make_v20_extra_tables()
