# Evidence Map

This file maps PACER claims to concrete artifact files.

## Default workload

- Source: `results/summary_n15000.json`, `results/metrics_n15000.csv`.
- Main table: `figures/table_main_results.tex`.
- Claim: PACER-A improves secure recall and ordered exactness over post-filtering while verifying fewer raw identifiers; PACER-C and PACER-X are exact; safe methods return zero residual-policy, tenant, and epoch violations.

## Efficiency and scalability

- Source: `results/cost_accounting/cost_accounting_summary.json`, `results/large_scale/large_scale_60k_summary.json`, `results/scale_frontier/scale_frontier_summary.json`, `results/memory_audit/memory_audit.json`, and `results/scalability.csv`.
- Main table: `figures/table_efficiency_scalability.tex`.
- Claim: latency, raw identifiers, policy checks, 60K full-gate behavior, 120K/240K frontier-scale behavior, structural metadata, and build cost are reported as optimizer-visible quantities.

## Access-path adapter contract

- Source: `prototype/access_path_adapter.py` and `results/adapter_contract_summary.json`.
- Claim: any engine adapter must expose summary-possible units, policy-relevant row slices, score upper bounds, and row-fact witnesses; the checker recomputes facts and reports zero pruned-visible rows, slice omissions, unsound bounds, or witness failures.

## Certification and proof replay

- Source: `results/proof_replay/proof_replay_summary.json`, `results/model_check/model_check_summary.json`, `figures/table_certificates.tex`, and `figures/table_deep_checks.tex`.
- Claim: every certified PACER output is ordered exact; proof replay and finite-domain model checking find no summary or certificate counterexample.

## SQL and policy workloads

- Source: `results/sql_policy/sql_policy_summary.json`, `results/sql_ast/sql_ast_summary.json`, `figures/table_sql_workloads.tex`.
- Claim: SQL-fragment and SQL-AST policy workloads use the reported 4,800 raw-check cap for PACER-A; PACER-X is exact.

## Updates and deletion safety

- Source: `results/dynamic_overlay/dynamic_overlay_summary.json`, `results/snapshot_stream/snapshot_stream_summary.json`, `figures/table_deletion.tex`, and `figures/table_dynamic_overlay.tex`.
- Claim: stale and future postings cannot cross the row verifier in PACER, while unsafe controls expose deleted or missing rows.

## Robustness and public data

- Source: `results/blind_robustness/blind_robustness_summary.json`, `results/multiseed/multiseed_summary.csv`, `results/external_summary.json`, `results/pendigits/summary_pendigits.json`, `figures/table_external.tex`, and `figures/table_blind_robustness.tex`.
- Claim: public feature matrices, repeated seeds, frozen blind configurations, high-cardinality tenants, and enterprise ACL/deletion stress preserve the qualitative result.
