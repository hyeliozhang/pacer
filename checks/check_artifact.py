#!/usr/bin/env python3
"""Consistency checks for the PACER artifact.

The checker intentionally uses only the files in this repository. It validates
that the shipped summaries, tables, public data inputs, and executable scripts
are present and that the main safety/certification counters support the paper's
reported claims.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []
REPORT: dict[str, Any] = {}


def fail(message: str) -> None:
    ERRORS.append(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def path(rel: str) -> Path:
    return ROOT / rel


def require_file(rel: str) -> Path:
    p = path(rel)
    require(p.is_file() and p.stat().st_size > 0, f"missing or empty file: {rel}")
    return p


def load_json(rel: str) -> Any:
    p = require_file(rel)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        fail(f"invalid JSON in {rel}: {exc}")
        return {}


def rows_in_csv(rel: str) -> int:
    p = require_file(rel)
    try:
        with p.open(newline="", encoding="utf-8") as fh:
            return max(0, sum(1 for _ in csv.DictReader(fh)))
    except Exception as exc:  # pragma: no cover - diagnostic path
        fail(f"invalid CSV in {rel}: {exc}")
        return 0


def metric(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    fail(f"missing metric {names} in row with keys {sorted(row)}")
    return float("nan")


def all_zero(mapping: dict[str, Any], keys: list[str], prefix: str) -> None:
    for key in keys:
        require(float(mapping.get(key, 0)) == 0.0, f"{prefix}.{key} is nonzero")


def validate_manifest() -> None:
    required = [
        "README.md",
        "ARTIFACT_README.md",
        "EVIDENCE.md",
        "REPRODUCIBILITY.md",
        "DATA_LICENSE.md",
        "LICENSE",
        "Dockerfile",
        "requirements.txt",
        "environment.lock.txt",
        "reproduce_checks.sh",
        "reproduce_default_workload.sh",
        "reproduce_scale_frontier.sh",
        "prototype/trustvql.py",
        "prototype/access_path_adapter.py",
        "prototype/run_experiments_chunked.py",
        "prototype/run_scale_frontier.py",
        "prototype/make_v16_tables.py",
        "prototype/plot_results.py",
        "tests/test_invariants.py",
        "figures/table_main_results.tex",
        "figures/table_efficiency_scalability.tex",
        "figures/table_deep_checks.tex",
        "figures/fig_executor_contract.pdf",
        "figures/fig_semantics_architecture.pdf",
        "data/external/pendigits/SHA256SUMS",
    ]
    for rel in required:
        require_file(rel)


def validate_default_workload() -> None:
    summary = load_json("results/summary_n15000.json")
    require(summary.get("n") == 15000, "default workload n must be 15000")
    require(summary.get("queries") == 320, "default workload must have 320 queries")
    require(summary.get("query_method_rows") == 5120, "default workload row count mismatch")
    require(rows_in_csv("results/metrics_n15000.csv") == 5120, "metrics_n15000.csv row count mismatch")

    overall = summary.get("overall", {})
    pacer_a = overall.get("PACER-A", {})
    pacer_c = overall.get("PACER-C", {})
    pacer_x = overall.get("PACER-X", {})
    post = overall.get("PostFilter-IVF", {})

    require(metric(pacer_a, "recall_at_k") > metric(post, "recall_at_k"), "PACER-A recall must exceed post-filtering")
    require(metric(pacer_a, "secure_topk_exact") > metric(post, "secure_topk_exact"), "PACER-A exactness must exceed post-filtering")
    require(metric(pacer_a, "raw_candidates") < metric(post, "raw_candidates"), "PACER-A raw identifiers must be below post-filtering")
    require(metric(pacer_a, "recall_at_k") >= 0.98, "PACER-A default recall below expected range")
    require(metric(pacer_a, "secure_topk_exact") >= 0.88, "PACER-A default exactness below expected range")
    require(metric(pacer_c, "secure_topk_exact") == 1.0, "PACER-C must be ordered-exact")
    require(metric(pacer_x, "secure_topk_exact") == 1.0, "PACER-X must be ordered-exact")

    checks = summary.get("claim_checks", {})
    all_zero(checks, [
        "safe_policy_deletion_tenant_violations_total",
        "pacer_certificate_false_positive_total",
        "pacer_certify_ordered_failures_total",
    ], "summary_n15000.claim_checks")

    REPORT["default"] = {
        "n": summary.get("n"),
        "queries": summary.get("queries"),
        "pacer_a_recall_at_k": round(metric(pacer_a, "recall_at_k"), 3),
        "pacer_a_ordered_exactness": round(metric(pacer_a, "secure_topk_exact"), 3),
        "pacer_a_raw_identifiers": round(metric(pacer_a, "raw_candidates")),
        "postfilter_recall_at_k": round(metric(post, "recall_at_k"), 3),
        "postfilter_ordered_exactness": round(metric(post, "secure_topk_exact"), 3),
        "postfilter_raw_identifiers": round(metric(post, "raw_candidates")),
    }


def validate_scale() -> None:
    large = load_json("results/large_scale/large_scale_60k_summary.json")
    require(large.get("n") == 60000, "60K scale summary has wrong n")
    methods = large.get("methods_summary", {})
    pacer_a = methods.get("PACER-A", {})
    pacer_x = methods.get("PACER-X", {})
    post = methods.get("PostFilter-IVF", {})
    require(metric(pacer_a, "raw_candidates_mean") < metric(post, "raw_candidates_mean"), "60K PACER-A raw identifiers must be below post-filtering")
    require(metric(pacer_x, "ordered_exact_secure_topk") == 1.0, "60K PACER-X must be exact")
    checks = large.get("claim_checks", {})
    all_zero(checks, ["safe_method_violations_total", "pacer_x_exact_failures", "pacer_c_exact_failures"], "large_scale.claim_checks")
    require(checks.get("pacer_a_raw_less_than_postfilter") is True, "60K raw-work claim must hold")

    frontier = load_json("results/scale_frontier/scale_frontier_summary.json")
    ns = {item.get("n"): item for item in frontier.get("summaries", [])}
    require({120000, 240000}.issubset(ns), "frontier summary must include 120K and 240K")
    frontier_report = {}
    for n in [120000, 240000]:
        item = ns[n]
        methods = item.get("methods_summary", {})
        checks = item.get("claim_checks", {})
        require(metric(methods.get("PACER-X", {}), "ordered_exact_secure_topk") == 1.0, f"{n} PACER-X must be exact")
        require(checks.get("pacer_x_exact_failures") == 0, f"{n} PACER-X exact failures")
        require(checks.get("safe_violations_total") == 0, f"{n} safety violations")
        require(checks.get("pacer_a_raw_less_than_postfilter") is True, f"{n} raw-work claim must hold")
        frontier_report[str(n)] = {
            "pacer_a_raw_identifiers": round(metric(methods.get("PACER-A", {}), "raw_candidates_mean")),
            "pacer_x_exact": metric(methods.get("PACER-X", {}), "ordered_exact_secure_topk"),
        }

    REPORT["scale"] = {
        "60k_pacer_a_recall": round(metric(pacer_a, "recall_at_k"), 3),
        "60k_pacer_a_raw_identifiers": round(metric(pacer_a, "raw_candidates_mean")),
        "frontier": frontier_report,
    }


def validate_deep_checks() -> None:
    proof = load_json("results/proof_replay/proof_replay_summary.json")
    all_zero(proof.get("claim_checks", {}), [
        "summary_false_negative_clusters_total",
        "certified_proof_failures_without_oracle",
        "certified_oracle_exact_failures",
        "pacer_x_oracle_exact_failures",
    ], "proof_replay.claim_checks")

    model = load_json("results/model_check/model_check_summary.json")
    all_zero(model.get("claim_checks", {}), [
        "summary_false_negative_clusters",
        "certified_proof_failures_without_oracle",
        "pacer_x_exact_failures",
        "safe_output_violations",
        "pacer_a_certified_exact_failures",
    ], "model_check.claim_checks")

    adapter = load_json("results/adapter_contract_summary.json")
    require(adapter.get("passed") is True, "adapter contract must pass")
    all_zero(adapter.get("failures", {}), [
        "visible_rows_in_pruned_units",
        "slice_omissions",
        "bound_violations",
        "witness_failures",
    ], "adapter.failures")

    REPORT["deep_checks"] = {
        "proof_replay_rows": proof.get("rows"),
        "model_check_cases": model.get("enumerated_policy_cases"),
        "adapter_checked_units": adapter.get("checked_units"),
        "adapter_witness_rows": adapter.get("witness_rows_checked"),
    }


def validate_sql_and_updates() -> None:
    sql = load_json("results/sql_policy/sql_policy_summary.json")
    sql_methods = sql.get("methods", {})
    require(metric(sql_methods.get("PACER-X-SQL", {}), "secure_topk_exact") == 1.0, "PACER-X-SQL must be exact")
    require(metric(sql_methods.get("PACER-A-SQL", {}), "raw_candidates") == 4800.0, "PACER-A-SQL raw cap mismatch")

    ast = load_json("results/sql_ast/sql_ast_summary.json")
    ast_methods = ast.get("methods", {})
    require(ast.get("configured_raw_cap") == 4800, "SQL-AST raw cap mismatch")
    require(metric(ast_methods.get("PACER-X-SQL-AST", {}), "secure_topk_exact") == 1.0, "PACER-X-SQL-AST must be exact")
    require(metric(ast_methods.get("PACER-X-SQL-AST", {}), "certificate_errors") == 0.0, "PACER-X-SQL-AST certificate errors")

    dynamic = load_json("results/dynamic_overlay/dynamic_overlay_summary.json")
    all_zero(dynamic.get("claim_checks", {}), [
        "delta_pacer_exact_failures",
        "delta_pacer_recall_failures",
        "fresh_rebuild_exact_failures",
    ], "dynamic_overlay.claim_checks")

    snapshot = load_json("results/snapshot_stream/snapshot_stream_summary.json")
    all_zero(snapshot.get("claim_checks", {}), [
        "safe_future_deleted_policy_tenant_violations",
        "pacer_x_exact_failures",
    ], "snapshot_stream.claim_checks")

    REPORT["sql_updates"] = {
        "sql_queries": sql.get("queries"),
        "sql_ast_queries": ast.get("queries"),
        "dynamic_epochs": dynamic.get("epochs"),
        "snapshot_rows": snapshot.get("rows"),
    }


def validate_public_data_and_robustness() -> None:
    blind = load_json("results/blind_robustness/blind_robustness_summary.json")
    checks = blind.get("claim_checks", {})
    require(checks.get("config_count") == 27, "blind robustness config count mismatch")
    require(checks.get("all_safe_method_violations") == 0, "blind robustness safety violations")
    require(checks.get("pacer_a_beats_postfilter_exact_configs") == 27, "PACER-A must beat post-filtering exactness on all blind configs")
    require(checks.get("pacer_a_beats_postfilter_raw_configs") == 27, "PACER-A must beat post-filtering raw work on all blind configs")
    require(checks.get("pacer_c_exact_configs") == 27, "PACER-C exact blind configs mismatch")
    require(checks.get("pacer_x_exact_configs") == 27, "PACER-X exact blind configs mismatch")

    external = load_json("results/external_summary.json")
    datasets = external.get("datasets", {})
    require({"digits", "breast-cancer", "wine", "diabetes"}.issubset(datasets), "missing scikit-learn dataset summary")
    for name, item in datasets.items():
        methods = item.get("methods", {})
        require(metric(methods.get("PACER-X", {}), "secure_topk_exact") == 1.0, f"{name}: PACER-X must be exact")
        require(metric(methods.get("PACER-X", {}), "certificate_false_positive_total") == 0.0, f"{name}: PACER-X certificate false positives")

    pendigits = load_json("results/pendigits/summary_pendigits.json")
    pend_methods = pendigits.get("overall", {})
    require(metric(pend_methods.get("PACER-X", {}), "secure_topk_exact") == 1.0, "Pendigits PACER-X must be exact")

    sums = require_file("data/external/pendigits/SHA256SUMS")
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        p = path(rel) if rel.startswith("data/") else sums.parent / rel
        require(p.is_file(), f"missing Pendigits file listed in SHA256SUMS: {rel}")
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        require(actual == expected, f"SHA256 mismatch for {rel}")

    REPORT["robustness_public_data"] = {
        "blind_configs": blind.get("configs"),
        "external_datasets": sorted(datasets),
        "pendigits_queries": pendigits.get("queries"),
    }


def main() -> int:
    validate_manifest()
    validate_default_workload()
    validate_scale()
    validate_deep_checks()
    validate_sql_and_updates()
    validate_public_data_and_robustness()

    if ERRORS:
        print(json.dumps({"status": "fail", "errors": ERRORS, "report": REPORT}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"status": "pass", "report": REPORT}, indent=2, sort_keys=True))
    print("PACER artifact check: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
