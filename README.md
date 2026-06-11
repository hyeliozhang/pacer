# PACER Artifact

This repository contains the reviewer-facing artifact for PACER, a proof-carrying executor contract for policy-constrained vector top-k query processing.

## Current closure

The artifact ties the main experimental claims to generated tables, JSON summaries, replay checks, and executable gates. This clean repository intentionally excludes paper sources, PDFs, local formatting checks, reviewer notes, host-specific paths, caches, and internal review material.

The package includes the prototype, reproduction scripts, public/synthetic data inputs, result summaries, generated figures/tables, and lightweight checks needed to inspect the reported systems evidence.

## Key results

- Default workload: 15,000 vectors, 64 dimensions, 320 queries, 16 methods, 5,120 method-query rows.
- PACER-A: 0.981 secure recall@10 and 0.881 ordered exactness with zero returned-policy, tenant, or epoch violations.
- PACER-C and PACER-X: 1.000 ordered exactness on the default workload; PACER-X certificates have zero replay failures.
- Scale gates: PACER-A reaches 0.967 recall with 688 raw checks at 60K; on 120K/240K frontier audit and adapter-contract checks it verifies 683/835 raw identifiers, and PACER-X remains ordered exact.
- Memory/build audit: structural metadata is 33.0 bytes/record at 15K, 39.8 at 60K, 46.0 at 120K, and 49.4 at 240K; build cost is reported as an optimizer-facing quantity.
- Blind robustness: 27 frozen configurations; PACER-A beats post-filtering in exactness and raw-work on every configuration; PACER-C and PACER-X are exact.

## Reproduction entry points

Use the quick gate first:

```bash
bash reproduce_checks.sh
```

To regenerate the default synthetic workload from the repository root:

```bash
bash reproduce_default_workload.sh
```

The artifact checker validates generated evidence, safety counters, certificate replay, finite model checking, SQL workloads, public-feature checks, dynamic update checks, scalability, adapter-contract, and memory audits, and table consistency. The access-path adapter contract can also be inspected directly with `python prototype/access_path_adapter.py`.

## Files

- `prototype/`: transparent Python prototype and experiment drivers.
- `results/`: generated CSV/JSON summaries.
- `figures/`: generated PDF figures and LaTeX tables.
- `Dockerfile`, `environment.lock.txt`: optional containerized reviewer environment for the quick gates.
