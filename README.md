# PACER

PACER is a CPU-only research prototype for proof-carrying,
policy-constrained vector top-k query processing over attributed vector
relations. The code implements an IVF-style access path, conservative policy
summaries, row-slice candidate generation, row-level visibility checks, and
certificate replay for ordered secure top-k answers.

The repository contains the executable artifact used for the paper
`PACER: Proof-Carrying Policy-Constrained Vector Top-k Query Processing`.
It includes the prototype, shipped result summaries, generated paper figures
and tables, public feature-matrix inputs, and consistency checks that connect
the code to the reported evidence.

## Quick Start

The fastest path is the containerized check:

```bash
docker build -t pacer-artifact .
docker run --rm pacer-artifact
```

On a local Python 3.11 installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash reproduce_checks.sh
```

`reproduce_checks.sh` compiles the prototype, runs deterministic invariant
tests, exercises the access-path adapter contract, regenerates the LaTeX
tables from the shipped summaries, and validates the main JSON/CSV evidence.

## Main Evidence

- Default workload: 15,000 vectors, 64 dimensions, 320 queries, 16 methods,
  and 5,120 method-query rows.
- PACER-A obtains 0.981 secure recall@10 and 0.881 ordered exactness while
  checking 707 raw identifiers per query; post-filtering obtains 0.610 and
  0.169 while checking 1,893 identifiers.
- PACER-C and PACER-X are ordered-exact on the default workload, with zero
  replayed certificate failures for PACER-X.
- Scale checks include a 60K full gate and 120K/240K frontier probes; PACER-X
  remains ordered-exact and PACER-A uses fewer raw identifiers than
  post-filtering.
- The artifact also includes proof replay, finite model checking, SQL-policy
  workloads, SQL-AST workloads, update/deletion checks, public-feature matrix
  runs, memory accounting, and adapter-contract checks.

## Reproduction Levels

- Fast consistency check: `bash reproduce_checks.sh`.
- Default workload regeneration: `bash reproduce_default_workload.sh`.
- 120K/240K frontier-scale regeneration: `bash reproduce_scale_frontier.sh`.

The full default workload is intentionally larger than the quick check because
it evaluates 320 queries across 16 methods. The repository ships the generated
summaries so the evidence can be inspected without rerunning the long jobs.

## Repository Layout

- `prototype/`: PACER implementation and experiment drivers.
- `tests/`: deterministic invariant tests for pruning, SQL-AST semantics,
  update visibility, and certificate conditions.
- `checks/`: artifact consistency checker.
- `results/`: shipped CSV/JSON summaries used by the paper tables.
- `figures/`: generated PDF figures and LaTeX table fragments.
- `data/external/pendigits/`: public Pendigits input files and checksums.
- `ARTIFACT_README.md`: detailed reproduction notes.
- `EVIDENCE.md`: claim-to-file evidence map.
- `REPRODUCIBILITY.md`: expected commands, outputs, and runtime notes.
