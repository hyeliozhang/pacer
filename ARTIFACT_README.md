# PACER Artifact Guide

This guide describes how to inspect and rerun the PACER artifact. The code is
designed for CPU-only execution and avoids external services.

## Environment

The recommended environment is Docker:

```bash
docker build -t pacer-artifact .
docker run --rm pacer-artifact
```

For a local run, use Python 3.11 and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The prototype uses NumPy for vector operations, scikit-learn only for public
feature-matrix datasets, matplotlib for figure regeneration, and pytest for the
test suite.

## Quick Check

Run:

```bash
bash reproduce_checks.sh
```

This command performs four checks:

1. Python compilation for the prototype, tests, and artifact checker.
2. Deterministic invariant tests in `tests/test_invariants.py`.
3. Access-path adapter contract execution in `prototype/access_path_adapter.py`.
4. Evidence validation through `checks/check_artifact.py`.

The generated logs are written to `checks/*.log` and `results/*.log`.

## Regenerating Paper Tables and Figures

The shipped results can be turned into LaTeX tables with:

```bash
python prototype/make_v16_tables.py
```

Experiment figures can be regenerated with:

```bash
python prototype/plot_results.py
```

If a TeX engine is unavailable, the conceptual diagrams use a deterministic
matplotlib fallback. The repository already includes the generated PDFs and
LaTeX table fragments under `figures/`.

## Regenerating Experiments

Default synthetic workload:

```bash
bash reproduce_default_workload.sh
```

Frontier-scale workload:

```bash
bash reproduce_scale_frontier.sh
```

The default workload is the heaviest script because it evaluates 320 queries
across 16 methods. Runtime varies by CPU and available memory. The shipped
summaries under `results/` are the reference outputs used by the evidence
checker.

## Data

Most workloads are synthetic and generated deterministically by the prototype.
The public Pendigits files are included in `data/external/pendigits/` with
`SHA256SUMS`. The scikit-learn feature-matrix runs use datasets available from
the local scikit-learn package.
