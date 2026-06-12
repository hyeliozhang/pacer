#!/usr/bin/env bash
set -euo pipefail
# Full default-workload regeneration. This can take substantially longer than
# reproduce_checks.sh on shared CPU-only machines because it evaluates 320
# queries across 16 methods in fresh worker processes.
mkdir -p checks results
PACER_PROGRESS=${PACER_PROGRESS:-1} python prototype/run_experiments_chunked.py --out results --progress > results/run_experiments_chunked_final.log
python prototype/make_v16_tables.py > results/make_v16_tables.log
python prototype/plot_results.py > results/plot_results.log
python checks/check_artifact.py | tee checks/check_artifact.log
