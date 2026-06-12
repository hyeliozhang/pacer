#!/usr/bin/env bash
set -euo pipefail
# Regenerate the compact 120K/240K frontier-scale audit used by the paper's
# efficiency/scalability table.  This is intentionally smaller than the default
# 320-query workload so it can be rerun on CPU-only machines.
mkdir -p checks results
python prototype/run_scale_frontier.py > results/run_scale_frontier.log
python prototype/make_v16_tables.py > results/make_v16_tables.log
python checks/check_artifact.py | tee checks/check_artifact.log
