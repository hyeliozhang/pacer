#!/usr/bin/env bash
set -euo pipefail

mkdir -p checks results

python -m py_compile prototype/*.py checks/check_artifact.py tests/test_invariants.py
python tests/test_invariants.py > checks/test_invariants.log
pytest -q tests > checks/pytest.log
python prototype/access_path_adapter.py > results/run_adapter_contract.log
python prototype/make_v16_tables.py > results/make_v16_tables.log
python checks/check_artifact.py | tee checks/check_artifact.log
