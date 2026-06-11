#!/usr/bin/env bash
set -euo pipefail
test -s prototype/run_experiments_chunked.py
python -m py_compile prototype/*.py checks/check_artifact.py checks/final_submission_gate.py checks/strict_reviewer_gate.py checks/reference_audit.py checks/official_supplement_gate.py tests/test_invariants.py
python tests/test_invariants.py > checks/test_invariants.log
pytest -q tests > checks/pytest.log
python prototype/access_path_adapter.py > results/run_adapter_contract.log
python - <<'PY' > results/run_blind_robustness.log
import json
from pathlib import Path
summary = json.loads(Path('results/blind_robustness/blind_robustness_summary.json').read_text())
print(json.dumps({
    'status': 'frozen_blind_robustness_outputs_present',
    'note': 'Quick checks validate the frozen blind-robustness outputs. Run prototype/run_blind_robustness.py for full regeneration.',
    'configs': summary.get('configs'),
    'query_method_rows': summary.get('query_method_rows'),
    'runtime_seconds_original': summary.get('runtime_seconds'),
    'manifest_sha256': summary.get('manifest_sha256'),
    'claim_checks': summary.get('claim_checks', {}),
}, indent=2))
PY
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('results/blind_robustness/blind_robustness_summary.json').read_text())
Path('results/run_blind_robustness.time').write_text(f"recorded_from_frozen_summary_runtime_seconds={summary.get('runtime_seconds')}\n")
PY
python - <<'PY2' > results/run_scale_frontier.log
import json
from pathlib import Path
summary = json.loads(Path('results/scale_frontier/scale_frontier_summary.json').read_text())
print(json.dumps({
    'status': 'frozen_scale_frontier_outputs_present',
    'note': 'Quick checks validate the frozen 120K/240K frontier-scale outputs. Run reproduce_scale_frontier.sh for full regeneration.',
    'schema': summary.get('schema'),
    'rows': summary.get('rows'),
    'configs': [s.get('n') for s in summary.get('summaries', [])],
    'claim_checks': [s.get('claim_checks', {}) for s in summary.get('summaries', [])],
}, indent=2))
PY2
python prototype/make_v16_tables.py > results/make_v16_tables.log
python checks/final_submission_gate.py > checks/final_submission_gate.log
python checks/strict_reviewer_gate.py > checks/strict_reviewer_gate.log
python checks/reference_audit.py > checks/reference_audit.log
python checks/check_artifact.py > checks/check_artifact.log
cat checks/check_artifact.log
