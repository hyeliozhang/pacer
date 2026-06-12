# Reproducibility

PACER is implemented as a deterministic CPU-only Python artifact. The shipped
summaries under `results/` are the reference outputs used by the paper tables;
the scripts below rerun the checks or regenerate the corresponding outputs.

## Fast Check

```bash
bash reproduce_checks.sh
```

Expected output: `PACER artifact check: pass`.

This check compiles the code, runs deterministic invariant tests, executes the
access-path adapter contract, rebuilds table fragments from shipped summaries,
and validates the main claim counters.

## Default Workload

```bash
bash reproduce_default_workload.sh
```

This regenerates the 15K-vector, 320-query workload and updates the table and
figure files. The workload is larger than the fast check and can take
substantially longer on shared CPU-only machines.

## Frontier-Scale Workload

```bash
bash reproduce_scale_frontier.sh
```

This regenerates the compact 120K/240K frontier-scale audit used to check
certificate behavior and raw-identifier accounting beyond the default workload.

## Inspecting Evidence

The most useful files are:

- `results/summary_n15000.json`
- `results/large_scale/large_scale_60k_summary.json`
- `results/scale_frontier/scale_frontier_summary.json`
- `results/proof_replay/proof_replay_summary.json`
- `results/model_check/model_check_summary.json`
- `results/sql_policy/sql_policy_summary.json`
- `results/sql_ast/sql_ast_summary.json`
- `results/dynamic_overlay/dynamic_overlay_summary.json`
- `results/snapshot_stream/snapshot_stream_summary.json`
- `results/blind_robustness/blind_robustness_summary.json`
- `results/external_summary.json`
- `results/pendigits/summary_pendigits.json`

`checks/check_artifact.py` validates these files and reports the key values in
a compact JSON summary.
