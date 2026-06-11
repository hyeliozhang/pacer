# PACER status

The package is in final reviewer-facing state. The paper body fits within the ICDE 12-page body limit, the AI-generated content acknowledgement and references begin after the body, and no appendix is included in the main PDF.

## Strong-review closure

- Novelty is framed as a proof-carrying DBMS operator contract for policy-constrained vector top-k, not as another filtered ANN candidate source.
- Theory covers selected top-k semantics, fixed-budget post-filter rank-depth limits, conservative SQL summary pruning, strict-score certificate soundness, certificate tightness, snapshot safety, and witness sufficiency.
- Efficiency/scalability is now explicit in the main text: a cost model, optimizer-facing counters, an efficiency/scalability table, 60K full gate, 120K/240K frontier-scale audits and adapter-contract conformance, structural metadata/build-cost audit, and optional Docker environment are all paper-facing.
- Artifact gates validate paper/result alignment, reference coverage, source defaults, package hygiene, PDF rendering, replay checks, SQL evidence, scalability, and memory accounting.

## Intended reviewer workflow

Run `bash reproduce_checks.sh` for a fast consistency pass. Run `bash reproduce_default_workload.sh` to regenerate the default workload. Run `bash reproduce_scale_frontier.sh` to regenerate the 120K/240K frontier audit and adapter-contract check. The generated outputs are checked by `checks/check_artifact.py` and `checks/final_submission_gate.py`.
