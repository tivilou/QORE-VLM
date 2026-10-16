# Robust scale-invariant Top5 membership objective: CPU audit only

- Date: 2026-10-16
- Scope: rag-selector
- Status: candidate
- Supersedes: None
- Superseded By: None
- Gate: Input identity, affine invariance, collapse rejection, outlier and discrete membership counterexamples, finite gradient and exact source checks; separately preregister paired training and task utility gates.
- Authorization: Implementation and frozen train-cache CPU audit for this checkpoint; no new real-data fitting, model loading or dataset download launched.

## Decision

Implement a train-only median/IQR-normalized membership surrogate; repair independent arm loss dispatch, wrapped targets, complete bundle hash and exact source archival. Keep the new objective separate from the closed100-case runner.

## Reason

Historical loss rewarded unchanged-order compression and lost104 protected candidates; simple standard-deviation normalization also admitted an extreme-low-score shortcut.

## Consequences

Twenty-one tests and frozen59 CPU audits pass locally and on the verified primary. No real optimizer/head fitting or new evaluation was performed; no promotion or attribution claim follows.

## Implementation and evidence

- New train-only module: `applications/rag/scale_invariant_top5_objective.py`; contract: `configs/experiments/reader_top5_objective_audit.json`; audit: `scripts/collab/five_ideas/audit_reader_top5_objective.py`.
- Normalize with differentiable median/IQR scale, reject collapsed scales, retain positive dimensionless membership margin against fifth-of-other49, supervise no more than the frozen replacement capacity, and protect all immutable selected candidates. Use actual membership counts as a separate gate; the smooth objective is not a proof of retention.
- Local and verified primary CPU results agree:708 affine checks,177 collapse rejections,59 outlier checks,24 saved arrays and6 independently reopened score-gradient replays.21 unit tests pass in each environment; unchanged historical6-plugin plan validates.
- New objective is not wired into the closed historical runner. Historical `_new_loss` names continue to mean `top5_retention_objective`, not this candidate. Do not use that wrapper as a new-objective experiment command.
- Exact source archival is prospective. It cannot repair missing execution-source/optimizer-state provenance from the prior run. No new real optimizer states were generated in this objective-only audit.
- Next: independently named paired training contract/runner with optimizer/update replay, unchanged semantic qualification and task gates. Keep the current100 evaluation panel L0 and retain matched classical control; do not tune it using exposed Silver outcomes.
- Main server isolated source directory: `/home/Q-DUET-VLM/reader-top5-objective-audit-20261016`; canonical checkout and source artifacts were not overwritten.
- [Owner log](../../../../docs/rag-research-log/20261016T-082015Z-reader-top5-objective-cpu-audit.md); [audit receipt](../refs/reader_top5_objective_audit_20261016_receipt.json).
