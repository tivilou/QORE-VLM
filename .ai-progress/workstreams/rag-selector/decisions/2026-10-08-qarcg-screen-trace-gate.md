# Q-ARCG screen: no promotion; repair provenance first

- Date: 2026-10-08
- Scope: rag-selector
- Status: active
- Supersedes: None
- Superseded By: None
- Gate: recover compact run provenance and diagnose loss/rescue before choosing another experiment
- Authorization: result inspection and durable recording only; no new experiment or formal run

## Decision

The uploaded 100-case trace gives Reader `3.55/5`, Q-ARCG `3.49/5`, and matched classical control `3.53/5` Silver overlap. Q-ARCG versus Reader is `2/90/8` improved/equal/worse with mean delta `-0.06` and bootstrap 95% interval `[-0.12,0.00]`. Neither screen gate passes. Do not promote this configuration or increase its budget solely to rescue the exposed Silver result.

This is a bounded, training-dependent screen rather than a full-budget falsification of quantum Reader calibration. Preserve the broader mechanism as inconclusive pending training/provenance evidence; no L1/L2 or quantum advantage claim is supported. This record narrows the older screen kill wording to stopping promotion of the tested configuration, not declaring every related trainable mechanism impossible.

## Evidence and cumulative lesson

See [trace audit](../refs/qarcg_reader_screen_20261007_trace_audit.md). Preserving the relevance anchor avoids the previous large regression, but 13 changed cases produce only two gains and eight losses, with entrants only from score ranks 6-9. Repeated feature-fusion failures plus this result make blindly strengthening span corrections unjustified. Keep the positive anchor carryover and investigate discrimination and weak-target quality instead.

## Next action

Recover `summary.json`, `report.md`, `run_metadata.json`, and `upload_manifest.json` from the collaborator's existing run. They are not currently found on either inspected GitHub branch or the exchange. Verify usable training cases, loss histories, and executed code/config identity before attributing failure to the gate, features, or training budget. Then audit all 13 changed cases, including gains, neutral swaps, and losses, against the fixed detail/Reader artifacts. This analysis requires no new dataset or training run. Future mechanism implementation requires a separate approved design.
