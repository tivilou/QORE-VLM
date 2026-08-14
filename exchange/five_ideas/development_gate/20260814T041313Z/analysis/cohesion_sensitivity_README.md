# Cohesion Sensitivity Replay

This offline replay uses the passage-free `qubo_payload.jsonl.gz` from the
Phase 2 development gate. It reconstructs the exact 15-candidate QUBO pool
with the project's `build_qubo_matrix_from_w` and brute solver, then applies
the calibrated cohesion correction for an eta grid.

The eta=0 row reproduces the recorded Answer Scorer control Recall and
redundancy exactly. The replay reports selection-level Recall, redundancy,
energy, delta statistics, and selection changes. It does not estimate new
generation F1 because the compact artifact contains no generated answers.

Input run: `20260814T041313Z`

Outputs:

- `cohesion_sensitivity_aggregate.csv`: eta-level summary.
- `cohesion_sensitivity.csv`: per-question replay rows.

The replay is not a Wiki-DPR run and does not access raw passages.
