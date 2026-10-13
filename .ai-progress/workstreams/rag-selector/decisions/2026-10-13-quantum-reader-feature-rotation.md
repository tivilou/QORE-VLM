# Exploratory question-conditioned quantum Reader feature rotation

- Date: 2026-10-13
- Scope: rag-selector
- Status: candidate
- Supersedes: None
- Superseded By: None
- Gate: identity/norm/signed-readout replay, task-aligned protection and loss-only controls, then paired utility versus frozen Reader; attribution separately versus matched classical.
- Authorization: User said 开始 for next mechanism design. This turn performs design and deterministic saved-tensor diagnostics only; no real fitting or dataset download. Experimental execution stays collaborator-only.

## Decision

Retain original DPR encoder and classifier as frozen anchors. Investigate a
question-conditioned real-unitary transform of the final CLS representation
before the original relevance classifier. Zero parameters must yield identity
and reproduce Reader Top5. Tentative encoding pads768 to1024 coordinates;
10qubits,2 RY/CRY ring layers,40 conditional angles/200 trainable parameters.
Signed overlap readout and state-preparation costs are part of the design.

Use a same200-parameter classical Givens control. Isolate old/new loss and
old/new head using matched arms: the new protection criterion retains immutable
items in Top5 without requiring rescued positives to score below them. Freeze
labels and never pass online target masks. No new module implemented this turn.

## Reason

The old screen is a valid null: bounded correction cannot overtake168/228
matched negatives;60 remaining pairs were not inverted. Pair failure is not
equivalent to impossible slot rescue:54/135 unique missed-direct candidates
are within the optimistic baseline Top5 score-box envelope, across35/59
questions.99 pairs show initial protection-order tension, but joint movement
does not remove extra pair-box feasibility. These arithmetic bounds do not
certify shared-head learnability.

Original classifier replay for all159 cached cases passed, max error1.87e-6.
Feature-norm score radius median13.007; this admits a different score formation
envelope but does not prove reachability of the shallow candidate or utility.

## Consequences

Candidate staysexploratory_only: same exposed100 L0 panel, limited train59,
provisional support supervision, no formal full-data qualification. Current
design has high novelty collision with OFT/BOFT and Q-LoRA/H-LoRA; it does not
claim parameter advantage over every classical implementation. Classical
simulation is permitted. No quantum advantage, L1/L2 or improvement is claimed.

Full design and prospective screen/kill/transfer/resource requirements:
`docs/rag-research-log/20261013T-quantum-reader-feature-rotation-design.md`.
Deterministic scripts: `audit_protected_objective_geometry.py` and
`audit_reader_rotation_envelope.py`. Envelope uses saved tensors on the verified
3080Ti server; raw tensors stay in18083 storage. No model/data download.
