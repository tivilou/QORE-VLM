# Support-qualified boundary targets: limited offline implementation

- Stage: target compilation and synthetic validation only; no fitting authorization.
- Candidate: support-qualified quantum semantic boundary learning, retaining frozen
  DPR relevance and the existing four-qubit semantic head. This compiler itself is
  deterministic classical bookkeeping, not a new quantum circuit.
- Hypothesis: answer-containment mistakes and ambiguous targets reward score
  compression rather than support discrimination. Qualified direct-vs-selected-
  irrelevant pairs may improve the existing quantum semantic head's supervision.
- Recovery evidence: complete1600 support audit; optimizer repair established an
  active but nonselective correction. Fixed fusion, broad set penalties and weak
  labels have not beaten the Reader. Do not escalate depth, amplitude or epochs
  without the missing supervision/learning-path evidence.

## Insertion and interface

Standalone `applications/rag/support_qualified_boundary_targets.py`, version1.
Input: immutable old-training diagnostic Top50 trace + complete review metadata +
separate qualification overlay. Output: `rag.support_boundary_target_draft.v1`.
No import or registration in the online selector, Reader, trainer or evaluator.
No network, torch/model loading, passage reclassification or reference editing.
The explicitly selected CLI is the only composition; there is no auto-discovery.
Unqualified/default mode emits proposals but zero qualification-checked pairs.

Every source file is hash checked by the CLI. The overlay binds the entire input,
including label metadata, order and scores. Source question/text hashes and frozen
Top5 selection are checked again. Existing question flags remain excluded even
if an overlay says qualified; resolution requires a later versioned policy.

## Qualification contract, not automatic certification

Question scope must explicitly resolve time, location, version, entity, metric
and relation to the unchanged reference. Each qualified question/item declares
independently-adjudicated confidence with distinct primary/independent reviewer
identities and artifacts. CLI reopens local packet witnesses and verifies bytes.
This verifies declared provenance, NOT semantic agreement or real independence.
Human/independent review remains a separate responsibility. Current real overlays
are entirely pending. Neither absence of a flag nor weak containment qualifies.

## Target and protection policy

- Only missed direct vs Reader-selected irrelevant pairs; no direct-vs-direct.
- Partial, uncertain, contradictory and missing annotations abstain, never0.
- All existing selected direct witnesses must be qualified before any pairs for
  that question are emitted; set-level validator forbids dropping those witnesses.
- Normalize pair weights to sum1 per question, not proportional to pair count.
- Replacement budget is bounded by distinct missed-direct and selected-irrelevant
  counts, not pair count. More than5 direct passages does not force all into5.
- No pair means no forced binary training example. Zero-direct cases are kept in
  diagnostics, not silently dropped or relabeled as negative questions.
- Every output stays `training_consumable=false`, `full434_qualified=false`,
  `training_authorized=false`, including synthetically qualified fixtures.

## Evidence and future gate

Real32-case replay: 9 unflagged provisional cases yield28 proposal pairs and at
most10 irrelevant replacement slots. No real item independently qualified:0
draft pairs. This is not accuracy, benefit, a formal training cohort or L1/L2.
Synthetic tests exercise exact values, masks, weight normalization, null behavior,
protected selections, immutable inputs, stale/duplicate identities, witness hashes,
exclusive outputs and excluded Silver lane. No model trace is claimed here.
Architecture evidence is limited offline bookkeeping, not a completed formal
`sample-trace.v2` training experiment or quantum attribution test.

Next: independently qualify priority cases5/6/8/13/16/17/21/26/27 and their
retention/boundary witnesses, then establish whole434 supervision and a fresh
347/87 protocol. Specify prospective pair loss, retention constraint and matched
nonpositive controls plus bounded-score reachability before wiring any loss.
Old heads have seen old partitions; exposed100 Silver stays evaluation-only.
Future collaborator-local quantum/classical/scalar/compression arms share targets,
splits, budgets and captured representations. A new head run requires fresh
validation improvement without direct-retention harm, active gradient/correction
variation and support-specific effects against controls. Stop if qualifications
are inadequate or effects remain compression-only; a repaired optimizer alone
does not pass utility. Production loss and residual limits remain unchanged now.

## Rollback

Remove only newly introduced offline compiler, CLI, test and contract; retain all
old review/source artifacts. Publication supplies hash-checked backups, diff,
literal-command verification and an executable scratch-tested rollback.
