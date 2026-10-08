# Semantic Reader: qualify optimization before a controlled rerun

- Date: 2026-10-08
- Status: active
- Supersedes: None (bounded follow-up repair; historical result stays frozen)
- Superseded By: None
- Gate: train-only learning qualification passed; real-task utility and matched-classical gates remain pending
- Scope: rag-selector
- Authorization: user's “开始” resumes the recommended training-path repair. Prepare a separately registered collaborator handoff; do not run a real task experiment on development servers.
- Predecessor: [training-collapse analysis](2026-10-08-semantic-reader-training-collapse.md).
- Session: [implementation outcome](../sessions/2026-10/20261008T130613Z-9d0ac2.md).

## Why this repair, rather than another architecture

Cumulative case studies show that replacing the relevance anchor with span confidence, broad diversity or score mixing has repeatedly harmed selection. The previous semantic run does not test a healthy semantic intervention: both semantic heads degenerated to the unchanged Reader on all 100 questions. Repair that implementation failure before treating representation capacity or the quantum interaction as falsified. Retain the anchor, quantum line and matched semantic classical control; do not increase data, epochs or circuit depth.

## Bounded evidence and the second failure

Use only `training_001/002` base/pooled/scalar/weak-mask arrays from the already verified archive SHA256 `4d18cca65cfc8db290020ccc1fea1c5d64773b95be488c9a68d78b9e807c581a`. Alternate these two cases for a fixed 64 updates per head/policy. No evaluation arrays, Silver labels or Generator results enter this qualification. This is engineering fitting on two training cases, not an independent task experiment or a complete replay of the old 434-case optimizer.

First, decoupled projection-only AdamW prevented parameter shrinkage. However, at the old uniform learning rate its wide projection saturated: quantum/classical saturation fractions were about `100%/99%`; normalized within-case score-correction spread was only `1.31e-8/5.97e-8`. A weak “any nonzero correction” check initially passed even the old optimizer. That check was tightened explicitly before the collaborator rerun: a common offset or one-ULP variation is not evidence of useful candidate-dependent learning. This diagnostic refinement does not rewrite any historical screen utility gate.

One bounded adjustment followed: projection-matrix learning rate is the original base LR divided by `sqrt(fan_in)`, derived from input dimension rather than a Silver-driven search. At H=768 this is `0.003/sqrt(3840)`; other parameters retain LR=0.003. Compare historical coupled Adam, unscaled projection-only AdamW, and fan-in-scaled AdamW separately on both semantic heads. The first two fail the strengthened engineering guard; both fan-in-scaled arms pass. These diagnostic policy controls are not added as extra real-data search arms.

Local quantum/classical projection norm ratios are `1.0189/1.0201`, mean candidate encoding spreads `0.30276/0.32740`, normalized correction spreads `0.00069046/0.00049987`, and maximum absolute corrections `0.009789/0.007433`. Both retain exact initial null and nonzero finite projection/readout gradients after leaving null. Reopened parameters, Adam moments and the last update/signals replay exactly on the qualification's CPU environment. Preserved nonconstant scores do not establish improved Top-5 selection or quantum attribution.

## Frozen repair boundary

- Policy: `adamw_projection_fanin_v1`. AdamW decays only `project.weight`; circuit, gate/readout, null scale and all biases receive zero decay. Both semantic controls have identical groups, projection initialization, budget, data, order, loss and seed. Scalar control has no projection, so its group has zero decay and unchanged base LR; its optimizer behavior is explicitly changed too.
- Head, Reader revision, masks/pooling, loss/anchor penalty, requested 512 training questions, three epochs, seed, 100 fixed evaluation candidate sets, posthoc metrics/bootstrap/utility gates and circuit depth remain unchanged.
- Original config/plan/output namespace remain frozen. New config/plan and output namespace are `semantic_reader_training_repair_100`.
- Training health is checked at update 64 and each epoch end on the first two usable training cases. Require finite data gradients with live projection/readout/scale in the current check window, projection norm ratio >1e-6, encoding spread >1e-6, and candidate-dependent actual float32 correction spread / base spread >1e-6. These are collapse guards, not task success thresholds.
- Save per-step data-gradient norms, witness pre/post parameters, optimizer moments/steps and epoch moments. Reopen/replay the selected updates, as well as the pre-existing tensor/selection trace.
- A failed learning gate stops before evaluation inference and preserves/uploads training failure summary, identities and numeric checkpoint. A short synthetic fixture reports `deferred_below_64_updates`, not a false learning pass.
- Raw execution source hashes stay recorded. Deployment qualification additionally compares CRLF-normalized source hashes, because Windows/Linux line endings alone differ; no semantic source mismatch is ignored.

## Next task and claim boundary

The next useful measurement is a collaborator-controlled rerun of the same four-arm screen with this new config, not a larger dataset or deeper circuit. Its existing positive-evidence/protected-direct and matched-classical gates are unchanged. If it fails the learning guard, inspect the preserved checkpoint before another task run. If it learns but fails utility, analyze rescued/harmed cases against cumulative historical evidence and question the weak supervision before proposing another mechanism. L0 only; no task improvement, L1/L2 or quantum advantage follows from qualification.

## References

- Qualification: [compact train-only report](../refs/semantic_reader_training_qualification_20261008.json).
- Checks and rollback: [verification](../refs/semantic_reader_training_repair_verification_20261008.md).
- Owner log: [Chinese explanation](../../../../docs/rag-research-log/20261008T-date-only-semantic-reader-training-repair.md).
- Config/plan: `configs/experiments/semantic_reader_training_repair_100.json` and `_plan.json`.
- Collaborator wrapper: `scripts/collab/five_ideas/run_semantic_reader_training_repair_100.sh`.
