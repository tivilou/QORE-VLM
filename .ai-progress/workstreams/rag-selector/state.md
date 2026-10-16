# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline. Establish independently verified full-data utility before L1/L2; current100-case panel remains L0.

## Current State
- [IMPLEMENTED / CPU AUDIT PASSED; NO REAL FIT] New train-only `robust_standardized_top5_membership.v1`: differentiable median/IQR scaling, fifth-of-other49 positive-margin membership, feasible rescue capacity and all-immutable protection. New loss is separate from the closed100 runner; selection utility is untested.
- Both local and verified primary CPU audits pass on59 frozen training cases:708 affine checks,177 collapse rejections,59 outlier checks,24 saved arrays and6 reopened score-gradient replays.21 tests pass in each environment. No real head training, Reader loading, dataset download, Silver/evaluation-metric access or GPU use.
- [DELIVERY REPAIRED] Independent loss-suffix dispatch, wrapped `row["target"]`, full64-character bundle hash, prospective exact-source archive/hash check and arm-objective provenance implemented. Historical runner retains historical loss definitions; do not run it expecting the new objective.
- Server isolated implementation: `/home/Q-DUET-VLM/reader-top5-objective-audit-20261016`; output `.cache/audit/20261016T082513220974Z/` under that path. Canonical checkout and old artifacts untouched. Local audit `.cache/reader_top5_objective_audit/20261016T082015507042Z/`.
- [RESULT AUDITED / CURRENT CONFIGURATION CLOSED] Collaborator rotation run `20261015T022004545902Z` completed59-question/228-pair training and100-case evaluation. Reader/quantum/classical useful totals203/147/170; direct138/90/112; fixedSilver355/175/200. Utility and quantum-attribution gates fail.
- Quantum-minus-Reader useful mean−0.56,95%CI[−0.80,−0.32],14/44/42 wins/ties/losses; quantum-minus-classical−0.23,[−0.40,−0.07]. This is a negative fixed-panel screen, not a full-data result or rejection of the entire quantum Reader family.
- Training actually changes scores: quantum119/228 pair inversions and22 eligible positives rescued, but104/183 protected candidates lost. Classical116/228,28 rescues,78 protected losses. Quantum changes97 evaluation sets, adding57/removing113 useful passages.
- Objective shortcut diagnosed: positive affine compression to10% preserves all Reader rankings yet lowers training objective3.979→1.065. Actual quantum final objective1.177 and training score std4.159→0.456. Pair loss and zero-margin squared protection can reward compression without reliable Top5 retention.
- [HISTORICAL ABLATION INVALID / SOURCE EXCEPTION] Previous old/new named loss arms are exact duplicates within each head family. Quantum old-loss late gradients match new loss, not old. Executed runner SHA differs from reportedGit revision; exact executed file absent. Repository delivery defects are now repaired prospectively. Metrics/checkpoint replay remain valid; historical full source/optimizer replay and loss causal attribution do not.
- Final support audit remains closed:505 dispositions; immutable59/228/75-slot training bundle;135 distinct eligible missed-direct candidates;183 conservative protected candidates, not all direct. No pending annotation queue. Training/evaluation question hashes do not overlap.
- Fixed10-data-qubit/2-layer/200-parameter rotation, frozenReader and matched classical control were used. No Generator/evaluator, retrieval change, local model fitting or Wiki-DPR download occurred during this audit.

## Next Actions
1. Create an independently named, preregistered paired quantum/classical training contract and runner for the candidate objective. Preserve qualified targets, Reader/rotation capacity/budget and task gates. Do not use the closed historical wrapper as a new-objective experiment.
2. Add optimizer-state/update replay and discrete Top5 health checks before collaborator delivery. `formal_training_ready` remains false; this checkpoint has train-cache objective/score-gradient evidence only. Loss repair alone is not quantum novelty.
3. Then hand off one-command real training to the collaborator, with unchanged utility/direct-protection/attribution gates. No blind capacity/epoch increase, Silver tuning or scale-up of the failed configuration. Cached-feature checks can use CPU; fresh Reader extraction benefits from GPU.
4. Documentation closeout pending: versioned idea explanation and portal result association; do not claim the portal was refreshed.

## Blockers
- No result-upload blocker: six primary source artifacts and receipts verified. Exact executionrunner source and optimizer-state history were not supplied; full training reproduction and intended cross-loss comparison remain incomplete.
- Fixed100 panel and historical research opinions are exposed. Research labels are not officialgold; prior87/Q-ARCG/source434 provenance limitations persist. Fresh heads do not erase exposure.

## Validation
- Current objective receipt: [CPU audit](refs/reader_top5_objective_audit_20261016_receipt.json). Local Torch2.3.1CPU/NumPy1.23.5; server Torch2.7.1/NumPy1.26.4 with CPU tensors. Python compile, Bash syntax and unchanged6-plugin historical plan pass.
- Synthetic tests distinguish head/loss dispatch, exercise the actual wrapped-row interface, and check quantum/classical gradient flow, membership counterexamples and exact source readback. No new model fitting or selection output is implied.
- Six actual source sizes/SHA agree with uploadmanifest and service sidecars; full3.27GB NPZ SHA checked remotely. Selected4651/9318 arrays are finite and subsetSHA matches receipt; no claim that every raw array received numerical checks.
- All159×6=954 trace rankings,600 posthoc metric rows, and reported paired bootstrap intervals recomputed.795 trained-head output replays pass with maxabs9.06e-6,0Top5 order differences; initialnull maxabs1.91e-6.30 gradient witnesses examined; no full AdamW-state replay.
- Compact summary/runmetadata/uploadmanifest archived locally under `exchange/five_ideas/quantum_reader_rotation_screen_100/20261015T022004545902Z/`; private targets, fulltrace and NPZ remain exchange/cache only.
- Audit environment: existinglambeq,NumPy1.23.5,Torch2.3.1CPU; no packages installed. Collaborator environment and pinnedReader identity are in original runmetadata.
- Current session: [20261016T083400Z-e91426](sessions/2026-10/20261016T083400Z-e91426.md).

## Relevant Decisions And Evidence
- Active implementation: [objective engineering](decisions/2026-10-16-reader-top5-objective-engineering.md); [owner report](../../../../docs/rag-research-log/20261016T-082015Z-reader-top5-objective-cpu-audit.md).
- Active result disposition: [2026-10-16 rotation result](decisions/2026-10-16-quantum-reader-rotation-screen-result.md).
- Owner report: [rotation result and compression diagnosis](../../../../docs/rag-research-log/20261015T-022004Z-quantum-reader-rotation-screen-result.md).
- Receipt: [rotation screen audit](refs/quantum_reader_rotation_screen_20261015_receipt.json); private working audit `.cache/quantum_reader_rotation_result_audit/20261015T022004545902Z/`.
- Historical contract: [crossed screen](decisions/2026-10-13-quantum-reader-rotation-screen.md); [engineering](decisions/2026-10-13-reader-rotation-engineering.md); [previous closed runner](decisions/2026-10-12-protected-boundary-reader-runner.md).
- Support-source finalization: [505 dispositions](decisions/2026-10-11-final-support-boundary-resolution.md); feasibility audit `docs/rag-research-log/20261013T-protected-boundary-feasibility-audit.md`. Durable rules and complete decision navigation: `critical-invariants.md`.
- Previous implementation plan `plans/protected-boundary-runner/v001.json` stays completed; it is not an authorization to launch a new experiment.
