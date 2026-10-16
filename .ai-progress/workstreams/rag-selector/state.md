# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline. Establish independently verified full-data utility before L1/L2; current100-case panel remains L0.

## Current State
- [RESULT AUDITED / CURRENT CONFIGURATION CLOSED] Collaborator rotation run `20261015T022004545902Z` completed59-question/228-pair training and100-case evaluation. Reader/quantum/classical useful totals203/147/170; direct138/90/112; fixedSilver355/175/200. Utility and quantum-attribution gates fail.
- Quantum-minus-Reader useful mean−0.56,95%CI[−0.80,−0.32],14/44/42 wins/ties/losses; quantum-minus-classical−0.23,[−0.40,−0.07]. This is a negative fixed-panel screen, not a full-data result or rejection of the entire quantum Reader family.
- Training actually changes scores: quantum119/228 pair inversions and22 eligible positives rescued, but104/183 protected candidates lost. Classical116/228,28 rescues,78 protected losses. Quantum changes97 evaluation sets, adding57/removing113 useful passages.
- Objective shortcut diagnosed: positive affine compression to10% preserves all Reader rankings yet lowers training objective3.979→1.065. Actual quantum final objective1.177 and training score std4.159→0.456. Pair loss and zero-margin squared protection can reward compression without reliable Top5 retention.
- [ABLATION INVALID / SOURCE EXCEPTION] Old/new named loss arms are exact duplicates within each head family. Quantum old-loss late gradients match new loss, not old. Executed runner SHA differs from reportedGit revision; exact executed file absent. Repository also has target-wrapper and truncated bundle-hash delivery defects. Metrics/checkpoint output replay are independently valid; full source/optimizer replay and loss causal attribution are not.
- Final support audit remains closed:505 dispositions; immutable59/228/75-slot training bundle;135 distinct eligible missed-direct candidates;183 conservative protected candidates, not all direct. No pending annotation queue. Training/evaluation question hashes do not overlap.
- Fixed10-data-qubit/2-layer/200-parameter rotation, frozenReader and matched classical control were used. No Generator/evaluator, retrieval change, local model fitting or Wiki-DPR download occurred during this audit.

## Next Actions
1. Before any rerun, archive exact executed runner bytes/patches and repair explicit head/loss dispatch, target schema and full64-character bundle hash. Keep original result artifacts unchanged.
2. Recommended next research step: train-cache scale-invariant, actualTop5-membership objective audit with compression/tie/protection counterexamples; retain quantum Reader integration and matched classical control. A loss repair alone is not quantum novelty. New implementation/fitting is not started or authorized by this analysis.
3. Only after engineering gates pass, design a paired candidate run with unchanged utility/direct-protection/attribution gates. Do not widen capacity/epochs, tune Silver, or scale this failed configuration.
4. Documentation closeout pending: versioned idea explanation and portal result association; do not claim the portal was refreshed.

## Blockers
- No result-upload blocker: six primary source artifacts and receipts verified. Exact executionrunner source and optimizer-state history were not supplied; full training reproduction and intended cross-loss comparison remain incomplete.
- Fixed100 panel and historical research opinions are exposed. Research labels are not officialgold; prior87/Q-ARCG/source434 provenance limitations persist. Fresh heads do not erase exposure.

## Validation
- Six actual source sizes/SHA agree with uploadmanifest and service sidecars; full3.27GB NPZ SHA checked remotely. Selected4651/9318 arrays are finite and subsetSHA matches receipt; no claim that every raw array received numerical checks.
- All159×6=954 trace rankings,600 posthoc metric rows, and reported paired bootstrap intervals recomputed.795 trained-head output replays pass with maxabs9.06e-6,0Top5 order differences; initialnull maxabs1.91e-6.30 gradient witnesses examined; no full AdamW-state replay.
- Compact summary/runmetadata/uploadmanifest archived locally under `exchange/five_ideas/quantum_reader_rotation_screen_100/20261015T022004545902Z/`; private targets, fulltrace and NPZ remain exchange/cache only.
- Audit environment: existinglambeq,NumPy1.23.5,Torch2.3.1CPU; no packages installed. Collaborator environment and pinnedReader identity are in original runmetadata.
- Current session: [20261016T052658Z-da80a5](sessions/2026-10/20261016T052658Z-da80a5.md).

## Relevant Decisions And Evidence
- Active result disposition: [2026-10-16 rotation result](decisions/2026-10-16-quantum-reader-rotation-screen-result.md).
- Owner report: [rotation result and compression diagnosis](../../../../docs/rag-research-log/20261015T-022004Z-quantum-reader-rotation-screen-result.md).
- Receipt: [rotation screen audit](refs/quantum_reader_rotation_screen_20261015_receipt.json); private working audit `.cache/quantum_reader_rotation_result_audit/20261015T022004545902Z/`.
- Historical contract: [crossed screen](decisions/2026-10-13-quantum-reader-rotation-screen.md); [engineering](decisions/2026-10-13-reader-rotation-engineering.md); [previous closed runner](decisions/2026-10-12-protected-boundary-reader-runner.md).
- Support-source finalization: [505 dispositions](decisions/2026-10-11-final-support-boundary-resolution.md); feasibility audit `docs/rag-research-log/20261013T-protected-boundary-feasibility-audit.md`. Durable rules and complete decision navigation: `critical-invariants.md`.
- Previous implementation plan `plans/protected-boundary-runner/v001.json` stays completed; it is not an authorization to launch a new experiment.
