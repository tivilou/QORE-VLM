# RAG Passage Selection with QORE

## Goal

- Improve fixed Top-50 -> Top-5 with defensible utility evidence while retaining the quantum research line; keep retrieval, production selector, Generator, evaluator, labels, and answer path fixed.

## Current State

- No real-data L1/L2 selector result exists. Prior selector/scorer screens and cumulative lessons are indexed in `critical-invariants.md`.
- Repaired screen `five_ideas/semantic_reader_training_repair_100/20261008T145652Z/` is audited. Reader/quantum semantic/classical/scalar positive/direct totals are `203/138`, `204/139`, `203/138`, `203/139`; fixed Silver totals `355/356/354/353`. Quantum improves 1 question, ties 99, harms 0 on positive/direct counts; delta `+0.01`, CI `[0,0.03]`; both gates fail. L0/inconclusive, not a broad-family kill.
- Training repair works on the 434-case cohort: 1302 updates per arm, finite live gradients, preserved projection norms and 5000 nonzero score corrections. Quantum changes two Top-5 sets: case 82 is a direct rescue, 79 is an irrelevant swap. Classical changes two neutral sets; scalar changes ten with cancelling positive gains/losses.
- Corrections mostly flatten Reader scores (quantum median correlation `-0.870`); low-ranked nonpositive candidates get favorable relative corrections at least as often as positives in coarse rank bands. Deep-rank recovery is structurally restricted by the bounded residual; quantum encoding coordinates are 76.3% near tanh limits, but not completely collapsed. Supervision/selection-boundary adequacy is unresolved.
- The semantic Reader screen completed at `five_ideas/quantum_semantic_reader_screen_100/20261008T101111Z/`. All 100-case metrics and trace/checkpoint/selection replay are audited. Reader/quantum semantic/classical semantic positive/direct/membership totals are each `203/138/355`; scalar quantum is `203/138/348`, with 16 changed Top-5 sets. Both declared gates fail; this is L0 only.
- Training used 434/512 usable weak-containment cases for three epochs. Both semantic heads collapse: projection L2 about `1.159 -> 1e-10 -> 1e-20 -> 1e-30`; all 5000 final residuals are zero, and every evaluation score vector equals Reader. A label-free Adam decay-only arithmetic probe nearly reproduces shrinkage. This configuration is blocked for training repair, not a negative test of a successfully trained semantic/quantum mechanism.
- GitHub was still at `b2ca094` on discovery; this repaired result was on 18083 only. Compact receipt-verified outputs are copied for GitHub publication. Executed source/config/plan match `b2ca094`; previous/current Reader scores and usable training identities match exactly. Historical configs/results stay frozen.
- Prior Q-ARCG full-panel/13-case audit is complete; no reliable gain, and some fixed-membership losses merely exchange Silver fillers. Missing compact provenance remains separate. Details are indexed in `critical-invariants.md`.
- Current work is result audit and next-direction recommendation; no new real-data experiment or head implementation. Formal full-data/replication and quantum attribution remain unestablished.

## Current Decision

- Selector-only boundary: `decisions/2026-09-05-selector-only-qore-qes.md`.
- Cumulative-evidence rule: `critical-invariants.md` and indexed shared decision.
- Closed candidate and failed fixed-slice gate: `decisions/2026-10-01-dpr-reader-span-relevance-screen.md`.
- Execution topology: `../../shared/decisions/2026-09-05-project-execution-topology.md`.
- Current result follow-up: `decisions/2026-10-08-qarcg-screen-trace-gate.md`.
- Metric distinction and provisional semantic-head direction: `decisions/2026-10-08-silver-membership-vs-evidence-retention.md`.
- Current repair gate: `decisions/2026-10-08-semantic-reader-training-collapse.md`; frozen design: `decisions/2026-10-08-quantum-semantic-reader-screen.md`.
- Qualified repair and next controlled measurement: `decisions/2026-10-08-semantic-reader-optimizer-repair.md`.
- Current outcome/next gate: `decisions/2026-10-09-semantic-reader-repair-result.md`.

## Next Actions

- Recommend train-only supervision and boundary-discrimination audit before a new implementation: deterministic high-score weak negatives versus lower-score positives, support-versus-containment and simple compression control. Provisional next direction is boundary-discriminative quantum semantic training, not blind amplification/deeper circuits; exploratory only pending support/readiness evidence and owner approval. Complete train text requires a bounded collaborator-local export; do not use exposed eval Silver as supervision.
- Separately recover the prior Q-ARCG run's `summary.json`, `report.md`, `run_metadata.json`, and `upload_manifest.json`. The completed 13-case join is not to be repeated; the new run does not reconstruct missing old provenance.
- Retain the prior span/relevance screen's separate publication repair: compact files were not recovered and its manifest runner hash differed from the committed runner hash.

## Blockers

- Official gold passage identity/usable-context coverage remains unresolved; Silver labels remain L0 diagnostics.
- Training is numerically active, but other 432 semantic inputs/raw support texts and full all-step optimizer states are absent. Recorded two-case traces cannot certify supervision quality. E-drive disk pressure interrupted local raw download; the partial NPZ is explicitly marked and excluded from replay.
- Prior Q-ARCG training count/loss and executed revision remain unverified; weak-target quality versus representation/optimization attribution remains inconclusive.

## Validation

- Latest: all six artifact receipts/source/config/plan/model/input identities, 100-case statistics and nine tensor/selection checks pass on server, including 18 optimizer-update witnesses. Prior baseline scores/cohort/input parity exact. Independent local all-100 metrics/bootstrap recompute; seven new audit tests pass locally/server. Recovery validates; L0 card retains expected inconclusive/attribution warnings. No data/model download, fitting or new task run.

- Repair: 44/44 focused tests locally (`lambeq`) and on canonical server (`py310`); original baseline and repaired fixtures reopen/replay, including repaired optimizer updates. Four-plugin plan/config/Bash checks pass. Independent server train-only qualification agrees on policy outcomes (not byte-identical across Torch versions). Hash-protected rollback checks pass and actual rollback on a disposable copy restores baseline. Raw hashes retained; CRLF-normalized qualification identity fixes cross-platform line-ending mismatch.

- Original collapse audit and repair qualification remain recorded in their linked verification files; historical gates/outputs are unchanged.
- Canonical checkout remains `/home/Q-DUET-VLM/QORE-VLM-phase1-gate` on `five-ideas-development`. Raw current cache is `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/semantic_reader_20261008T101111Z/`, outside Git; authoritative raw files stay on 18083. Unrelated changes are preserved; no dataset/model download or new task experiment on our servers.

## Pointers

- Repair module/tests: `applications/rag/semantic_reader_training.py`, `applications/rag/tests/test_semantic_reader_training.py`; new runner/wrapper: `scripts/collab/five_ideas/run_semantic_reader_training_repair_100.py` and `.sh`.
- Current audit: `refs/semantic_reader_repair_20261008T145652Z_audit.json`; recovery/card: `refs/semantic_reader_repair_20261009_recovery.json`, `refs/semantic_reader_repair_20261009_evidence_card.json`; verifier: `refs/semantic_reader_repair_result_verification_20261009.md`; log: `docs/rag-research-log/20261009T-date-only-semantic-reader-repair-result.md`.

- Semantic mechanism/tests: `applications/rag/quantum_semantic_reader.py`, `applications/rag/tests/test_quantum_semantic_reader.py`; runner/wrapper: `scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py` and `.sh`.
- Current audit: `refs/quantum_semantic_reader_20261008T101111Z_audit.json`; recovery: `refs/semantic_reader_training_collapse_recovery_20261008.json`; card: `refs/quantum_semantic_reader_20261008_evidence_card.json`.
- Owner log: `docs/rag-research-log/20261008T-101111Z-quantum-semantic-reader-training-collapse.md`; offline tool/tests: `scripts/collab/five_ideas/analyze_quantum_semantic_reader_screen.py`, `test_semantic_reader_audit.py`.
- Frozen config/plan: `configs/experiments/quantum_semantic_reader_screen_100.json` and `_plan.json`; original recovery: `refs/semantic_reader_recovery_20261008.json`.
- Q-ARCG result audit: `refs/qarcg_reader_screen_20261007_trace_audit.md`; owner log: `docs/rag-research-log/20261008T-date-only-qarcg-screen-trace-audit.md`.
- Paired semantic audit: `refs/qarcg_screen_20261007_semantic_case_audit.md` and `refs/qarcg_screen_20261007_paired_case_audit.json`; log: `docs/rag-research-log/20261008T-date-only-qarcg-paired-case-analysis.md`; helper: `scripts/collab/five_ideas/analyze_qarcg_reader_screen_cases.py`.
- Verification record: `refs/quantum_semantic_reader_preflight_20261008.md`.
- Current result verification: `refs/quantum_semantic_reader_result_verification_20261008.md`.
- Latest result session: `sessions/2026-10/20261009T003323Z-0d8405.md`; repair implementation: `sessions/2026-10/20261008T130613Z-9d0ac2.md`.
