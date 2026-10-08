# RAG Passage Selection with QORE

## Goal

- Improve fixed Top-50 -> Top-5 with defensible utility evidence while retaining the quantum research line; keep retrieval, production selector, Generator, evaluator, labels, and answer path fixed.

## Current State

- No real-data L1/L2 selector result exists. Prior selector/scorer screens and cumulative lessons are indexed in `critical-invariants.md`.
- Optimizer repair is implemented and train-only qualified: decoupled projection-only AdamW plus projection LR `0.003/sqrt(5H)`. On two saved training cases / 64 alternating updates, quantum/classical retain projection norm ratios `1.0189/1.0201`, encoding spreads `0.3028/0.3274`, and input-dependent effective float32 corrections. Parameters/moments/signals replay exactly. This is engineering qualification, not task utility.
- The semantic Reader screen completed at `five_ideas/quantum_semantic_reader_screen_100/20261008T101111Z/`. All 100-case metrics and trace/checkpoint/selection replay are audited. Reader/quantum semantic/classical semantic positive/direct/membership totals are each `203/138/355`; scalar quantum is `203/138/348`, with 16 changed Top-5 sets. Both declared gates fail; this is L0 only.
- Training used 434/512 usable weak-containment cases for three epochs. Both semantic heads collapse: projection L2 about `1.159 -> 1e-10 -> 1e-20 -> 1e-30`; all 5000 final residuals are zero, and every evaluation score vector equals Reader. A label-free Adam decay-only arithmetic probe nearly reproduces shrinkage. This configuration is blocked for training repair, not a negative test of a successfully trained semantic/quantum mechanism.
- The original GitHub submission `44c9f70` is a tokenizer compatibility fix; its compact results have been published and file-level executed identity recovered. The original config/plan/results remain frozen; the repair has a separate config/plan/output namespace `semantic_reader_training_repair_100`. No repaired task screen has run.
- The collaborator uploaded the Q-ARCG screen trace at `five_ideas/qarcg_reader_screen_100/20261007T155432Z/`: Reader/Q-ARCG/classical mean Silver overlap is `3.55/3.49/3.53`. Q-ARCG versus Reader is `2/90/8` better/equal/worse; both positive screen gates fail. This configuration is not promoted; the broader training-dependent mechanism remains inconclusive.
- Full-panel/13-changed-case semantic audit is complete. The Silver reference contains fillers (61 questions have fewer than five positive candidates). Q-ARCG/Reader positive evidence is `204/203`, direct `138/138`, unanimous positive `171/171`; no reliable gain. Six of eight fixed-membership losses are not positive/direct evidence losses.
- A new one-command collaborator screen is ready. It checks train-only health at update 64 / epoch ends, saves optimizer-update witnesses, and stops/preserves/uploads a failed checkpoint before evaluation. Formal full-data/replication and quantum attribution remain unestablished.

## Current Decision

- Selector-only boundary: `decisions/2026-09-05-selector-only-qore-qes.md`.
- Cumulative-evidence rule: `critical-invariants.md` and indexed shared decision.
- Closed candidate and failed fixed-slice gate: `decisions/2026-10-01-dpr-reader-span-relevance-screen.md`.
- Execution topology: `../../shared/decisions/2026-09-05-project-execution-topology.md`.
- Current result follow-up: `decisions/2026-10-08-qarcg-screen-trace-gate.md`.
- Metric distinction and provisional semantic-head direction: `decisions/2026-10-08-silver-membership-vs-evidence-retention.md`.
- Current repair gate: `decisions/2026-10-08-semantic-reader-training-collapse.md`; frozen design: `decisions/2026-10-08-quantum-semantic-reader-screen.md`.
- Qualified repair and next controlled measurement: `decisions/2026-10-08-semantic-reader-optimizer-repair.md`.

## Next Actions

- Collaborator updates `five-ideas-development`, retains the existing `QORE_EXCHANGE_TOKEN`, then runs `bash scripts/collab/five_ideas/run_semantic_reader_training_repair_100.sh` locally. No real-data run on our development servers. Analyze training health before unchanged evidence-utility / matched-classical gates; do not enlarge data, epochs or circuit depth.
- Separately recover the prior Q-ARCG run's `summary.json`, `report.md`, `run_metadata.json`, and `upload_manifest.json`. The completed 13-case join is not to be repeated; the new run does not reconstruct missing old provenance.
- Retain the prior span/relevance screen's separate publication repair: compact files were not recovered and its manifest runner hash differed from the committed runner hash.

## Blockers

- Official gold passage identity/usable-context coverage remains unresolved; Silver labels remain L0 diagnostics.
- Old training remains incompletely observable: other 432 semantic inputs and all-step optimizer states are absent. The two-case repair qualification does not guarantee health on the full training cohort or adequacy of weak labels; the new runner guards this before evaluation.
- Prior Q-ARCG training count/loss and executed revision remain unverified; weak-target quality versus representation/optimization attribution remains inconclusive.

## Validation

- Repair: 44/44 focused tests locally (`lambeq`) and on canonical server (`py310`); original baseline and repaired fixtures reopen/replay, including repaired optimizer updates. Four-plugin plan/config/Bash checks pass. Independent server train-only qualification agrees on policy outcomes (not byte-identical across Torch versions). Hash-protected rollback checks pass and actual rollback on a disposable copy restores baseline. Raw hashes retained; CRLF-normalized qualification identity fixes cross-platform line-ending mismatch.

- Independent current audit on local and canonical server: all six artifact/manifest receipts and source/config/plan/input hashes pass; 100x50 identity/order, usable train/eval disjointness, primary/protection/bootstrap reports and all producer-declared tensor/checkpoint/Top-5 replays pass. Combined regression 33/33 on both, including five new audit tests. Recovery validator passes; evidence card valid at L0 with explicit inconclusive/attribution warnings (strict mode intentionally does not pass).
- Canonical checkout remains `/home/Q-DUET-VLM/QORE-VLM-phase1-gate` on `five-ideas-development`. Raw current cache is `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/semantic_reader_20261008T101111Z/`, outside Git; authoritative raw files stay on 18083. Unrelated changes are preserved; no dataset/model download or new task experiment on our servers.

## Pointers

- Repair module/tests: `applications/rag/semantic_reader_training.py`, `applications/rag/tests/test_semantic_reader_training.py`; new runner/wrapper: `scripts/collab/five_ideas/run_semantic_reader_training_repair_100.py` and `.sh`.
- Qualification: `refs/semantic_reader_training_qualification_20261008.json`; verifier: `refs/semantic_reader_training_repair_verification_20261008.md`; log: `docs/rag-research-log/20261008T-date-only-semantic-reader-training-repair.md`.

- Semantic mechanism/tests: `applications/rag/quantum_semantic_reader.py`, `applications/rag/tests/test_quantum_semantic_reader.py`; runner/wrapper: `scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py` and `.sh`.
- Current audit: `refs/quantum_semantic_reader_20261008T101111Z_audit.json`; recovery: `refs/semantic_reader_training_collapse_recovery_20261008.json`; card: `refs/quantum_semantic_reader_20261008_evidence_card.json`.
- Owner log: `docs/rag-research-log/20261008T-101111Z-quantum-semantic-reader-training-collapse.md`; offline tool/tests: `scripts/collab/five_ideas/analyze_quantum_semantic_reader_screen.py`, `test_semantic_reader_audit.py`.
- Frozen config/plan: `configs/experiments/quantum_semantic_reader_screen_100.json` and `_plan.json`; original recovery: `refs/semantic_reader_recovery_20261008.json`.
- Q-ARCG result audit: `refs/qarcg_reader_screen_20261007_trace_audit.md`; owner log: `docs/rag-research-log/20261008T-date-only-qarcg-screen-trace-audit.md`.
- Paired semantic audit: `refs/qarcg_screen_20261007_semantic_case_audit.md` and `refs/qarcg_screen_20261007_paired_case_audit.json`; log: `docs/rag-research-log/20261008T-date-only-qarcg-paired-case-analysis.md`; helper: `scripts/collab/five_ideas/analyze_qarcg_reader_screen_cases.py`.
- Verification record: `refs/quantum_semantic_reader_preflight_20261008.md`.
- Current result verification: `refs/quantum_semantic_reader_result_verification_20261008.md`.
- Latest repair session: `sessions/2026-10/20261008T130613Z-9d0ac2.md`; prior audit: `sessions/2026-10/20261008T114922Z-c7fb30.md`.
