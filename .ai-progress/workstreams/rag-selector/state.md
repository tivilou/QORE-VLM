# RAG Passage Selection with QORE

## Goal

- Improve fixed Top-50 -> Top-5 with defensible utility evidence while retaining the quantum research line; keep retrieval, production selector, Generator, evaluator, labels, and answer path fixed.

## Current State

- No real-data L1/L2 selector result exists. Prior selector/scorer screens and cumulative lessons are indexed in `critical-invariants.md`.
- The semantic Reader screen completed at `five_ideas/quantum_semantic_reader_screen_100/20261008T101111Z/`. All 100-case metrics and trace/checkpoint/selection replay are audited. Reader/quantum semantic/classical semantic positive/direct/membership totals are each `203/138/355`; scalar quantum is `203/138/348`, with 16 changed Top-5 sets. Both declared gates fail; this is L0 only.
- Training used 434/512 usable weak-containment cases for three epochs. Both semantic heads collapse: projection L2 about `1.159 -> 1e-10 -> 1e-20 -> 1e-30`; all 5000 final residuals are zero, and every evaluation score vector equals Reader. A label-free Adam decay-only arithmetic probe nearly reproduces shrinkage. This configuration is blocked for training repair, not a negative test of a successfully trained semantic/quantum mechanism.
- The GitHub submission `44c9f70` is a tokenizer compatibility fix; compact results were initially exchange-only and have now been copied for focused GitHub publication. Reported HEAD is `5ce58ba` while executed semantic source hash matches `44c9f70`; file-level identity is recovered. Canonical server and local mirror have the fix. No repair implementation or new task run occurred.
- The collaborator uploaded the Q-ARCG screen trace at `five_ideas/qarcg_reader_screen_100/20261007T155432Z/`: Reader/Q-ARCG/classical mean Silver overlap is `3.55/3.49/3.53`. Q-ARCG versus Reader is `2/90/8` better/equal/worse; both positive screen gates fail. This configuration is not promoted; the broader training-dependent mechanism remains inconclusive.
- Full-panel/13-changed-case semantic audit is complete. The Silver reference contains fillers (61 questions have fewer than five positive candidates). Q-ARCG/Reader positive evidence is `204/203`, direct `138/138`, unanimous positive `171/171`; no reliable gain. Six of eight fixed-membership losses are not positive/direct evidence losses.
- Current work is result audit and durable lesson publication only. Formal full-data/replication and quantum attribution remain unestablished.

## Current Decision

- Selector-only boundary: `decisions/2026-09-05-selector-only-qore-qes.md`.
- Cumulative-evidence rule: `critical-invariants.md` and indexed shared decision.
- Closed candidate and failed fixed-slice gate: `decisions/2026-10-01-dpr-reader-span-relevance-screen.md`.
- Execution topology: `../../shared/decisions/2026-09-05-project-execution-topology.md`.
- Current result follow-up: `decisions/2026-10-08-qarcg-screen-trace-gate.md`.
- Metric distinction and provisional semantic-head direction: `decisions/2026-10-08-silver-membership-vs-evidence-retention.md`.
- Current repair gate: `decisions/2026-10-08-semantic-reader-training-collapse.md`; frozen design: `decisions/2026-10-08-quantum-semantic-reader-screen.md`.

## Next Actions

- Recommend repair/qualification of the null-compatible learning path and regularization groups using saved train-only representations; require preserved feature variation, nonzero finite data gradients after null, and effective float32 score corrections before proposing another collaborator screen. Await user authorization for training-code repair/rerun. Do not enlarge data/epochs/circuit depth or tune on Silver to conceal collapse.
- Separately recover the prior Q-ARCG run's `summary.json`, `report.md`, `run_metadata.json`, and `upload_manifest.json`. The completed 13-case join is not to be repeated; the new run does not reconstruct missing old provenance.
- Retain the prior span/relevance screen's separate publication repair: compact files were not recovered and its manifest runner hash differed from the committed runner hash.

## Blockers

- Official gold passage identity/usable-context coverage remains unresolved; Silver labels remain L0 diagnostics.
- Semantic learning path is degenerate; full per-step task gradients/Adam state and other 432 semantic train inputs are absent. Decay-only agreement is strong diagnostic evidence, not full causal optimizer replay. Weak-label adequacy remains unresolved.
- Prior Q-ARCG training count/loss and executed revision remain unverified; weak-target quality versus representation/optimization attribution remains inconclusive.

## Validation

- Independent current audit on local and canonical server: all six artifact/manifest receipts and source/config/plan/input hashes pass; 100x50 identity/order, usable train/eval disjointness, primary/protection/bootstrap reports and all producer-declared tensor/checkpoint/Top-5 replays pass. Combined regression 33/33 on both, including five new audit tests. Recovery validator passes; evidence card valid at L0 with explicit inconclusive/attribution warnings (strict mode intentionally does not pass).
- Canonical checkout remains `/home/Q-DUET-VLM/QORE-VLM-phase1-gate` on `five-ideas-development`. Raw current cache is `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/semantic_reader_20261008T101111Z/`, outside Git; authoritative raw files stay on 18083. Unrelated changes are preserved; no dataset/model download or new task experiment on our servers.

## Pointers

- Semantic mechanism/tests: `applications/rag/quantum_semantic_reader.py`, `applications/rag/tests/test_quantum_semantic_reader.py`; runner/wrapper: `scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py` and `.sh`.
- Current audit: `refs/quantum_semantic_reader_20261008T101111Z_audit.json`; recovery: `refs/semantic_reader_training_collapse_recovery_20261008.json`; card: `refs/quantum_semantic_reader_20261008_evidence_card.json`.
- Owner log: `docs/rag-research-log/20261008T-101111Z-quantum-semantic-reader-training-collapse.md`; offline tool/tests: `scripts/collab/five_ideas/analyze_quantum_semantic_reader_screen.py`, `test_semantic_reader_audit.py`.
- Frozen config/plan: `configs/experiments/quantum_semantic_reader_screen_100.json` and `_plan.json`; original recovery: `refs/semantic_reader_recovery_20261008.json`.
- Q-ARCG result audit: `refs/qarcg_reader_screen_20261007_trace_audit.md`; owner log: `docs/rag-research-log/20261008T-date-only-qarcg-screen-trace-audit.md`.
- Paired semantic audit: `refs/qarcg_screen_20261007_semantic_case_audit.md` and `refs/qarcg_screen_20261007_paired_case_audit.json`; log: `docs/rag-research-log/20261008T-date-only-qarcg-paired-case-analysis.md`; helper: `scripts/collab/five_ideas/analyze_qarcg_reader_screen_cases.py`.
- Verification record: `refs/quantum_semantic_reader_preflight_20261008.md`.
- Current result verification: `refs/quantum_semantic_reader_result_verification_20261008.md`.
- Latest audit session: `sessions/2026-10/20261008T114922Z-c7fb30.md`; prior publication: `sessions/2026-10/20261008T034644Z-e11204.md`.
