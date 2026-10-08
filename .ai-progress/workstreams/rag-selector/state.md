# RAG Passage Selection with QORE

## Goal

- Improve fixed Top-50 -> Top-5 with defensible utility evidence while retaining the quantum research line; keep retrieval, production selector, Generator, evaluator, labels, and answer path fixed.

## Current State

- No real-data L1/L2 selector result exists. Prior selector/scorer screens and cumulative lessons are indexed in `critical-invariants.md`.
- The user-approved quantum semantic Reader head is implemented for collaborator execution: frozen final hidden states -> question-conditioned pooling -> trainable classical projection -> four-qubit gated residual anchored to raw relevance. Four arms: Reader, semantic quantum, same-budget semantic classical and scalar quantum. All trained arms use the same 512-question weak train cohort and three epochs. The body remains frozen, not end-to-end fine-tuned.
- Primary is positive-consensus retention, direct retention is protected, fixed Silver membership is secondary. This exposed 100-case panel is L0 only. All evaluation representations/head signals and selected training witnesses/checkpoints are saved, reopened and replayed; compact reports and raw artifacts upload automatically to a new 18083 timestamp directory.
- The collaborator uploaded the Q-ARCG screen trace at `five_ideas/qarcg_reader_screen_100/20261007T155432Z/`: Reader/Q-ARCG/classical mean Silver overlap is `3.55/3.49/3.53`. Q-ARCG versus Reader is `2/90/8` better/equal/worse; both positive screen gates fail. This configuration is not promoted; the broader training-dependent mechanism remains inconclusive.
- Full-panel/13-changed-case semantic audit is complete. The Silver reference contains fillers (61 questions have fewer than five positive candidates). Q-ARCG/Reader positive evidence is `204/203`, direct `138/138`, unanimous positive `171/171`; no reliable gain. Six of eight fixed-membership losses are not positive/direct evidence losses.
- Implementation and prior paired audits were published to GitHub `five-ideas-development` at `36006e60d4168e98f7777b48b76a5ce2b7c3e7fe`; documentation follow-up does not alter experiment code. Real-data execution remains pending collaborator action.

## Current Decision

- Selector-only boundary: `decisions/2026-09-05-selector-only-qore-qes.md`.
- Cumulative-evidence rule: `critical-invariants.md` and indexed shared decision.
- Closed candidate and failed fixed-slice gate: `decisions/2026-10-01-dpr-reader-span-relevance-screen.md`.
- Execution topology: `../../shared/decisions/2026-09-05-project-execution-topology.md`.
- Current result follow-up: `decisions/2026-10-08-qarcg-screen-trace-gate.md`.
- Metric distinction and provisional semantic-head direction: `decisions/2026-10-08-silver-membership-vs-evidence-retention.md`.
- Active implementation: `decisions/2026-10-08-quantum-semantic-reader-screen.md`.

## Next Actions

- Collaborator runs `bash scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.sh` after syncing the development branch, in their existing experiment environment with exchange token configured. Verify training coverage/loss/model/code/config/receipts first, then paired primary/direct gates and rescued/harmed cases against cumulative lessons. No real-data run or dataset/model download occurred on the development server; formal novelty and independent-data gates remain pending.
- Separately recover the prior Q-ARCG run's `summary.json`, `report.md`, `run_metadata.json`, and `upload_manifest.json`. The completed 13-case join is not to be repeated; the new run does not reconstruct missing old provenance.
- Retain the prior span/relevance screen's separate publication repair: compact files were not recovered and its manifest runner hash differed from the committed runner hash.

## Blockers

- Official gold passage identity/usable-context coverage remains unresolved; Silver labels remain L0 diagnostics.
- Prior Q-ARCG training count/loss and executed revision remain unverified; weak-target quality versus representation/optimization attribution remains inconclusive.

## Validation

- Local and primary 3080Ti combined regression both 28/28 after prior audit helper sync. Tiny real DPRReader API tests use random local weights, no checkpoint download. Four-plugin plan and mechanism-recovery validators pass. Local/server synthetic detached-value, pooling, checkpoint/witness/eval, exact-null and JSON/selection replay pass; server Python compile, Bash syntax, locked config and one-command wrapper checks pass. Exchange-only trace/NPZ ignore rules verified.
- SSH is restored. Canonical checkout remains `/home/Q-DUET-VLM/QORE-VLM-phase1-gate` on `five-ideas-development`. Raw audit/source artifacts remain outside Git in `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/`. Synthetic fixtures: local `semantic_reader_preflight/` in that cache; server `/tmp/qore-semantic-reader-preflight/`. Unrelated user changes are preserved.

## Pointers

- Semantic mechanism/tests: `applications/rag/quantum_semantic_reader.py`, `applications/rag/tests/test_quantum_semantic_reader.py`; runner/wrapper: `scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py` and `.sh`.
- Semantic config/plan: `configs/experiments/quantum_semantic_reader_screen_100.json` and `_plan.json`; recovery: `refs/semantic_reader_recovery_20261008.json`; owner log: `docs/rag-research-log/20261008T-date-only-quantum-semantic-reader-implementation.md`.
- Q-ARCG result audit: `refs/qarcg_reader_screen_20261007_trace_audit.md`; owner log: `docs/rag-research-log/20261008T-date-only-qarcg-screen-trace-audit.md`.
- Paired semantic audit: `refs/qarcg_screen_20261007_semantic_case_audit.md` and `refs/qarcg_screen_20261007_paired_case_audit.json`; log: `docs/rag-research-log/20261008T-date-only-qarcg-paired-case-analysis.md`; helper: `scripts/collab/five_ideas/analyze_qarcg_reader_screen_cases.py`.
- Verification record: `refs/quantum_semantic_reader_preflight_20261008.md`.
- Latest session: `sessions/2026-10/20261008T034644Z-e11204.md`; implementation outcome: `sessions/2026-10/20261008T034024Z-24e3d9.md`.
