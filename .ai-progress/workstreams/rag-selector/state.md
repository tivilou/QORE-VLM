# RAG Passage Selection with QORE

## Goal

- Improve fixed Top-50 -> Top-5 selection with defensible utility evidence; keep retrieval, production selector, Generator, evaluator, labels, and answer path fixed.

## Current State

- No real-data L1/L2 selector result exists. Prior selector/scorer screens and cumulative lessons are indexed in `critical-invariants.md`.
- The user-approved quantum semantic Reader head is implemented for collaborator execution: frozen final hidden states -> question-conditioned pooling -> trainable classical projection -> four-qubit gated residual anchored to raw relevance. Four arms: Reader, semantic quantum, same-budget semantic classical and scalar quantum. All trained arms use the same 512-question weak train cohort and three epochs. The body remains frozen, not end-to-end fine-tuned.
- Primary is positive-consensus retention, direct retention is protected, fixed Silver membership is secondary. This exposed 100-case panel is L0 only. All evaluation representations/head signals and selected training witnesses/checkpoints are saved, reopened and replayed; compact reports and raw artifacts upload automatically to a new 18083 timestamp directory.
- The fixed 100 x 50 DPR Reader trace is complete at `five_ideas/dpr_reader_tensor_trace_100/20260930T143444Z/`. It exposed some low-ranked direct-consensus passages with strong span logits and a separate fragment-redundancy pattern; this is L0 evidence.
- The 100-case label-blind span/relevance screen completed on 2026-10-04. Mean Silver overlap was Top-k `3.55/5`, QORE `2.59/5`, classical fusion `2.70/5`, and fixed quantum interaction `2.20/5`; the quantum and exact Born controls selected identically. The preregistered gate failed.
- The screen was a post-Reader feature test, not a quantum component integrated into DPR. The circuit had zero trainable parameters and an exact classical equivalent; no quantum advantage or real-data utility result is established.
- Q-ARCG (Quantum Applicability-gated Residual Calibration) is implemented as the next exploratory Reader-integrated candidate. It preserves raw Reader relevance as an exact null anchor and adds a trainable four-qubit applicability-gated bounded residual with a same-budget classical control. Synthetic preflight passed locally and on the 3080Ti Torch environment; this remains L0 engineering evidence only.
- The approved collaborator screen is implemented: it uses up to 512 `nq_open/train` weak answer-string containment cases for head-only training, and the registered 100 x 50 detail artifact for evaluation-only ranking. Silver is opened only after all online rankings complete; Generator/evaluator are not called.
- The collaborator uploaded the Q-ARCG screen trace at `five_ideas/qarcg_reader_screen_100/20261007T155432Z/`: Reader/Q-ARCG/classical mean Silver overlap is `3.55/3.49/3.53`. Q-ARCG versus Reader is `2/90/8` better/equal/worse; both positive screen gates fail. This configuration is not promoted; the broader training-dependent mechanism remains inconclusive.
- Full-panel/13-changed-case semantic audit is complete. The Silver reference contains fillers (61 questions have fewer than five positive candidates). Q-ARCG/Reader positive evidence is `204/203`, direct `138/138`, unanimous positive `171/171`; no reliable gain. Six of eight fixed-membership losses are not positive/direct evidence losses.

## Current Decision

- Selector-only boundary: `decisions/2026-09-05-selector-only-qore-qes.md`.
- Cumulative-evidence rule: `critical-invariants.md` and indexed shared decision.
- Closed candidate and failed fixed-slice gate: `decisions/2026-10-01-dpr-reader-span-relevance-screen.md`.
- Execution topology: `../../shared/decisions/2026-09-05-project-execution-topology.md`.
- Current result follow-up: `decisions/2026-10-08-qarcg-screen-trace-gate.md`.
- Metric distinction and provisional semantic-head direction: `decisions/2026-10-08-silver-membership-vs-evidence-retention.md`.
- Active implementation: `decisions/2026-10-08-quantum-semantic-reader-screen.md`.

## Next Actions

- Recover this run's already generated `summary.json`, `report.md`, `run_metadata.json`, and `upload_manifest.json`; neither inspected GitHub branch nor 18083 currently contains them. Verify training coverage/loss and executed code/config identity before attributing the result to weak labels versus representation/optimization. The 13-case join is complete; do not repeat it.
- Finish focused publication, then collaborator runs `bash scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.sh` in their existing experiment environment with exchange token configured. Verify training coverage/loss/model/code/config/receipts first, then paired primary/direct gates and rescued/harmed cases against cumulative lessons. No real-data run or dataset/model download occurred on the development server; formal novelty and independent-data gates remain pending.
- Retain the prior span/relevance screen's separate publication repair: compact files were not recovered and its manifest runner hash differed from the committed runner hash.

## Blockers

- Official gold passage identity/usable-context coverage remains unresolved; Silver labels remain L0 diagnostics.
- Prior Q-ARCG training count/loss and executed revision remain unverified. New semantic implementation is validated on the canonical server, pending focused publication confirmation.

## Validation

- Nine Q-ARCG core tests plus four screen contract tests pass locally/remote (13/13 on the 3080Ti); Python compilation, JSON parsing, Bash syntax, `--validate-only`, and the three-plugin plan validator pass.
- Uploaded trace is complete for 100 x 50 candidates; the paired gate and redundancy metrics were independently recomputed from the trace and pinned detail input.
- Q-ARCG trace hash matches upload receipt; all 100 cases have 50 finite scores per arm and five unique selected candidates. Scaled residual bounds pass; bootstrap recomputation uses the declared 2000 replicates and seeds. Overall mechanism observability remains partial because training and per-candidate head signals are absent.
- Offline paired-case helper: 7/7 contract tests pass in existing `lambeq` environment; registered 100x50 identity/order and label-ceiling checks pass. Historical Reader baseline drift is at most `0.0000567`. Service downloads reused existing artifacts only.
- Semantic head: local combined regression 28/28; primary 3080Ti existing py310 core/screen/semantic tests 21/21. Tiny real DPRReader API tests use random local weights, no checkpoint download. Four-plugin plan and mechanism-recovery validators pass. Local/server synthetic detached-value, pooling, checkpoint/witness/eval, exact-null and JSON/selection replay pass; server Python compile, Bash syntax and locked config pass.
- SSH is restored. Canonical checkout remains `/home/Q-DUET-VLM/QORE-VLM-phase1-gate` on `five-ideas-development`. Raw audit/source artifacts remain outside Git in `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/`. Synthetic fixtures: local `semantic_reader_preflight/` in that cache; server `/tmp/qore-semantic-reader-preflight/`. Unrelated user changes are preserved.

## Pointers

- Semantic mechanism/tests: `applications/rag/quantum_semantic_reader.py`, `applications/rag/tests/test_quantum_semantic_reader.py`; runner/wrapper: `scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py` and `.sh`.
- Semantic config/plan: `configs/experiments/quantum_semantic_reader_screen_100.json` and `_plan.json`; recovery: `refs/semantic_reader_recovery_20261008.json`; owner log: `docs/rag-research-log/20261008T-date-only-quantum-semantic-reader-implementation.md`.
- Scoring module: `applications/rag/dpr_span_relevance_quantum.py`.
- Runner/wrapper: `scripts/collab/five_ideas/run_dpr_span_relevance_quantum_screen_100.py` and `.sh`.
- Config and plugin plan: `configs/experiments/dpr_span_relevance_quantum_screen_100.json` and `_plan.json`.
- Reader trace analysis: `refs/dpr_reader_tensor_trace_20260930T143444Z_analysis.md`.
- Screen result analysis: `refs/dpr_reader_span_relevance_screen_100_analysis.md`.
- Q-ARCG decision and implementation gate: `decisions/2026-10-05-qarcg-reader-integrated-residual.md`.
- Q-ARCG mechanism dossier: `refs/mechanism_recovery_2026-10-05-q-arcg.json`.
- Q-ARCG contract: `applications/rag/qarcg_reader.py`; synthetic preflight: `scripts/collab/five_ideas/run_qarcg_reader_preflight.sh`.
- Q-ARCG screen: `scripts/collab/five_ideas/run_qarcg_reader_screen_100.sh`; config/plan: `configs/experiments/qarcg_reader_screen_100.json` and `_plan.json`; decision: `decisions/2026-10-05-qarcg-reader-screen-implementation.md`.
- Q-ARCG result audit: `refs/qarcg_reader_screen_20261007_trace_audit.md`; owner log: `docs/rag-research-log/20261008T-date-only-qarcg-screen-trace-audit.md`.
- Paired semantic audit: `refs/qarcg_screen_20261007_semantic_case_audit.md` and `refs/qarcg_screen_20261007_paired_case_audit.json`; log: `docs/rag-research-log/20261008T-date-only-qarcg-paired-case-analysis.md`; helper: `scripts/collab/five_ideas/analyze_qarcg_reader_screen_cases.py`.
- Verification record: `refs/quantum_semantic_reader_preflight_20261008.md`.
- Latest session: `sessions/2026-10/20261008T034024Z-24e3d9.md`.
