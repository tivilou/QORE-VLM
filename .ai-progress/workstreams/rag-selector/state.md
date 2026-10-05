# RAG Passage Selection with QORE

## Goal

- Improve fixed Top-50 -> Top-5 selection with defensible utility evidence; keep retrieval, production selector, Generator, evaluator, labels, and answer path fixed.

## Current State

- No real-data L1/L2 selector result exists. Prior selector/scorer screens and cumulative lessons are indexed in `critical-invariants.md`.
- The fixed 100 x 50 DPR Reader trace is complete at `five_ideas/dpr_reader_tensor_trace_100/20260930T143444Z/`. It exposed some low-ranked direct-consensus passages with strong span logits and a separate fragment-redundancy pattern; this is L0 evidence.
- The 100-case label-blind span/relevance screen completed on 2026-10-04. Mean Silver overlap was Top-k `3.55/5`, QORE `2.59/5`, classical fusion `2.70/5`, and fixed quantum interaction `2.20/5`; the quantum and exact Born controls selected identically. The preregistered gate failed.
- The screen was a post-Reader feature test, not a quantum component integrated into DPR. The circuit had zero trainable parameters and an exact classical equivalent; no quantum advantage or real-data utility result is established.
- Q-ARCG (Quantum Applicability-gated Residual Calibration) is implemented as the next exploratory Reader-integrated candidate. It preserves raw Reader relevance as an exact null anchor and adds a trainable four-qubit applicability-gated bounded residual with a same-budget classical control. Synthetic preflight passed locally and on the 3080Ti Torch environment; this remains L0 engineering evidence only.
- The approved collaborator screen is implemented: it uses up to 512 `nq_open/train` weak answer-string containment cases for head-only training, and the registered 100 x 50 detail artifact for evaluation-only ranking. Silver is opened only after all online rankings complete; Generator/evaluator are not called.

## Current Decision

- Selector-only boundary: `decisions/2026-09-05-selector-only-qore-qes.md`.
- Cumulative-evidence rule: `critical-invariants.md` and indexed shared decision.
- Closed candidate and failed fixed-slice gate: `decisions/2026-10-01-dpr-reader-span-relevance-screen.md`.
- Execution topology: `../../shared/decisions/2026-09-05-project-execution-topology.md`.

## Next Actions

- Repair publication/provenance: compact `result.json`, `report.md`, and `run_metadata.json` are listed in the manifest but were not present on 18083, and the manifest runner hash differs from the committed runner hash.
- Hand the collaborator the one-command Q-ARCG screen wrapper and inspect the GitHub compact outputs plus 18083 selector trace. Do not treat a positive Silver diagnostic as L1/L2 or a quantum-advantage claim.

## Blockers

- Official gold passage identity/usable-context coverage remains unresolved; Silver labels remain L0 diagnostics.
- GitHub compact result publication is not yet independently recovered; local GitHub access was unavailable during inspection.
- No model or Wiki-DPR download was needed for this frozen-trace screen, and no Generator/evaluator call occurred. The Q-ARCG preflight likewise loaded no task data or corpus; only the server-side Torch differentiability check ran on synthetic tensors.

## Validation

- Nine Q-ARCG core tests plus four screen contract tests pass locally/remote (13/13 on the 3080Ti); Python compilation, JSON parsing, Bash syntax, `--validate-only`, and the three-plugin plan validator pass.
- Uploaded trace is complete for 100 x 50 candidates; the paired gate and redundancy metrics were independently recomputed from the trace and pinned detail input.

## Pointers

- Scoring module: `applications/rag/dpr_span_relevance_quantum.py`.
- Runner/wrapper: `scripts/collab/five_ideas/run_dpr_span_relevance_quantum_screen_100.py` and `.sh`.
- Config and plugin plan: `configs/experiments/dpr_span_relevance_quantum_screen_100.json` and `_plan.json`.
- Reader trace analysis: `refs/dpr_reader_tensor_trace_20260930T143444Z_analysis.md`.
- Screen result analysis: `refs/dpr_reader_span_relevance_screen_100_analysis.md`.
- Q-ARCG decision and implementation gate: `decisions/2026-10-05-qarcg-reader-integrated-residual.md`.
- Q-ARCG mechanism dossier: `refs/mechanism_recovery_2026-10-05-q-arcg.json`.
- Q-ARCG contract: `applications/rag/qarcg_reader.py`; synthetic preflight: `scripts/collab/five_ideas/run_qarcg_reader_preflight.sh`.
- Q-ARCG screen: `scripts/collab/five_ideas/run_qarcg_reader_screen_100.sh`; config/plan: `configs/experiments/qarcg_reader_screen_100.json` and `_plan.json`; decision: `decisions/2026-10-05-qarcg-reader-screen-implementation.md`.
- Latest session: `sessions/2026-10/20261005T110326Z-c146a1.md`.
