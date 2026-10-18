# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2. Training-only and small blind diagnostics remain L0.

## Current State
- [ACTIVE / L0; REAL COHORT PENDING] The larger untouched utility contract and cohort exporter are implemented for 256 fresh NQ validation questions, each with a fixed DPR/Wiki-DPR Top-50.
- The deterministic sample excludes exact question identities from the protected434 cohort and the historical100 panel before scoring. Answers, gold/Silver/support labels, generations, predictions and selector outcomes are forbidden from the cohort artifact.
- No real cohort, Reader features or method outcomes have been produced. The plugin plan therefore remains `approved`, not `implemented`, and no new utility claim follows.
- Execution is split deliberately: the collaborator's existing data-bearing runtime exports the cohort; a live `gpu-training` host later runs the pinned Reader and frozen heads; 3080Ti verifies hashes, hosts 18083 and aggregates results.

## Plan Control
- Active Plan: plans/reader-larger-untouched-utility/v001.json
- Plan Version: 1
- Plan Status: active
- Focus Step: LU-03 — materialize and freeze the real cohort
- Next Action: collaborator runs `bash scripts/collab/five_ideas/prepare_reader_larger_untouched_cohort_256.sh` from the project root and returns the compact GitHub manifest; large artifacts upload automatically to18083.
- Blocker: the real 256 x 50 cohort has not yet been materialized and hash-frozen.
- Last Checkpoint: 2026-10-18

## Frozen Gate
1. Sample 256 from `nq_open/validation[512:3610]` by seed-locked SHA-256 order after exact protected434 and historical100 exclusions.
2. Materialize exactly 50 unique DPR/Wiki-DPR candidates per question; freeze cohort identity before Reader scoring.
3. Continue to blind review only if quantum changes at least25 cases and differs from matched classical on at least15.
4. For promotion, require quantum useful totals +5 versus Reader and +3 versus classical, no direct-evidence decline versus Reader, case-win margins +3/+1, positive clustered-bootstrap lower bounds, and zero unresolved quantum-selected items.

## Validation
- Synthetic cohort/export tests: 3 passed.
- Plugin-plan validation: 3 plugins passed.
- Validate-only: 256 cases; `model_loaded=false`; `wiki_dpr_started=false`.
- Python compilation and Bash syntax validation passed.
- No full corpus was downloaded and no formal experiment was run locally or on3080Ti.
- Session: [larger untouched cohort export implementation](sessions/2026-10/20261018T164330Z-a4c7e1.md).
- GitHub `five-ideas-development` is published at `ed558d2`; [publication mapping](refs/reader_larger_untouched_utility_20261018_publication.json) records local/remote commits and validation.

## Relevant Decisions
- [Larger untouched utility study contract](decisions/2026-10-18-reader-larger-untouched-utility.md).
- [Blind validation87 utility result](decisions/2026-10-18-reader-unseen-blind-review.md).
- [Completed untouched validation transport gate](decisions/2026-10-18-reader-unseen-transport-gate.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
