# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2. Training-only and small blind diagnostics remain L0.

## Current State
- [ACTIVE / L0 BLIND UTILITY GATE PASSED] Two isolated reviewers completed all91 passages from the15 changed validation87 cases. They agreed on85 items; six disagreements were resolved by a third blind adjudication before the method key was opened.
- On changed cases, Reader / quantum / matched classical retained useful evidence `38 / 41 / 40` and direct evidence `25 / 25 / 25`. Quantum versus Reader was `4 wins / 1 loss / 10 ties`, point delta `+3`; versus classical it was `2 / 1 / 12`, delta `+1`.
- All five prospective checks passed. This is the first independent evidence that the frozen quantum head's unseen selection changes are beneficial, but it is still a small L0 audit:72/87 cases were unchanged, no L1/L2 or quantum-advantage claim follows.
- Review files, blind consensus and score are hash-verified on the3080Ti aggregation host. GPU work remains assigned only through the live-checked `gpu-training` role.

## Plan Control
- Completed Plan: plans/reader-unseen-blind-review/v001.json
- Plan Version: 1
- Plan Status: complete
- Completed Steps: BR-01 through BR-06
- Next Action: Freeze a new larger untouched question cohort and its utility gate before extracting features or inspecting outcomes.
- Blocker: No larger untouched cohort or preregistered contract exists yet.
- Last Checkpoint: 2026-10-18

## Next Actions
1. Design a fresh, non-overlapping >=200-question fixed-Top50 cohort; exclude all train59, validation87 and repeatedly inspected 100-case panel identities.
2. Freeze the source/model/revision, trained quantum and matched-classical checkpoints, changed-case review rule, uncertainty analysis and stop gate before one-shot execution.
3. Run feature extraction on the bound live GPU host or collaborator machine, aggregate on3080Ti, then repeat method-blind semantic review only for changed-set unions.

## Blockers
- A provenance-locked fresh cohort with sufficient question count is not yet materialized.
- The present effect is too small and selected-case-conditioned to establish population utility or quantum attribution.

## Validation
- Review A/B:91/91 each; label uncertainty0;85 agreements and6 blind adjudications; final consensus SHA `6834692a...0b38`.
- Score SHA `e5f88b14...d65e`; packet SHA `eb5aef15...e565`; sealed key SHA `bb26843f...b3b`.
- Blind workflow tests:3 passed after adding exact-coverage adjudication; no key is loaded by merge/adjudicate.
- Aggregated artifacts: `/home/Q-DUET-VLM/reader-membership-screen-20261016/exchange/five_ideas/reader_unseen_blind_review_87/20261018T073840422368Z` on3080Ti.
- Research log: [validation87 blind utility result](../../../docs/rag-research-log/20261018T-blind-unseen-utility-gate.md).
- Session: [two-reviewer blind utility audit](sessions/2026-10/20261018T081258Z-e3f3ff.md).

## Relevant Decisions
- [Blind utility gate passed; larger untouched study next](decisions/2026-10-18-reader-unseen-blind-review.md).
- [Completed untouched validation transport gate](decisions/2026-10-18-reader-unseen-transport-gate.md).
- [Completed sequential contract/result](decisions/2026-10-18-reader-sequential-protected-updates.md).
- [Frozen membership contract](decisions/2026-10-16-reader-membership-screen-handoff.md); [final train targets](decisions/2026-10-11-final-support-boundary-resolution.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
