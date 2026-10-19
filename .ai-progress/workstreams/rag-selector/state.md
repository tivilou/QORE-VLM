# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2. Training-only and small blind diagnostics remain L0.

## Current State
- [ACTIVE / L0 DATA FREEZE PASSED] Real run `20261018T141409477054Z` froze 256 fresh NQ validation questions and 12,800 DPR/Wiki-DPR candidates. All questions are unique; every case has 50 unique ranked candidates with finite scores.
- Independent 3080Ti replay found zero overlap with the protected434 plus historical100 identities, zero forbidden answer/label/result fields, and exact cohort/lock/manifest/receipt/source agreement. Cohort SHA is `49c57f20...455a02`; identity SHA is `25a71785...25397`.
- This is not a selector result: Reader, quantum and classical scoring have not run, so no utility, L1/L2 or quantum-advantage conclusion exists.
- GitHub has no collaborator commit after `5815c74`; all required exchange artifacts are complete on18083, but the compact GitHub return is absent.

## Plan Control
- Active Plan: plans/reader-larger-untouched-utility/v001.json
- Plan Version: 1
- Plan Status: active
- Completed Steps: LU-01 through LU-03
- Focus Step: LU-04 — verify cohort and freeze transport inputs
- Next Action: bind the verified cohort SHA, pinned Reader revision, exact sequential quantum/classical checkpoint hashes and source identities into one immutable scoring manifest; validate it before GPU scoring.
- Blocker: None.
- Last Checkpoint: 2026-10-19

## Frozen Gate
1. Cohort identity and Top-50 are immutable; no post-score resampling or replacement.
2. Continue to blind review only if quantum changes at least25 cases and differs from matched classical on at least15.
3. For promotion, require quantum useful totals +5 versus Reader and +3 versus classical, no direct-evidence decline versus Reader, case-win margins +3/+1, positive clustered-bootstrap lower bounds, and zero unresolved quantum-selected items.

## Validation
- Receipt: [larger untouched cohort verification](refs/reader_larger_untouched_cohort_20261018_receipt.json).
- Research log: [256-question cohort freeze](../../../docs/rag-research-log/20261018T-141409Z-reader-larger-untouched-cohort-freeze.md).
- Exchange path: `/srv/qore-collab-uploads/five_ideas/reader_larger_untouched_cohort_256/20261018T141409477054Z` on3080Ti.
- Six archived executed sources match Git revision `5815c74`; four declared upload payloads match service receipts by bytes and SHA-256.
- Session: [real cohort verification and plan advance](sessions/2026-10/20261019T001305Z-200087.md).

## Relevant Decisions
- [Larger untouched utility study contract and checkpoint](decisions/2026-10-18-reader-larger-untouched-utility.md).
- [Blind validation87 utility result](decisions/2026-10-18-reader-unseen-blind-review.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
