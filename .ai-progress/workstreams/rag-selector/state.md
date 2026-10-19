# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2. Training-only and small blind diagnostics remain L0.

## Current State
- [ACTIVE / L0 INPUTS FROZEN] The 256-question x Top-50 cohort, DPR Reader revision, five Reader snapshot files, and frozen sequential quantum/classical restored states are bound by one immutable scoring manifest.
- 3080Ti replay verified the 256 cohort locks, 12,800 candidate rows, 438 MB Reader weight file, 286 MB checkpoint, two 85-entry restored states, both historical executed-source archives and current method sources. Manifest SHA is `c46eab87...8267b`; identity SHA is `c94001af...69a3e`.
- `scoring_started=false`: Reader, quantum and classical scoring have not run, so no transport, utility, L1/L2 or quantum-advantage conclusion exists.
- GitHub `five-ideas-development` contains the input-freeze implementation through `e424e04`; the successful compact manifest is stored in Git and its source archive remains on18083.

## Plan Control
- Active Plan: plans/reader-larger-untouched-utility/v001.json
- Plan Version: 1
- Plan Status: active
- Completed Steps: LU-01 through LU-04
- Focus Step: LU-05
- Focus Title: Run frozen Reader and head transport
- Next Action: Implement and synthetic-validate the immutable-manifest-driven 256-question Reader transport runner, then live-check Titan capacity before formal scoring.
- Blocker: None.
- Last Checkpoint: 2026-10-19

## Frozen Gate
1. Cohort identity and Top-50 are immutable; no post-score resampling or replacement.
2. Continue to blind review only if quantum changes at least25 cases and differs from matched classical on at least15.
3. For promotion, require quantum useful totals +5 versus Reader and +3 versus classical, no direct-evidence decline versus Reader, case-win margins +3/+1, positive clustered-bootstrap lower bounds, and zero unresolved quantum-selected items.

## Validation
- Receipt: [larger untouched cohort verification](refs/reader_larger_untouched_cohort_20261018_receipt.json).
- Scoring manifest: [immutable transport inputs](../../../configs/experiments/reader_larger_untouched_scoring_manifest.json); [verification receipt](refs/reader_larger_untouched_scoring_manifest_20261019_receipt.json).
- Research log: [256-question cohort freeze](../../../docs/rag-research-log/20261018T-141409Z-reader-larger-untouched-cohort-freeze.md).
- Exchange path: `/srv/qore-collab-uploads/five_ideas/reader_larger_untouched_cohort_256/20261018T141409477054Z` on3080Ti.
- Scoring-manifest exchange path: `/srv/qore-collab-uploads/five_ideas/reader_larger_untouched_scoring_manifest/20261019T003901353299Z` on3080Ti.
- Local regression: 9 tests passed; real manifest replay passed with 3 arms and no scores produced.
- Session: [immutable scoring-input freeze](sessions/2026-10/20261019T004136Z-95f859.md).

## Relevant Decisions
- [Larger untouched utility study contract and checkpoint](decisions/2026-10-18-reader-larger-untouched-utility.md).
- [Blind validation87 utility result](decisions/2026-10-18-reader-unseen-blind-review.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
