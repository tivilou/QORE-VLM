# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2. Training-only and small blind diagnostics remain L0.

## Current State
- [ACTIVE / L0 TRANSPORT PASSED; BLIND UTILITY PENDING] One immutable 256-question x Top-50 score pass ran on the bound Titan GPU host with the frozen DPR Reader and sequential quantum/classical heads.
- Quantum changed the Reader Top-5 set on `61/256` questions and differed from matched classical on `47/256`, passing the prospective `25/15` transport gate. Classical changed `29/256`; the changed-set union is67 questions and404 unique within-question passages.
- Independent replay recomputed all768 Reader/quantum/classical score vectors and Top-5 sets with zero numerical difference. Answers, support/Silver labels and Generator remained unopened, so there is no semantic utility, L1/L2 or quantum-advantage conclusion yet.
- GitHub `five-ideas-development` contains the runner and source-archive repair through `a3f4169`; complete run artifacts are hash-verified on3080Ti/18083.

## Plan Control
- Active Plan: plans/reader-larger-untouched-utility/v001.json
- Plan Version: 1
- Plan Status: active
- Completed Steps: LU-01 through LU-05
- Focus Step: LU-06
- Focus Title: Run blinded semantic utility review
- Next Action: Construct the 67-question/404-passage method-blind changed-set packet, then obtain two isolated reviews and blind adjudication before unblinding utility.
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
- Transport receipt: [256-question label-blind run](refs/reader_larger_untouched_transport_20261019_receipt.json); full artifacts: `/srv/qore-collab-uploads/five_ideas/reader_larger_untouched_transport_256/20261019T014046Z` on3080Ti.
- Research log: [transport result and next gate](../../../docs/rag-research-log/20261019T-014046Z-reader-larger-untouched-transport.md).
- Regression: 9 local tests passed; plugin plan validated; real input replay, complete trace validation and768-vector independent replay passed.
- Session: [LU-05 frozen Reader transport](sessions/2026-10/20261019T015518Z-3cba09.md).

## Relevant Decisions
- [Larger untouched utility study contract and checkpoint](decisions/2026-10-18-reader-larger-untouched-utility.md).
- [Blind validation87 utility result](decisions/2026-10-18-reader-unseen-blind-review.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
