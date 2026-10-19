# State: RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2, while keeping ordinary utility separate from L4 quantum attribution.

## Current State
- [ACTIVE / FD-03 COMPLETE; FD-04 ACTIVE] The complete real NQ/Wiki-DPR population is frozen and independently verified; no Reader/head scoring has started.
- Run `20261019T131332698621Z` scanned all 3,098 fixed validation positions. Historical100 and spent256 excluded 356 distinct in-pool identities; protected434 overlapped the interval by zero; 2,742 unique eligible questions remain.
- Every question has exactly 50 unique ordered candidates: 137,100 coordinates total. Hash-frozen A/B contain 1,371 questions each.
- Read-only streaming replay on 3080Ti passed question/candidate identity, order, split, target boundary, 2,742 lock samples, four exchange files/receipts and seven executed-source members. Raw text remains exchange-only.

## Plan Control
- Active Plan: plans/reader-full-data-replication/v001.json
- Plan Version: 1
- Plan Status: active
- Completed Steps: FD-01, FD-02, FD-03
- Focus Step: FD-04
- Next Action: Bind the independently verified 2742-case cohort and the existing immutable Reader/head identities into a complete-data scoring manifest; do not start GPU scoring.
- Blocker: None for FD-04. FD-05 scoring remains separately unauthorized and requires a live capacity-checked `gpu-training` host.
- Last Checkpoint: 2026-10-20

## Evidence Boundary
1. FD-03 establishes the exact complete population and retrieval coordinates only; it contains no Reader, quantum or classical scores and is not utility evidence.
2. The exact frozen identities are population `3f67fdce...11d7`, cohort `3809bb8a...d297`, split A `3add755b...86c` and split B `11b33cce...749`.
3. Raw `cohort.json`, `full_population_lock.json` and executed sources stay under authenticated 18083 path `five_ideas/reader_full_data_replication/20261019T131332698621Z`; GitHub stores only compact summary, manifest, receipts and the verification receipt.
4. A future complete/A/B Reader-gate pass can establish ordinary-method L2 under the project profile; matched-classical, finite-shot and resource audits separately govern quantum attribution and L4.

## Validation
- FD-03 receipt: [real population verification](refs/reader_full_data_materialization_20261019_receipt.json).
- Compact artifacts: `exchange/five_ideas/reader_full_data_replication/20261019T131332698621Z/`.
- FD-02 receipt: [synthetic fixture and replay](refs/reader_full_data_fd02_fixture_20261019_receipt.json).
- Contract implementation: `applications/rag/reader_full_data_replication.py`; collaborator runner/wrapper: `scripts/collab/five_ideas/prepare_reader_full_data_replication.py` and `.sh`.
- Sessions: [FD-02 implementation](sessions/2026-10/20261019T111520Z-d45268.md), [FD-03 authorization](sessions/2026-10/20261019T123455Z-062f0e.md).
- Session: [FD-03 independent verification](sessions/2026-10/20261019T234305Z-231fe5.md).

## Relevant Decisions
- [Active complete-data replication contract and checkpoints](decisions/2026-10-19-reader-full-data-replication.md).
- [Completed 256-question precursor](decisions/2026-10-18-reader-larger-untouched-utility.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
