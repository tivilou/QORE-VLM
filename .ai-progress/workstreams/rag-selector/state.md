# State: RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2, while keeping ordinary utility separate from L4 quantum attribution.

## Current State
- [ACTIVE / FD-02 COMPLETE; FD-03 AWAITING AUTHORIZATION] The complete-population freezer/exporter, exact artifact validator and collaborator one-command wrapper are implemented.
- A full-scale synthetic fixture scanned all 3,098 fixed source positions, excluded exactly 790 synthetic exposed identities, froze 2,308 eligible cases x50 candidates and replayed balanced `1,154/1,154` A/B partitions.
- Eight focused/regression tests, target-boundary validation, source/manifest replay and remote Bash syntax validation passed. No real NQ question, Wiki-DPR passage, Reader score, label or review was opened.
- The frozen sequential quantum Reader head still has only the prior fresh 256-question L0 semantic result; complete-data utility and quantum attribution remain unmeasured.

## Plan Control
- Active Plan: plans/reader-full-data-replication/v001.json
- Plan Version: 1
- Plan Status: active
- Completed Steps: FD-01, FD-02
- Focus Step: FD-03
- Next Action: After explicit FD-03 authorization, have the collaborator run the one-command complete-population materialization; do not start real data access before that authorization.
- Blocker: FD-03 requires explicit authorization and collaborator access to the complete NQ/Wiki-DPR runtime.
- Last Checkpoint: 2026-10-19

## Evidence Boundary
1. FD-02 establishes implementation/replay integrity only; it is not a real-data experiment or method-utility result.
2. Real “full data” remains every eligible identity in `nq_open/validation[512:3610]` after exact protected434, historical100 and spent256 exclusions, with no sampling or replacement.
3. The collaborator must derive and hash-lock the real eligible count and Wiki-DPR Top50 before any score exists; labels, answers, generations and review outcomes remain prohibited.
4. A future Reader-gate pass can establish ordinary-method L2 under the project profile; matched-classical, finite-shot and resource audits separately govern quantum attribution and L4.

## Validation
- FD-02 receipt: [synthetic fixture and replay](refs/reader_full_data_fd02_fixture_20261019_receipt.json).
- Contract implementation: `applications/rag/reader_full_data_replication.py`.
- Collaborator runner/wrapper: `scripts/collab/five_ideas/prepare_reader_full_data_replication.py` and `.sh`.
- Tests: `applications/rag/tests/test_reader_full_data_replication.py`; focused/regression run reports `8 passed`.
- Experiment config/plan: `configs/experiments/reader_full_data_replication.json` and `reader_full_data_replication_plan.json`; four plugins validate.
- No dataset/model/retrieval was loaded in `--validate-only`; no real-data or formal scoring execution occurred.
- Session: [FD-02 implementation checkpoint](sessions/2026-10/20261019T111520Z-d45268.md).

## Relevant Decisions
- [Active complete-data replication contract and FD-02 checkpoint](decisions/2026-10-19-reader-full-data-replication.md).
- [Completed 256-question precursor](decisions/2026-10-18-reader-larger-untouched-utility.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
