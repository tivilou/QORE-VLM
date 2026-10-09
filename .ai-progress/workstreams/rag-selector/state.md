# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50->Top5 with quantum main line/Reader anchor; independent full-data utility before L1/L2.

## Current State
- Existing32-case training-supervision export source/numerical audit verified; raw18083 files unchanged at `five_ideas/semantic_reader_supervision_audit/20261009T045728Z/`.
- Cases1-12 now each50/50 reviewed;165 new+618 unchanged=783/1600,817 remaining. Single primary-model provisional judgments; scores withheld while reading, not independently blind, three-model consensus or officialgold.
- Case9 has6 provisional direct, Reader selects3 plus2 partial;all5 methods miss ranks6/21/22, last2 weakfalse. Weak-boundary audit0 pairs despite direct-slot gap2. Direct-vs-partial comparison needs graded qualification, not partial-as-negative.
- Case10 retains first-migration/reference-order conflict;case11 full50 contain0 direct/1 partial/49 irrelevant for Finals single-game3-point record;case12 Reader retains its only direct. Do not manufacture5 golds or rescue targets where direct evidence is absent.
- Original all-method selected union/boundary/control scope220 complete;32x5 selected counts unchanged:Reader/Qsemantic/Csemantic/monotonic direct50/partial54,scalar direct50/partial53/uncertain7 vs6. No heldout utility or confirmed new gain.
- Existing9 target flags unchanged;other23 remain provisionally unflagged, not qualified. Entire434 training supervision uncertified;claim ceilingL0.

## Current Decision
- `decisions/2026-10-09-support-qualified-boundary-review.md`: target qualification, positive retention and incremental evidence.
- Source/sampling: `decisions/2026-10-09-semantic-supervision-result.md`, `decisions/2026-10-09-train-only-supervision-audit.md`.
- Stable topology, principles and cumulativehistory: `critical-invariants.md`.

## Next Actions
1. Continue original cases13-32 fullTop50 review(817 passages);keep raw references,target flags and missing labels;no inference rerun.
2. After coverage/qualification, define confidence/missing/partial-aware boundary targets and positive retention;synthetic target tests before training protocol.
3. Only after qualification, design full434 supervision/fresh347/87 training with quantum semantic,matched classical/scalar/compression controls. Oldheads saw both partitions;exposed100 Silver evaluation-only.

## Blockers
-817/1600 material and whole434 supervision not adjudicated;9 targets need condition,event,type,version/formula resolution. Single-model bias and geography/time granularity persist.
- Historical432 raw-text parity and formal dataset/baseline qualification incomplete.
- Strict scalarfloat equality cross-environment fragile;independent1e-9 accommodation disclosed,production unchanged.

## Validation
-783 unique IDs,source hashes/quotes,prior618 intact;12 completecases;all160 selectedslots covered. Incremental replay/target/tamper14/14;mechanical join only.
- Existing numerical audit and36 original synthetic tests unchanged. No model/data download,Reader/Generator inference or training.

## Pointers
- Latest refs: `refs/semantic_supervision_20261009T045728Z_fullcase_09_12_{annotations.json,review.json,join.json,validate.py,tests.py,verification.md}`.
- Immutable priorfullcase/boundary/partial records retained;scope `refs/semantic_supervision_20261009T045728Z_boundary_scope.json`.
- Ownerlog: `docs/rag-research-log/20261009T-date-only-fullcase-support-09-12.md`.
- Session: `sessions/2026-10/20261009T070715Z-4618f3.md`.
