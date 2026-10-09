# RAG Passage Selection with QORE

## Goal
- Improve fixed Top-50 -> Top-5 with quantum main line/Reader anchor; independent full-data utility before L1/L2.

## Current State
- Existing32-case supervision export numerical audit/source identity remains verified; original raw files unchanged at18083 `five_ideas/semantic_reader_supervision_audit/20261009T045728Z/`.
- Existing220 boundary/control/all-method selected scope remains complete. Full Top50 review now complete for original cases1-8:169 new+449 preserved=618/1600;982 still unreviewed. Single primary-model provisional judgments; scores withheld while reading but not independently blind/three-model consensus.
- Cases6/8 add3 provisional direct weak negatives at Reader ranks8/16 and6;all methods miss them. Original weak-boundary audit has0 pairs on each of these2 cases. Geographic answer granularity and dated actor alternatives need explicit qualification, not automatic reference rewriting.
- Case3 contains another direct support at Reader rank9, weak-label false: anatomical ordered list places cannon after knee without literal Below Knee. All five old methods miss it; one provisional direct-slot gap. Case1 already retains its only direct passage; cases2/4 retain unresolved scenario/formula target flags.
- Reader/Q-semantic/C-semantic/monotonic selected direct50,partial54. Scalar direct50,partial53,uncertain7 vs6; no confirmed gain or degradation. Seen training sample, no heldout utility.
- Scoped119 weak-positive candidates include19 irrelevant+2 contradictory;101 weak negatives include9 direct (some target conflicts). Do not infer population noise rates. Registered60 dependent pairs yield5 exploratory clean pairs/3 questions;Q crosses0, favors true over matched irrelevant on1. Compression control favors2 without changing selection.
-9 question-target risks recorded, original references unchanged; other23 are provisionally unflagged, not qualified. Full434 training labels remain uncertified. Current claim ceiling L0 diagnostic.

## Current Decision
- `decisions/2026-10-09-support-qualified-boundary-review.md` defines target qualification and positive-retention requirements.
- Prior result/source: `decisions/2026-10-09-semantic-supervision-result.md`; sampling/export: `decisions/2026-10-09-train-only-supervision-audit.md`.
- Stable scope/topology/history: `critical-invariants.md`.

## Next Actions
1. Continue original cases9-32 full Top50 review (982 passages); keep target flags, raw references and missing labels, no inference rerun.
2. Define confidence/missing/partial-aware boundary targets that protect valid selections; synthetic target tests before collaborator training protocol.
3. Only after qualification, design full434 supervision/fresh347/87 training with Q-semantic and matched classical/scalar/compression controls. Old heads saw both partitions; exposed100 Silver is diagnostic-only.

## Blockers
-982/1600 material and whole434 supervision not adjudicated;9 targets require event/type/version/context resolution. Single-review labels not official gold.
- Historical432 raw-text parity unproven; official passage mapping and formal dataset/baseline qualification remain incomplete.
- Historical strict float-stat equality is cross-environment fragile; independent1e-9 accommodation disclosed, production unchanged.

## Validation
-618 IDs/case links/quotes;prior449 retained without changes;cases1-8 each50/50;220/220 scope and all160 selected slots remain covered. Incremental replay/target/tamper12/12; mechanical join only, no label heuristics or new fitting.
- Source/input/head numerical audit and36 original synthetic tests are unchanged; no dataset/model download,Reader inference or Generator run in this continuation.

## Pointers
- `refs/semantic_supervision_20261009T045728Z_boundary_{scope.json,review.json,join.json,judgments.txt,join.py}`.
- Latest incremental refs: `refs/semantic_supervision_20261009T045728Z_fullcase_05_08_{annotations.json,review.json,join.json,validate.py,tests.py,verification.md}`.
- Latest owner log: `docs/rag-research-log/20261009T-date-only-fullcase-support-05-08.md`; original boundary log/records immutable.
- Original partial review and independent numerical audit are immutable prior refs.
- Session: `sessions/2026-10/20261009T060459Z-baf31a.md`; local readback/tamper6/6 and publication proof: `refs/semantic_supervision_20261009T045728Z_boundary_verification.md`.
- Latest session: `sessions/2026-10/20261009T062601Z-13d3ba.md`.
- Latest session: `sessions/2026-10/20261009T065123Z-049984.md`.
