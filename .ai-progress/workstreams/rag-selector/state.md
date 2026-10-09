# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50->Top5 on the quantum main line, anchored to frozen Reader; independent full-data utility before L1/L2.

## Current State
- All32 audit cases now50/50:1600/1600 reviewed,0 remaining. Added817 explicit primary-model judgments;prior783 unchanged;raw source/producer-null labels/references unchanged.
- Full audit labels:direct97,partial223,irrelevant1259,contradictory3,uncertain18. Provisional single-primary-model review,not officialgold or independentlyblind consensus;claim ceilingL0.
- Existing9 question-target flags retained. Among23 provisionally unflagged cases,13 have direct-slot gaps(total25),3 have no direct evidence(11/22/25),7 already retain their capped direct ceiling. Not formal training qualification.
- Reader selectsdirect50/partial54 of160 slots;Qsemantic/Csemantic/monotonic same;Qscalar direct50/partial53. New reviews don't change old method selections or prove utility.
- Weak containment cross-table shows32 direct weaknegative and58 irrelevant weakpositive in this selected32-case audit. Entire434 supervision remains uncertified.

## Current Decision
- `decisions/2026-10-09-support-qualified-boundary-review.md`: complete audit,scope qualification,partial/missing-aware supervision and positive retention.
- Source/sampling: `decisions/2026-10-09-semantic-supervision-result.md`, `decisions/2026-10-09-train-only-supervision-audit.md`.
- Stable topology/principles/history: `critical-invariants.md`.

## Next Actions
1. Qualify target scope and annotation confidence;design boundary-target compiler plus synthetic tests. Prioritize9 provisional cases with missed direct and selected irrelevant(5/6/8/13/16/17/21/26/27);keep4 direct-vs-partial-only gap cases separate(3/9/19/29). No training yet.
2. Protect already-correct Reader selections;mask flagged/missing/uncertain;partial is graded/abstained,not automaticnegative;never force direct-vs-direct preferences or5 golds.
3. Qualify full434 corpus supervision and preregister fresh347/87 before training. Existing heads saw old partitions;exposed100Silver remains evaluation-only;matched classical/scalar/compression controls required.

## Blockers
- Whole434 supervision and formal target qualification incomplete;9 knownquestion flags,implicit/time/geographic/metric scope and single-reviewer consistency need qualification.
- Historical432 raw-text parity/baseline qualification incomplete;independent scalar1e-9 cross-environment accommodation remains disclosed,production unchanged.

## Validation
-1600 unique original IDs;32 complete cases;817 explicit additions;prior783 metadata/9 flags retained;exact quotes/source hashes validated. Local andcanonical replay/tamper24/24;all160 slots per method unchanged.
- Hash-linked783 prior+817 delta(each<1MiB) reconstruct full review;byte-preserving attributes and explicit UTF-8/LF serialization verified. No models/corpus downloaded,Reader/Generator inference or new training.

## Pointers
- `refs/semantic_supervision_20261009T045728Z_complete_review_manifest.json`: hash-linked full1600 review entry.
- `refs/semantic_supervision_20261009T045728Z_complete_{annotations.json,review_delta.json,join.json,diagnostics.json,validate.py,analyze.py,tests.py,verification.md}`.
- Ownerlog: `docs/rag-research-log/20261009T-date-only-complete-support-audit.md`.
- Session: `sessions/2026-10/20261009T080222Z-f8abdc.md`.
