# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2. Training-only and label-blind diagnostics remain L0.

## Current State
- [BLOCKED / L0 BLIND UTILITY AUDIT] The frozen validation87 transport run completed and passed both review-effort gates: quantum changed13/87 Top5 sets, classical9/87, and quantum differed from classical on9/87. Independent replay verified261 score vectors and87 Top5 sets with answers/support labels unopened.
- This is transport evidence only. It does not show that any changed selection is better and does not support L1/L2 or quantum advantage.
- The next-stage blind packet is frozen:15 changed-union cases,91 passages, no answers/scores/ranks/method identities. Packet construction, consensus merger, post-consensus scorer and3 synthetic tests pass.
- Reviewer materials and sealed method key are separated on the3080Ti aggregation host. Two independent reviews are pending; the key must remain unopened until semantic-code consensus.
- Server roles are corrected:3080Ti owns orchestration/18083/aggregation; `gpu-training` currently resolves the live-checked Titan host. The recovered3060 cache and original sequential archive passed exact hashes.

## Plan Control
- Plan: plans/reader-unseen-blind-review/v001.json
- Plan Version: 1
- Plan Status: blocked
- Focus Step: BR-04
- Focus Status: blocked
- Completed Steps: BR-01, BR-02, BR-03
- Next Action: Give only the reviewer packet to two independent reviewers and collect complete review JSON files.
- Blocker: Two independent semantic reviews are not yet available; opening the sealed method key before consensus is prohibited.
- Last Checkpoint: 2026-10-18

## Next Actions
1. Collect two independent reviews from `reviewer_packet` only.
2. Merge semantic codes while blind; adjudicate every disagreement without opening the method key.
3. After complete consensus only, open the key and apply the frozen utility gate. Pass advances to a larger untouched study; failure closes this fitted candidate without validation retuning.

## Blockers
- No qualified validation87 semantic judgments exist yet.
- Independent reviewers must not receive `sealed_method_key.json`.

## Validation
- Transport run `20261018T073840422368Z`; trace SHA `c0de703d...`, values SHA `b282090b...`; independent audit passed.
- Validation cache hashes: index `f0052eee...`, features `1837d883...`; sequential archive `35d7f9c1...`.
- Blind packet SHA `eb5aef15...`;15 cases/91 items; banned-field audit passed. Three blind-workflow tests pass in `lambeq`; both workflow plans validate.
- Aggregated artifacts: `/home/Q-DUET-VLM/reader-membership-screen-20261016/exchange/five_ideas/{reader_unseen_transport_gate_87,reader_unseen_blind_review_87}/20261018T073840422368Z` on3080Ti.
- Session: [unseen transport recovery and blind packet](sessions/2026-10/20261018T075005Z-095e98.md).

## Relevant Decisions
- [Two-reviewer blind utility audit](decisions/2026-10-18-reader-unseen-blind-review.md).
- [Completed untouched validation transport gate](decisions/2026-10-18-reader-unseen-transport-gate.md).
- [Completed sequential contract/result](decisions/2026-10-18-reader-sequential-protected-updates.md).
- [Frozen membership contract](decisions/2026-10-16-reader-membership-screen-handoff.md); [final train targets](decisions/2026-10-11-final-support-boundary-resolution.md).
- [Role-separated server execution](../../shared/decisions/2026-10-18-role-separated-server-execution.md); full long-term principles/navigation: `critical-invariants.md`.
