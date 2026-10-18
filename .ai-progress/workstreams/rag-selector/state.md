# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on the quantum mainline toward independently verified full-data L2. Training-only and label-blind diagnostics remain L0.

## Current State
- [ACTIVE / L0 UNSEEN TRANSPORT] The validation87 cohort is genuinely outside the fitted training59 and exposed evaluation100, but it has no qualified support labels: all87 question scopes and4350 passage judgments remain pending. Utility cannot be claimed before blinded review.
- A prospective label-blind gate is frozen: quantum must change at least5/87 Top5 sets and differ from the matched classical arm on at least3/87 before semantic review is worth opening. These thresholds are review-effort gates, not performance gates.
- The isolated transport module, validation feature cache builder, fixed-head runner, independent audit,3-plugin plan and3 synthetic replay tests are implemented. Plugin-plan validation passes.
- Backup `gpushare-i1-57535` completed 87/87 frozen DPR Reader feature extraction on its RTX3060. All stored Reader scores and Top5 sets reproduced; cache hashes: index `f0052eee`, features `1837d883`. It became connection-refused before the validation cache/final result could be recovered.
- Primary3080Ti reconnected and the original sequential archive reverified at SHA `35d7f9c1...`, but the current allocation exposes no GPU and enforces a1GiB cgroup. DPR Reader forward makes SSH unresponsive even at microbatch1; this container cannot safely produce the missing cache.
- Prior completed run `20261018T045707538341Z`: quantum/classical each accumulated0->1->2->3 fitted rescues with zero original/accepted loss; quantum8 attempts, classical5. It establishes training feasibility only and no quantum advantage.

## Plan Control
- Plan: plans/reader-unseen-transport-gate/v001.json
- Plan Version: 1
- Plan Status: blocked
- Focus Step: UG-03
- Focus Status: blocked
- Completed Steps: UG-01, UG-02
- Next Action: Restore a bound server with a visible GPU, or recover the completed validation87 cache from the backup; then execute the fixed label-blind transport runner.
- Blocker: Primary is currently a1GiB CPU-only allocation; backup completed feature parity but remains connection-refused.
- Last Checkpoint: 2026-10-18

## Next Actions
1. Keep the verified original `sequential_values.npz`; recover the validation cache from the backup or regenerate it only on a visible-GPU bound server.
2. Apply the already-frozen quantum/classical final theta once, save all261 score vectors and all87 Top5 sets, and independently replay the transport gate.
3. If and only if both transport thresholds pass, create a score/rank/method-blind semantic review packet; qualified utility remains a separate experiment.

## Blockers
- No valid utility labels exist for validation87 yet; the current gate measures transport only.
- Primary capacity and backup reachability prevent recovery/regeneration of the completed validation feature cache; original frozen parameters are verified and intact.
- Tiny cross-environment Reader differences make approximate reconstruction unacceptable; the original hash-locked parameters must be used.

## Validation
- `applications/rag/tests/test_reader_unseen_transport.py`:3 passed in the existing `lambeq` environment, including full synthetic87 trace/audit replay.
- Research plugin architecture validator:3 plugins validated. Workflow plan validator passes.
- Backup feature extraction:87/87 score and Top5 parity, local cached Reader revision `38f47a49`, no model/data download, answers/support labels unopened.
- Primary retry: original archive hash passed; batch8 and resource-only microbatch1 both exceeded the usable1GiB CPU-only allocation. No transport scores were produced or inspected.
- Current implementation/config: `applications/rag/reader_unseen_transport.py`, `scripts/collab/five_ideas/{prepare,run,audit}_reader_unseen_*`, `configs/experiments/reader_unseen_transport_gate_87*.json`.
- Session: [transport implementation and server handoff](sessions/2026-10/20261018T061040Z-311464.md).
- Session: [primary capacity retry](sessions/2026-10/20261018T064311Z-d28ab2.md).

## Relevant Decisions
- [Untouched validation transport gate](decisions/2026-10-18-reader-unseen-transport-gate.md).
- [Completed sequential contract/result](decisions/2026-10-18-reader-sequential-protected-updates.md).
- [Frozen membership contract](decisions/2026-10-16-reader-membership-screen-handoff.md); [final train targets](decisions/2026-10-11-final-support-boundary-resolution.md).
- Server topology and collaboration boundaries: `../../project.md`; full long-term principles/navigation: `critical-invariants.md`.
