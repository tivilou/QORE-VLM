# Reader feature rotation engineering boundary

- Date: 2026-10-13
- Scope: rag-selector
- Status: active_engineering; pinned_reader_gate_pending
- Supersedes: None
- Authorization: user 开始 authorizes isolated module/control/loss/engineering fixture only; no real cohort fit/download.

## Decision

Implement opt-in classifier-input rotation, never mutate production Reader/selector.
10 data qubits,2layers RY then ring CRY,40 affine4-condition angles/200parameters.
Initialize every angle coefficient at zero. Preserve original classifier and signed
readout. Classical control uses seed+1 independent coordinate matchings with
identical active-pair counts and parameter budget. Fixed projection is row-unit
Gaussian of RMS-normalized pooled representation, compressed z/sqrt(1+z²); it is
not learned or Silver-tuned. No parameter/resource superiority is claimed.

New protection loss compares each immutable candidate to fifth-largest other
score. Training labels enter loss only. Stable unmasked Top5 and actual retained
membership must be audited, including ties. Old objective/head remain unchanged.

## Engineering evidence and gate

Eight local tests and seven initial canonical-server tests pass. Random one-layer
DPR architecture fixture passes448-array integrity,8head/4update replay, shared
complete-v2 validation and JSON schema. This is explicitly random-weight
architecture evidence, not the pinned pretrained checkpoint.

Cached pretrained Reader run was attempted after correcting raw config-file vs
runtime-object hashes. Server became unreachable during SSH banner exchange;
run completion/cause is unknown. Updated bounded batch4 runner is local/GitHub
only until server access returns. Plan remains approved/trace-readiness partial;
do not mark formal architecture readiness implemented or hand off real fitting.
No server restart, failover, installation or dataset download occurred.

## Follow-up

Resolve old process/output state when server returns; synchronize final version;
run offline cached Reader fixture and verify all checks. Only then freeze formal
resource/input budget and crossed old/new loss/head arms before collaborator run.
All prior support-audit dispositions and frozen59/228 inputs remain unchanged.

Evidence: `docs/rag-research-log/20261013T-quantum-reader-rotation-engineering.md`;
plan/config: `configs/experiments/quantum_reader_rotation_preflight{,_plan}.json`;
receipt: `refs/reader_rotation_engineering_20261013_receipt.json`.
