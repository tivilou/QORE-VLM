# Reader feature rotation engineering boundary

- Date: 2026-10-13
- Scope: rag-selector
- Status: active_engineering; pinned_reader_gate_passed
- Supersedes: None
- Superseded By: None
- Gate: pinned pretrained Reader fixture passes structure, full values, null/span parity and optimizer/Top5 replay; real utility still pending
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

Thirteen local tests pass, including full per-module streaming runner readback.
Fresh random one-layer DPR fixture `20261013T055251674685Z` passes448-array
integrity,8head/4update replay, shared complete-v2 and JSON schema. This is
explicitly random-weight architecture evidence, not the pinned checkpoint.

The original server's current runtime exposes no CUDA and has a1GiB cgroup
quota despite host-level RAM reports. Its bounded run reached Reader forward but
was killed by the kernel cgroup (`memory.events: oom=1, oom_kill=1`) before the
trace was complete. This is an environment failure, not model evidence.

On the separately bound `gpushare-i1-57535` development server, with RTX3060,
the immutable cached checkpoint and the same code completed the full fixture.
The remote validation has 448 arrays,8 head replays,4 optimizer updates,
complete numeric/Top5 replay, span parity, disabled restoration and frozen
Reader checks. The trace was reopened locally: shared `sample-trace.v2` and
JSON Schema validators pass (one expected evaluation-training not-applicable
warning). Remote values SHA256 is `85557961ce169217ba92bfb4b99be569bf6dd84656dff352f3cba14585f9ee91`.

For constrained CPU, materialize FP32 mapped parameters only during the actual
frozen DPR module forward and release encoder parameters back to meta. Keep
classifier/span heads resident. Known deterministic BERT buffers are restored;
unknown buffers, missing active parameters, shape/precision drift fail closed.
Only the two known unused pooler tensors may be ignored, and only when this
Reader architecture has no pooler. The original checkpoint remains immutable;
conversion changes serialization, not precision or model math. Keep source and
converted whole-file/per-tensor hashes in the manifest. Native streaming does
not require accelerate; ordinary low-memory loading uses it only if installed.

Final streaming loader is published. The pinned Reader engineering gate is now
passed for this synthetic fixture; this does not authorize a real cohort fit or
claim task utility. The separate 1GiB server remains unsuitable for this run.
No restart, failover or dataset download occurred; the gpushare environment did
receive the already-cached model and required Python packages.

## Follow-up

Keep the verified gpushare fixture receipt and freeze formal
resource/input budget and crossed old/new loss/head arms before collaborator run.
All prior support-audit dispositions and frozen59/228 inputs remain unchanged.

Evidence: `docs/rag-research-log/20261013T-quantum-reader-rotation-engineering.md`;
plan/config: `configs/experiments/quantum_reader_rotation_preflight{,_plan}.json`;
receipt: `refs/reader_rotation_engineering_20261013_receipt.json`.

Low-memory follow-up: `docs/rag-research-log/20261013T-reader-rotation-low-memory-preflight.md`;
receipt: `refs/reader_rotation_low_memory_20261013_receipt.json`.
