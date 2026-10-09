# RAG Passage Selection with QORE

## Goal
- Improve fixed Top-50 -> Top-5 while retaining the quantum line; achieve independent real-data utility before L1/L2 claims. Retrieval, production selector, Generator and evaluator stay frozen.

## Current State
- Repaired 100-case semantic screen is audited and published at canonical `5d4ad06`. Reader/quantum/classical/scalar positive/direct totals: `203/138`, `204/139`, `203/138`, `203/139`. Quantum is 1 win /99 ties /0 losses; +0.01 evidence/question, CI [0,0.03]; both gates fail. L0/inconclusive; family not closed.
- Training repair works, but corrections largely flatten Reader scores, with weak selective benefit and bounded deep-rank reachability. Support-versus-containment supervision remains unresolved. Detailed result and cumulative lessons are indexed in `critical-invariants.md`.
- User approved the next train-only audit. Isolated exporter/wrapper/config/plugin plan and hash-locked previous-run inputs are implemented. No new training objective or real-data inference has run on our server.
- Fixed32 sample: two original training witnesses +15 hashed prospective-training +15 prospective-validation cases from the same434 cohort; all50 candidates. Replays Reader, old quantum/classical semantic, scalar quantum and monotonic compression control. Saves full cases/values and separate blinded pending support review.
- Prospective347/87 split is for future fresh training only. Old heads saw both partitions; this audit is not independent validation. No exposed100-case Silver/raw evaluation input is used.
- Input bundle is ready on18083 under `five_ideas/semantic_reader_supervision_audit_input/repair_20261008T145652Z/`; only small old-head checkpoints/hashes/two witnesses, about3.7MB. Models/data/index are local-cache-only; missing cache stops, no automatic download/index creation.

## Current Decision
- Current implementation/gates: `decisions/2026-10-09-train-only-supervision-audit.md`.
- Result diagnosis: `decisions/2026-10-09-semantic-reader-repair-result.md`.
- Scope/topology/quantum principles and complete history: `critical-invariants.md` and indexed decisions. No cross-workstream context read without a relevant boundary request.

## Next Actions
1. Collaborator updates `five-ideas-development`, uses existing exchange token and runs `bash scripts/collab/five_ideas/run_semantic_reader_supervision_audit.sh` from their project root.
2. Script creates timestamped18083 directory and uploads declared raw cases/values/review, verifying receipts. Mirror only summary/report/run_metadata/upload_manifest <=1MiB to GitHub. `--upload-only <local run dir>` retries unchanged pending uploads without new inference.
3. Independently review the returned blind question/passage view first; then join support-versus-containment, score bands, matched nonpositive corrections and source trace. Distinguish witnesses from hash sample; do not call weak-positive counts support accuracy.
4. Only after diagnosis, propose/preregister boundary-discriminative quantum training with matched classical and truly fresh validation. No automatic new training, residual amplification, more epochs/data/circuit depth or panel-label tuning.

## Blockers
- Await collaborator's32-case raw export and independent support review. Support-quality/utility remains unknown.
- Original raw text hashes for432 train inputs were absent; exact IDs/order/masks and two original numerical witnesses are checked, but full historical text parity remains unproven.
- Official passage-gold alignment and previous Q-ARCG/span-fusion compact provenance gaps remain separate; current audit does not reconstruct them. See indexed decisions.

## Validation
- Local lambeq/server py310:36/36 synthetic tests (24 new+12 prior audit), no installs. Full32x50 detached fixture reopens and passes11 project-v2 structure/semantic/replay/null/blind checks; six-plugin plan validates; Bash syntax/validate/preflight pass.
- 18083 is healthy. Actual authenticated pinned-input download (3 files), directory creation and9 synthetic upload receipts pass. No real-data/model fitting/download or new training on development servers.
- Publication baseline canonical5d4ad06/local3aa97d0; only owned files are staged; unrelated changes preserved. Exact deployment and rollback commands live in the verification record.

## Pointers
- Wrapper/runner: `scripts/collab/five_ideas/run_semantic_reader_supervision_audit.sh` and `.py`; observation module: `applications/rag/semantic_supervision_audit.py`.
- Frozen config/plan: `configs/experiments/semantic_reader_supervision_audit.json` and `_plan.json`.
- Preflight: `refs/semantic_reader_supervision_audit_preflight_20261009.json`; verification: `refs/semantic_reader_supervision_audit_verification_20261009.md`.
- Owner log: `docs/rag-research-log/20261009T-date-only-semantic-reader-supervision-audit.md`.
- Previous result evidence: `refs/semantic_reader_repair_20261008T145652Z_audit.json`, recovery/card and verification indexed by result decision; original trained modules/configs remain unchanged.
- Session: `sessions/2026-10/20261009T010552Z-e2dec2.md`.
