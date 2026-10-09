# RAG Passage Selection with QORE

## Goal
- Improve fixed Top50 -> Top5 on quantum mainline, retaining the frozen Reader anchor; independently verified full-data utility before L1/L2.

## Current State
- Versioned support supervision axes and full434 export implemented. Question support, reference coverage, inference type, source grounding and six scope dimensions are separate. Old1600 opinions/flags/overlays remain unchanged.
- Full434 synthetic fixture emits21700 passage records, Reader scores/ranks/Top5, human Markdown, blind pending review, prospective347/87 split and non-consumable target draft. Local34 and canonical-server34 tests pass; server fixture/CLI readback pass. Real434 export has not run.
- Every review field starts pending; qualified real pairs remain0, training_authorized=false and training_consumable=false. Partial/disputed/external-context/uncertain items stay held_not_negative. Qualified selected direct evidence is protected; only qualified missed-direct versus selected-irrelevant pairs can enter a draft.
- Export wrapper uses collaborator-local cached retrieval/Reader, optionally downloads only pinned manifest/cohort from18083, and automatically creates timestamp upload directory. No model/corpus/checkpoint download, index construction, learned-head replay, Generator or evaluator.
- Prior round2 review closed at9/54 with47 label matches/7 disputes, all scopes conditional and challenge exposure; not new independent truth. Receipt files remain local history; full434 supervision and historical432 text parity remain unresolved.

## Current Decision
- `decisions/2026-10-09-support-supervision-axes-export.md`: implementation/export only; no fitting permission or relaxation of independent qualification.
- `decisions/2026-10-09-round2-scope-adjudication-close.md`: separate support/reference/inference/scope; do not reopen review merely for agreement.
- Constraints/topology and full history index: `critical-invariants.md`.

## Next Actions
1. Collaborator runs `bash scripts/collab/five_ideas/run_support_supervision_cohort_export.sh` with existing QORE_EXCHANGE_TOKEN; pushes compact outputs toGitHub. Raw outputs auto-upload to18083 under five_ideas/support_supervision_cohort_export/<UTC timestamp>/.
2. Check exact434/21700 coverage, source/order/hash, Reader identity, split and upload receipts. Then qualify a defensible subset with actual independent witnesses and explicit scope, not automatic weak/Silver label conversion.
3. Only after target/split qualification, design separately authorized protected quantum semantic-head pilot with matched classical controls, bounded reachability and candidate-specific correction checks.100Silver remains evaluation-only; initialize fresh heads for future347/87 protocol.

## Blockers
- Real434 evidence export pending; source cache/index/model mismatch stops rather than substitutes.
- Real qualified training pairs0; conditional scope, reference alternatives and single-passage grounding not certified. Witness schema passing is not truth/independence proof.
- Newly captured full text cannot establish historical432 text parity. Old heads saw both prospective roles, so no old-head independent-validation claim.

## Validation
- Local34 and server34 synthetic tests pass, including identity leakage, stale witness/binding, reference mismatch, partial protection, exact Reader tokenizer batch API, blind fields and compact privacy.
- Server full434 preflight and pure-stdlib axes compiler reopen outputs:21700 records, zero draft pairs, training false; Bash syntax passes. No actual dataset/model run on our servers.

## Pointers
- Code: `applications/rag/support_supervision_axes.py`; runner/config: `run_support_supervision_cohort_export.py` / `support_supervision_cohort_export.json`.
- Handoff: `docs/support-supervision-cohort-export.md`.
- Owner log: `docs/rag-research-log/20261009T130633Z-support-supervision-axes-export.md`.
- Exact engineering verification: local Temp/qore-support-cohort-export-20261009/verification.json; server /tmp/qore-support-cohort-export-20261009/verification.json. Synthetic engineering evidence, not experiment results.
