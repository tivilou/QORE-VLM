# Separate support supervision axes and export the complete registered training cohort

- Status: active
- Date: 2026-10-09
- Owner: rag-selector
- Supersedes: no prior labels or qualification compiler; follows round2 scope adjudication close

## Binding decision

Implement the versioned contract and export only, not a new training run. Preserve
Reader relevance and fixed Top50 identity/order. Keep question support, reference
coverage, inference type, source grounding and six question-scope dimensions
separate. Every annotation starts pending; previous review opinions are immutable.
Reference mismatch is not a binary negative; alternatives, minimal arithmetic and
intra-passage composition must have their own provenance, scope and quotes.

The pure-stdlib compiler emits a non-consumable draft only. Scope and item
qualification require distinct actual reviewers/artifacts, primary and independent
roles, exact annotation binding and no prior-review exposure for the independent
witness. A contract check is not certification of semantics or independence.
Held, partial, contradictory, uncertain, externally grounded and unresolved items
never default to negatives. Qualified selected direct passages are protected;
qualified missed direct versus selected irrelevant pairs get per-question weights
and replacement-slot counts. Existing compiler remains unchanged. No fitting gate
is opened by a schema, synthetic witness or successful export.

## Population and provenance

Use all434 registered usable cases from the prior512 NQ train population, 50
exact prior candidate IDs/weak masks per case. Stop on mismatch; do not resample.
Reader input construction/batching is the prior semantic-forward construction,
with frozen relevance output only. Reopen complete scalar values, score ranks,
Top5, exact text/title hashes and deterministic blinded permutation.

Exclude100Silver by hash, never import its text/answers/labels. Reuse prospective
347/87 split, but old heads saw both roles; fresh heads and separate authorization
are required for future independent protocol. Newly locked434 texts do not prove
missing historical432 text parity. This engineering trace has project-owned schema
rag.support_cohort_export.v1, not generic hidden-state sample-trace.v2 certification.
It contains no hidden tensors, learned head intervention or generated answers.

## Execution and publication

One-command collaborator wrapper: run_support_supervision_cohort_export.sh.
No fixed Python/project path. Cached models/datasets/index only; optional18083
download is pinned input manifest/cohort, not head checkpoints. Full434 export
runs on collaborator computer, not our code server. Automatic timestamp namespace
creation/upload with hash receipts, raw files exchange-only regardless of size,
four allowlisted compact files <=1MiB for GitHub. Interrupted export retains
partial JSONL for investigation (not automatic resume); upload failure reuses
complete outputs with --upload-only and zero new Reader passes.

## Evidence and next gate

Local34/server34 synthetic tests and server434x50 fixture/CLI pass; real434 export
has not run, qualified real pairs0, training_authorized=false. This is L0
engineering, not task utility or quantum advantage. Prior fusion failures,
compression-only corrections and weak-string errors motivate target repair;
stable Reader evidence motivates retaining the anchor. Next minimal action is
collaborator full export, then independently qualify a clean bounded subset.
Kill export on identity/coverage/nonfinite/cache/provenance/receipt failure; stop
before fitting for unresolved supervision or contaminated split. Do not amplify
residuals, deepen a circuit, force review agreement or fit exposed Silver instead.

See [handoff](../../../../docs/support-supervision-cohort-export.md),
[owner log](../../../../docs/rag-research-log/20261009T130633Z-support-supervision-axes-export.md).
