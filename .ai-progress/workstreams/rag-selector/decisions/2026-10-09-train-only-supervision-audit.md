# Train-only supervision and boundary audit before changing quantum training

- Date: 2026-10-09
- Scope: rag-selector
- Status: active
- Supersedes: None; executes the audit recommendation from the repaired-screen result
- Superseded By: None
- Authorization: user approved implementation and collaborator-local observation export; no new optimizer run, model/corpus download on development servers or production change
- Gate: source identity, complete trace/replay and upload pass; actual support-quality diagnosis awaits independent blinded review
- Session: [implementation](../sessions/2026-10/20261009T010552Z-e2dec2.md).

## Why this step

The repaired semantic head trained successfully but rescued only one evaluation passage/question (`204/139` versus Reader `203/138`, CI touches zero). Favorable correction is also common for wrong low-ranked candidates; larger classical/scalar corrections brought no positive gain. The relevant unresolved cause is support-versus-substring supervision and selective boundary discrimination, not automatic circuit depth, epochs or residual-amplitude escalation. Historical span/diversity replacements hurt the strong Reader anchor. Retain that anchor, the qualified optimizer and quantum semantic interaction; observe the training signal before designing a new loss.

## Frozen audit

- Source: `five_ideas/semantic_reader_training_repair_100/20261008T145652Z`; original summary/trace/NPZ SHA checks are mandatory before extracting inputs.
- Derivative input: `five_ideas/semantic_reader_supervision_audit_input/repair_20261008T145652Z/{input_manifest.json,cohort.json,head_checkpoints.npz}`. Manifest SHA `0464da7792e0fd2a0615c2e5a9058b641ef0d3b3300af6d0b39450b44fd39a81`. About 3.7 MB, containing hashes, weak masks, initial/final small head weights and two original input witnesses, not Reader weights/corpus or evaluation labels/text.
- First 512 cached NQ train questions must contain the same 434 usable identities and be disjoint from the 100 evaluation question hashes. Require exact candidate IDs/order and weak masks for every sampled case. Two original Reader representation witnesses must match at `rtol=atol=1e-4`.
- Hash-based future partition: lowest SHA256(`20261009:future-split:<question hash>`) 20%, rounded up = 87 validation / 347 training. This declaration is only for future freshly initialized training. Previous heads saw all 434; their results here are **not heldout validation**.
- Sample: first two original usable witnesses, then 15 hash-selected future-training and 15 future-validation cases excluding those witnesses; 32 unique questions, all 50 passages each. Sampling is fixed before scores/support judgments.
- Existing frozen heads only: Reader, quantum semantic, same-budget classical semantic, scalar quantum; additional positive monotonic `0.5 * Reader score` control has exactly unchanged Top-5. No optimizer/backward or new learned parameters.
- Pairs: strongest and weakest weak-negative Top-5 versus nearest two and deepest missed weak-positive; nearest-score weak-negative outsider in the same Reader rank band. These are dependent descriptive pairs, not independent trials or certified positive/negative support.

## Observability and review

`case_study.json` and human-readable `case_study.md` contain the question, train reference answers, all 50 full passage texts/IDs/retrieval scores, Reader score/rank, weak-match witnesses, all five method score vectors and Top-5, correction/compression and boundary diagnostics. `audit_values.npz` contains full base/pooled/scalar coordinates and per-head scores/encoded coordinates/observables/gates/residuals for 32x50; the two original witnesses additionally include complete selected hidden/token/mask/pooling-weight values.

Separate `support_review.json` uses deterministic shuffled review IDs and **omits scoring ranks/scores, method identities and weak labels**. All 1600 judgments are initially null. Later review uses `direct/partial/irrelevant/contradictory/uncertain`, with literal source quotes required for supporting/contradicting claims and rationale for every judgment. Merely finding an answer string is not a support judgment. The exporter produces review material; it does not fill or certify these judgments.

Project-owned `sample-trace.v2` structure/readiness enforces fixed samples/stages, full detached coverage, original witnesses, null parity, same-input final-head replay, raw input hashes, deterministic selection and blind-input parity. Synthetic fixture checks are engineering L0 only; no claim of actual support precision, generalization, L1/L2 or quantum advantage.

## Collaborator contract

Run from the project root in the collaborator's existing environment:

```bash
bash scripts/collab/five_ideas/run_semantic_reader_supervision_audit.sh
```

Reuse configured `QORE_EXCHANGE_TOKEN`. The script automatically downloads only the pinned small audit input if absent, opens existing NQ/Wiki-DPR/model caches, refuses to build a new index, creates `five_ideas/semantic_reader_supervision_audit/<UTC-run-id>/` on 18083, and uploads declared files with hash receipts. Missing cache/identity mismatch blocks extraction; required upload failure is nonzero and preserves the pending manifest. Optional `--wiki-dpr-cache` selects an existing nondefault HF dataset cache; no hard-coded collaborator Python/project path.

Raw case/review/value/split files remain exchange-only and are explicitly Git-ignored. Only `summary.json`, `report.md`, `run_metadata.json`, `upload_manifest.json` go to GitHub, each <=1 MiB. `--upload-only <existing local run directory>` resumes a hash-unchanged upload without another Reader/retrieval pass. It does not overwrite server results with changed bytes.

## Next gate

Analyze the returned full cases, first blind to scores, then join true support judgments to containment, rank bands and relative corrections. Report weak-positive false-support and weak-negative missed-support cases with literal text evidence, plus whether quantum corrections discriminate support from matched wrong candidates or merely flatten scores. Distinguish representative hash samples from the two original witnesses. Only then propose and preregister a boundary-discriminative quantum training objective with matched classical/control arms and genuinely fresh train/validation use. Do not automatically run that objective, enlarge residuals or train on the exposed 100-case Silver panel.

## Validation

24 new synthetic tests + 12 prior audit tests = 36/36 locally and on the canonical server; full 32x50 fixture readback/replay, two full hidden/pooling slices, exact null, blinded review and corruption/failure tests pass. Six-plugin plan validator passes. Bash syntax and wrapper validate/preflight pass. Actual 18083 input download, directory creation and nine synthetic upload receipts pass. No real-data model/corpus run or installs on development servers.

- [Pinned preflight](../refs/semantic_reader_supervision_audit_preflight_20261009.json)
- [Commands/rollback](../refs/semantic_reader_supervision_audit_verification_20261009.md)
- [Owner explanation](../../../../docs/rag-research-log/20261009T-date-only-semantic-reader-supervision-audit.md)
