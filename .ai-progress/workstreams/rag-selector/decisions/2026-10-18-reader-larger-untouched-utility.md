# Larger untouched Reader utility study

- Date: 2026-10-18
- Scope: rag-selector
- Status: closed
- Supersedes: None
- Superseded By: None
- Gate: all eight frozen LU-06 semantic utility checks pass after zero unresolved blind-review items
- Authorization: complete-population replication design and preregistration only; no formal run, L1/L2 claim, or quantum-advantage claim is authorized
- Evidence tier: L0 diagnostic

## Decision

The next test uses a fresh 256-question cohort drawn deterministically from
`nq_open/validation[512:3610]`. Exact UTF-8 question-text hashes from the
protected 434-question bundle and the repeatedly inspected historical
100-question panel are excluded before sampling. The sample order is fixed by
SHA-256 over the seed, dataset index and question hash. Each selected question
receives exactly 50 candidates from the frozen DPR/Wiki-DPR retrieval contract.

The cohort exporter may read question identity and retrieval data, but it may
not export or use answers, gold/Silver/support annotations, generations,
predictions or selector outcomes. The real `cohort.json` and its identity hash
must be uploaded and independently verified before Reader scoring starts. No
post-score resampling or replacement is allowed.

## Execution boundary

- The collaborator machine with the existing complete NQ/Wiki-DPR runtime
  performs only cohort materialization and authenticated upload.
- 3080Ti performs orchestration, hash verification, 18083 exchange and result
  aggregation; it is not the GPU experiment host.
- After the cohort is frozen, a live capacity-checked `gpu-training` host runs
  the pinned DPR Reader and frozen Reader/quantum/matched-classical heads.
- A semantic review is authorized only if the label-blind transport changes at
  least 25/256 cases and quantum differs from classical on at least 15/256.

## Prospective utility gate

On the changed-set union, two isolated reviewers judge useful and direct
evidence without method identity; disagreements are adjudicated while still
blind. Quantum must satisfy all of the following relative to frozen Reader and
the matched classical head:

1. useful-passage total: at least +5 versus Reader and +3 versus classical;
2. direct-evidence total: no decline versus Reader;
3. case-level useful win margin: at least +3 versus Reader and +1 versus
   classical;
4. question-cluster bootstrap lower confidence bound for both point deltas is
   greater than zero;
5. no quantum-selected passage remains uncertain after adjudication.

These are advancement gates, not an L1/L2 or quantum-advantage claim. A pass
would justify a separate full-data replication design. A failure stops this
frozen head configuration; it must not be rescued by changing the cohort,
thresholds or labels after outcomes are visible.

## Rationale

The earlier validation87 review produced a small favorable result, but only 15
cases changed. A larger identity-disjoint population is needed to test whether
that effect transports. Separating cohort freezing from scoring prevents
outcome-conditioned sampling, while the matched classical arm preserves the
minimum attribution comparison required by the quantum mainline.

## Implementation and validation

- Contract: `configs/experiments/reader_larger_untouched_utility_256.json`
- Plugin plan: `configs/experiments/reader_larger_untouched_utility_256_plan.json`
- Export core: `applications/rag/reader_larger_untouched.py`
- One-command handoff:
  `scripts/collab/five_ideas/prepare_reader_larger_untouched_cohort_256.sh`
- Synthetic tests: 3 passed.
- Plugin plan: 3 plugins validated.
- `--validate-only`: 256 cases declared; model and Wiki-DPR not started.
- Python compilation and remote Bash syntax validation passed.

The plugin plan remains `approved`, not `implemented`, until a real cohort
trace, structural and semantic readiness checks, and same-sample replay exist.

## 2026-10-19 checkpoint

Run `20261018T141409477054Z` produced 256 unique questions and 12,800 candidate rows. Server replay found zero overlap with the 534 historical identities, zero forbidden fields or structural errors, and exact cohort/lock/manifest/receipt/source agreement. Cohort SHA-256 is `49c57f20aa2319794e35264aad7a202ab343ff963667424f1685a3e623455a02`; identity SHA-256 is `25a717857ca9a9ff8892161a941101f8dc797d6a146eb35600a1958aa3925397`. Receipt: `../refs/reader_larger_untouched_cohort_20261018_receipt.json`.

## 2026-10-19 immutable input checkpoint

The pre-scoring manifest binds the cohort/lock, DPR Reader revision `38f47a4986084c53447ba92ab0a83076b58d86a8` and five snapshot files, sequential run `20261018T045707538341Z`, both complete 85-entry restored head states, distinct final theta arrays, and all cohort/checkpoint/method source identities. Real 3080Ti replay passed with `scoring_started=false`.

Manifest: `../../../../configs/experiments/reader_larger_untouched_scoring_manifest.json`; file SHA-256 `c46eab8726c5003dc56d7f8b3494888cf222ed63be4e3202e4e49e6d3df8267b`; identity `c94001af30b1e302db6401c3b74788d0c2041d1d831314c6ab30359525669a3e`; receipt `../refs/reader_larger_untouched_scoring_manifest_20261019_receipt.json`; exchange run `20261019T003901353299Z`.

LU-05 may implement and synthetic-test a manifest-driven runner, live-check Titan, and run one label-blind score pass. Semantic review remains unauthorized unless the 25-case transport and 15-case matched-control separation gates pass.

## 2026-10-19 LU-05 result

The single frozen score pass ran on `gpushare-i1-54996-titan5` from Git `a3f416973141c16c54e906585c70923037334e70`. It changed the quantum Top-5 set relative to Reader on 61/256 questions and separated quantum from the matched classical set on 47/256, passing the prospective 25/15 transport gate; classical changed 29/256 questions relative to Reader.

All 768 Reader, quantum and classical score vectors and every Top-5 were independently replayed with zero numerical difference. Answers, support and Silver labels remained unopened and no Generator was called.

The changed-set union contains 67 questions and 404 unique within-question passages. LU-06 is therefore authorized to construct the method-blind packet and obtain two isolated reviews plus blind adjudication. This result remains L0: the event-rate pass does not establish semantic utility, L1/L2, or quantum advantage.

Receipt: `../refs/reader_larger_untouched_transport_20261019_receipt.json`; full aggregation path: `/srv/qore-collab-uploads/five_ideas/reader_larger_untouched_transport_256/20261019T014046Z`.

## 2026-10-19 LU-06 result

The fixed blind packet contained all404 passages in the three-arm Top-5 union for the67 changed questions. Two isolated reviewers independently covered404/404 items and agreed on352 semantic-code pairs. A third adjudicator resolved all52 disputes while the method key remained sealed; unresolved and uncertain quantum-selected items are both zero.

After unblinding, Reader / quantum / classical retained useful evidence `141 /151 /142` and direct evidence `79 /85 /80` on changed cases. Quantum useful deltas are `+10` versus Reader and `+9` versus classical; useful case win margins are `+10` and `+9`. On the full256-question population, with exact zero deltas for189 unchanged questions, points deltas are `+16/+14` and the frozen10000-resample question-cluster bootstrap95% intervals are `[5,28]` and `[4,25]`. All eight prospective checks passed.

A separate audit implementation replayed reviewer agreement, blind adjudication, all method totals, case comparisons, bootstrap intervals and gate decisions exactly. The candidate advances to a new complete real-data replication design. The256 cohort is spent and must not be retuned or reused as fresh confirmation.

The result remains `L0_diagnostic` under the project `full_real_data_l2` overlay: the complete formal population, two independent confirmations, Generator utility, finite-shot execution and resource audit are not complete. Receipt: `../refs/reader_larger_untouched_blind_review_20261019_receipt.json`; evidence card: `../refs/reader_larger_untouched_blind_review_20261019_evidence_card.json`; owner log: `../../../../docs/rag-research-log/20261019T-084256Z-reader-larger-untouched-blind-utility.md`.
