# Larger untouched Reader utility study

- Date: 2026-10-18
- Status: ACTIVE / COHORT EXPORT IMPLEMENTED; REAL COHORT PENDING
- Evidence tier: L0 design and engineering only

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
