# Complete-data Reader replication

- Date: 2026-10-19
- Scope: rag-selector
- Status: active design
- Supersedes: None
- Superseded By: None
- Gate: complete population and both independent splits must pass the prospective Reader utility gate
- Authorization: plan and preregistration only; implementation, data materialization, formal scoring, review and claims require their own stage authorization
- Current evidence tier: L0 diagnostic

## Decision

The next formal target is not another sample. It is the exact eligibility
closure of `nq_open/validation[512:3610]`: scan all 3,098 positions, exclude
the frozen question hashes from the protected 434 cases, historical 100 cases
and spent 256-question cohort, and include every remaining unique question.
No eligible question may be sampled away, replaced or removed after scoring.

The exact realized count is intentionally not guessed from incomplete local
data. The data-bearing scan must derive it before any score exists, prove that
it lies between 2,308 and 2,742, and freeze both the full population digest and
ordered Wiki-DPR Top-50 identity digest. This is a stronger and more auditable
definition of “full data” than a hard-coded count inferred without the complete
dataset.

The complete population is sorted by a result-independent SHA-256 key and
alternated into `split_A` and `split_B`. The partitions must be disjoint,
balanced to within one question and have an exact union equal to the complete
population. They are the two independent confirmation units required by the
project's `full_real_data_l2` profile.

## Frozen intervention and review boundary

The DPR retrieval contract, Reader snapshot, sequential quantum checkpoint,
matched classical checkpoint, Top-5 size and tie break remain identical to the
verified 256-question run. The formal population receives no training, tuning,
threshold selection or checkpoint selection.

Questions and retrieval candidates may be exported before scoring; answers,
gold/Silver/support labels, generations and prior semantic outcomes may not be
read. After all three arms are scored and replayed, each split receives its own
method-blind packet containing every passage in the three-arm Top-5 union for
every changed question. Two isolated judgments and blind adjudication resolve
all codes. Both split consensuses must be complete and hash-locked before
either method key is opened.

## Prospective ordinary-utility L2 gate

For the complete population, `split_A` and `split_B` separately, quantum is
compared with the original Reader. Every scope must satisfy all of these:

1. mean useful-passage paired delta is positive and its question-cluster
   bootstrap 95% lower bound is greater than zero;
2. mean semantic-points (`useful + direct`) paired delta is positive and its
   bootstrap 95% lower bound is greater than zero;
3. aggregate direct-evidence delta is nonnegative;
4. useful case wins exceed losses;
5. no review item is unresolved and no quantum-selected item is uncertain;
6. identity, split, leakage, source and numeric replay audits all pass.

Passing all Reader gates permits `L2_reproducible_utility` under the selected
project profile. Failure in the complete population or either split closes this
frozen head; the result may not be rescued by changing the cohort, split,
threshold, checkpoint, metric or review rule.

## Separate quantum-attribution decision

The same useful, points, direct and case-win tests are also run against the
matched classical head in all three scopes. These comparisons do not redefine
ordinary-method L2. A Reader pass with a classical-control failure may still
be L2 ordinary utility, but it blocks quantum-specific attribution and the L4
path. A pass against both controls only authorizes a separately preregistered
finite-shot implementation and resource audit; it is not itself L4.

## Execution topology

- The collaborator's existing complete NQ/Wiki-DPR environment materializes
  the exhaustive cohort with a project-relative one-command wrapper.
- 3080Ti stores 18083 artifacts, verifies hashes, prepares sealed packets and
  aggregates; it is not assumed to provide GPU compute.
- A live, capacity-checked host bound to `gpu-training` performs the single
  immutable Reader/quantum/classical score pass.
- Raw questions, passages, scores and reviews stay on authenticated 18083;
  GitHub receives only compact summaries, manifests, hashes and provenance.

## Frozen artifacts

- Contract: `configs/experiments/reader_full_data_replication.json`
- Plugin plan: `configs/experiments/reader_full_data_replication_plan.json`
- Workflow plan: `../plans/reader-full-data-replication/v001.json`
- Precursor evidence card:
  `../refs/reader_larger_untouched_blind_review_20261019_evidence_card.json`
- Mechanism dossier:
  `../refs/reader_larger_untouched_blind_review_20261019_mechanism_recovery.md`

The canonical plugin-plan validator accepts all four declared plugins. No
formal data, model or review execution occurred during this design step.
