# Complete-data Reader replication

- Date: 2026-10-19
- Scope: rag-selector
- Status: active; FD-03 real population independently verified
- Supersedes: None
- Superseded By: None
- Gate: complete population and both independent splits must pass the prospective Reader utility gate
- Authorization: FD-03 real data materialization is authorized; formal scoring, review and claims remain separately unauthorized
- Current evidence tier: L0 diagnostic

## Decision

The next formal target is the exact eligibility closure of `nq_open/validation[512:3610]`: scan all 3,098 positions, exclude frozen question hashes from protected434, historical100 and spent256, and include every remaining unique question. No eligible question may be sampled, replaced or removed after scoring.

The data-bearing scan derives the count before any score exists, proves it lies in `2308..2742`, and freezes full-population and ordered Wiki-DPR Top-50 identity digests. This defines “full data” without guessing from incomplete local data.

The population is sorted by a result-independent SHA-256 key and alternated into disjoint `split_A`/`split_B`, balanced within one question with exact union coverage. These are the two confirmation units required by `full_real_data_l2`.

## Frozen intervention and review boundary

The DPR retrieval contract, Reader snapshot, sequential quantum checkpoint, matched classical checkpoint, Top-5 size and tie break remain identical to the verified 256-question run. The formal population receives no training, tuning, threshold or checkpoint selection.

Questions and retrieval candidates may be exported before scoring; answers, gold/Silver/support labels, generations and prior semantic outcomes may not be read. After all arms are scored/replayed, each split receives a method-blind three-arm Top-5-union packet for every changed question. Two isolated judgments and blind adjudication resolve all codes; both consensuses are complete and hash-locked before either method key opens.

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

Passing all Reader gates permits project-profile `L2_reproducible_utility`. Failure in the complete population or either split closes this frozen head; cohort, split, threshold, checkpoint, metric or review rule cannot rescue it.

## Separate quantum-attribution decision

The same tests run against the matched classical head in all scopes but do not redefine ordinary L2. A Reader pass with classical-control failure can still be ordinary L2 but blocks quantum attribution/L4. Passing both controls only authorizes separately preregistered finite-shot and resource audits; it is not L4.

## Execution topology

- The collaborator's complete NQ/Wiki-DPR environment materializes the cohort with a project-relative one-command wrapper.
- 3080Ti stores 18083 artifacts, verifies hashes, prepares sealed packets and aggregates; it is not the assumed GPU host.
- A live, capacity-checked `gpu-training` host performs the single immutable Reader/quantum/classical score pass.
- Raw questions, passages, scores and reviews stay on authenticated 18083; GitHub receives only compact provenance.

## Frozen artifacts

- Contract: `configs/experiments/reader_full_data_replication.json`
- Plugin plan: `configs/experiments/reader_full_data_replication_plan.json`
- Workflow plan: `../plans/reader-full-data-replication/v001.json`
- Precursor evidence card: `../refs/reader_larger_untouched_blind_review_20261019_evidence_card.json`
- Mechanism dossier: `../refs/reader_larger_untouched_blind_review_20261019_mechanism_recovery.md`

The canonical plugin-plan validator accepts all four declared plugins.

## FD-02 implementation checkpoint

The pure contract, exporter/wrapper and validator are implemented. The synthetic fixture accounted for all 3,098 positions, 790 exclusions, 2,308 cases x50 and balanced `1,154/1,154` A/B; eight tests, hash replay and remote `bash -n` passed without real content, score, label or review. See `../refs/reader_full_data_fd02_fixture_20261019_receipt.json`; entry: `scripts/collab/five_ideas/prepare_reader_full_data_replication.sh`.

On 2026-10-19 the user authorized only FD-03 population scan, frozen Top-50
export and upload; Reader/head scoring, review and claims remain unauthorized.
## FD-03 real population checkpoint

Run `20261019T131332698621Z` scanned all 3,098 positions. Protected434 had zero in-interval identities; historical100+spent256 excluded 356 distinct identities, leaving 2,742 questions, 137,100 coordinates and `1,371/1,371` A/B. Read-only 3080Ti replay verified every identity, rank, split, lock sample, source, upload and receipt without copying raw text to GitHub. No target/method output exists; scoring is unstarted. Receipt: `../refs/reader_full_data_materialization_20261019_receipt.json`.
