# Semantic Reader repair: active learning, one rescue, screen inconclusive

- Date: 2026-10-09
- Scope: rag-selector
- Status: active
- Supersedes: None; result follow-up to the registered optimizer repair
- Superseded By: None
- Authorization: audit uploaded results, preserve evidence and recommend the next direction; no new training run or implementation of a new mechanism
- Gate: selection and quantum-attribution diagnostic gates failed; preserve the broader family as inconclusive
- Source: `five_ideas/semantic_reader_training_repair_100/20261008T145652Z/`
- Session: [result outcome](../sessions/2026-10/20261009T003323Z-0d8405.md).

## Verified state

GitHub was still at `b2ca094` when checked. The collaborator's complete result is on 18083, not a new GitHub result commit. Six artifacts/manifest receipts match bytes/hashes, producer config and source hashes match `b2ca094`, Reader revision and fixed-input SHA match. All 100 case/candidate identities, scores, metrics and question-bootstrap statistics recompute. The nine producer tensor/selection checks pass independently, including replay of 18 selected pre/post optimizer updates. Previous/current Reader score vectors and usable training identities match exactly; the two saved training inputs are identical. This repair did not change the evaluation cohort or Reader.

434 mixed weak-containment cases, three epochs, 1302 updates per arm. Both semantic heads have nonzero finite data gradients, preserved projection norm, and all 5000 effective score corrections are nonzero. The training-health checks pass. This is no longer the earlier zero-residual/decay-collapse failure, but a healthy numerical path does not establish adequate supervision or capacity.

| Arm | Positive evidence | Direct evidence | All-three positive | Fixed Silver membership |
|---|---:|---:|---:|---:|
| Frozen Reader | 203 | 138 | 171 | 355 |
| Quantum semantic | 204 | 139 | 172 | 356 |
| Matched classical semantic | 203 | 138 | 171 | 354 |
| Quantum scalar | 203 | 139 | 171 | 353 |

Each row selects 500 slots across 100 cases. Quantum semantic versus Reader improves one question, ties 99 and worsens zero on positive/direct counts. Mean delta `+0.01` evidence per question, bootstrap CI `[0,0.03]`; lower bound is not strictly positive. The primary and matched-classical gates both fail. No L1/L2 or quantum advantage is established; exposed-panel L0 only. Do not convert a finite short null into a broad-family kill.

## Every changed semantic case

- Quantum case 82: Reader score-rank 6 / retrieved rank 28, direct positive, replaces score-rank 5 / retrieved rank 11, uncertain nonpositive. Base gap `0.0158432`; corrections `-0.4669268` versus `-0.4968855`, enough for a swap. All positive/direct/unanimous/membership deltas +1. This is the one genuine rescue; outcome-selected, exploratory, not independent confirmation.
- Quantum and classical case 79: score-rank 6 / retrieved rank 3 replaces score-rank 5 / retrieved rank 15; both passages are irrelevant. All evidence metrics unchanged. More ranking movement is not intrinsically useful.
- Classical case 52: two irrelevant passages exchange, fixed-membership delta -1 but positive/direct unchanged. This is a Silver-filler mismatch, not an evidence loss.
- Scalar changes ten cases. Positive rescue at 24 cancels positive harm at 19; direct rescue at 54 remains +1. No aggregate positive benefit. All fourteen method/case changes are in the audit, not only favorable cases.

## Mechanistic evidence versus hypotheses

Observed: quantum corrections have median within-question Pearson correlation `-0.8699` with base scores, classical `-0.8199`, scalar `-0.9044`. In the same 57 rescue-eligible questions, score-rank 6–10 missing positives get a favorable relative correction in `63/75` instances, but nonpositives get it in `192/210`. Rank 11–20 positive/nonpositive counts are `71/78` and `486/492`; rank 21–50 `109/112` and `1597/1598`. Candidate counts are dependent, descriptive and not adjusted for exact score/rank. They show that “low-ranked positives received a favorable correction” alone is not a semantic-discrimination proof: low-ranked wrong passages often receive it too.

Observed: quantum changes only two sets. Its correction range is smaller than the original 5/6 gap in 35 questions; larger range in the other 65 is necessary but not sufficient for any swap. Classical has larger mean correction spread (`0.3882` versus quantum `0.1860`) yet still changes only two sets and adds no positive evidence. Scalar changes ten sets without net positive gain. Blindly enlarging correction is therefore not the first recommendation.

Observed architectural limit: bounded scores imply any pair's maximum correction difference <=`2*0.25*std(base)`, even with gate 1. Of 75 missed positive candidates at ranks 6–10 in rescue-eligible cases, 18 exceed that bound versus the weakest nonpositive Top-5 member; ranks 11–20: 63/78; ranks 21–50: 110/112. These are candidates, not questions/slots, and are a necessary reachability condition, not a realizable simultaneous oracle. The current local residual is structurally unsuited to rescuing most deeply underestimated positives.

Observed: quantum/classical evaluation coordinates satisfy `abs(encoded)>0.99` in `76.285%/62.375%` of entries. Encodings still vary and gradients remain live, so this is partial saturation, not renewed complete collapse. Saturation may blunt fine distinctions; that causal claim is unproven. Both saved training cases' semantic witness ranking losses worsen versus their Reader losses, while the overall mean training loss modestly decreases; two cases cannot establish population overfitting/noise.

## Cumulative next direction and gate

Keep the quantum semantic interaction and stable relevance anchor as positive carryovers. Prior broad span fusion/diversity replacements hurt selection; prior scalar rescues have cancelling gains/losses. Current repair solves numerical collapse but mostly produces score flattening with weak selective benefit. **The next best step is a train-only supervision/selection-boundary audit, then a provisional boundary-discriminative quantum head training objective**, not deeper circuits, automatic larger data/epochs or residual amplification.

Before implementing that provisional objective, inspect deterministic train-only high-score weak negatives versus lower-score weak positives: certify their question/passage relation and distinguish string containment from support, measure correct relative discrimination and saturation on unseen training-validation cases, and retain matched classical plus a simple score-compression control. Train-only supervision cannot borrow the exposed evaluation Silver labels. If reliable support supervision or a selective signal cannot be established, stop this unchanged all-pairs-weak-loss/local-residual configuration's escalation, not RAG/quantum exploration. Treat broad scorer-integrated nonlocal correction as exploratory only until mechanism/readiness qualification exists.

There is no new experiment command in this result-review turn. The next audit/design needs owner confirmation; all formal data experiments remain collaborator-local. The recovery record uses `stop` for automatic next-study selection pending supervision diagnosis, not a project-goal stop.

## Evidence

- `../refs/semantic_reader_repair_20261008T145652Z_audit.json`
- `../refs/semantic_reader_repair_20261009_recovery.json`
- `../refs/semantic_reader_repair_20261009_evidence_card.json`
- `../refs/semantic_reader_repair_result_verification_20261009.md`
- `docs/rag-research-log/20261009T-date-only-semantic-reader-repair-result.md`
