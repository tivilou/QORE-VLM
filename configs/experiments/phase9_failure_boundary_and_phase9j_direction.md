# RAG Failure Boundary and Phase 9J Direction

## Evidence boundary

The project has a stable QORE + Answer Scorer reference path. The following
observations are measured results, not hypotheses:

| Boundary tested | Result | Consequence |
| --- | --- | --- |
| Passage cohesion, submodular, spectral/DPP, and QUBO variants | Recall/redundancy moved without reliable F1 gain | Do not treat passage geometry as answer utility |
| Answer-identity agreement/corroboration | Raw association fell after nuisance residualization | Do not build another answer-agreement objective |
| Retrieval depth | Top-200 versus Top-50 hit gain below the preregistered gate | Do not spend another selector experiment on depth alone |
| Context compression, order, and full Top-50 interventions | No positive, stable recovery | Do not assume more compact or differently ordered context fixes generation |
| Phase 9H failure localization | Errors mixed across extraction, format, semantic generation, and answerability | No single generation failure class authorizes a production patch |
| Phase 9I candidate coverage screen | Candidate oracle EM uplift +0.2727 on 22 eligible questions, but verifier recovered 16.7% of oracle gain and had 40.9% order agreement | The next test must target candidate choice, not passage selection |

## Current problem boundary

The frozen selected context can contain a correct answer candidate while the
baseline generator emits a wrong answer. This is a conditional candidate-choice
problem. It is not evidence that a new retrieval objective will help.

## Phase 9J hypothesis

For a fixed candidate set, a candidate whose likelihood increases under the
selected context relative to an empty context, and whose span is supported by
the DPR reader, is more likely to be the correct answer than a candidate chosen
by a generic free-form verifier.

The candidate set and the ranker are separate variables. Candidate generation
continues to see only the question and frozen selected context. The ranker sees
only runtime candidate strings, the same question/context, generator logits,
and frozen DPR reader logits. Gold answers, evaluator output, QORE scores, and
retrieval diagnostics are unavailable until all inference is complete.

## Signals

For candidate `a`:

```text
context_lift(a) = mean log P(a | q, C_selected)
                  - mean log P(a | q, C_empty)
reader_support(a) = max exact-span (DPR start_logit + end_logit)
```

The combined ranker uses rank-normalized signals:

```text
combined(a) = rank(context_lift(a)) + rank(reader_support(a))
```

The initial screen reports context-only, reader-only, and combined choices.
No learned weight, threshold, or post-hoc tuning is allowed.

## Claim ceiling and stop rule

Before execution this direction is `L0_diagnostic`. A positive screen cannot
establish utility. Formal promotion requires a predeclared EM uplift of at
least 0.03 with a positive paired-bootstrap lower bound, F1 loss no greater
than 0.01, order-invariance at least 0.90, and score cost no greater than 1.3x
baseline generation. A positive formal result authorizes only disjoint
replication. A failed combined-ranker gate closes candidate arbitration and
returns the project to a retriever/generator-boundary decision.

## Known collision and limitation

Candidate reranking, grounded verification, and answer likelihood scoring are
established neighboring techniques. Phase 9J is initially a causal diagnostic
of the project's failure boundary, not a novelty claim. A positive result would
need a later literature audit and matched-control study before any paper claim.
