# Phase 8A: Doubly-Robust Counterfactual Passage Influence

## Stage Status

Phase 8A-0 is design-only. It freezes the hypothesis, plugin boundary,
randomization, estimator, privacy contract, and decision gate. It does not add a
runner, alter the selector, or authorize a Wiki-DPR execution.

## Question

For the five passages selected by the established QORE plus Answer Scorer
baseline, does randomized passage inclusion have a stable, passage-specific
effect on generated-answer F1 after accounting for the known static signals?

This asks about generator utility directly. It does not optimize similarity,
diversity, agreement, corroboration, or scalar uncertainty.

## Frozen Baseline

- Dataset: `nq_open` validation, questions 200-249.
- Corpus: Wiki-DPR `psgs_w100.nq.compressed`, `nprobe=64`.
- Candidate count: 50.
- Selector: `qore_answer`, K=5, seed 42, baseline enhancer gamma 1.0.
- Generator: pinned `NousResearch/Meta-Llama-3-8B-Instruct` revision
  `53346005fb0ef11d3b6a83b12c895cca40156b6c`, matching the reference Llama3-8B
  family used by the recent mechanism diagnostics.
- Metrics: unchanged `evaluate_answer` EM/F1.

Questions 200-249 are outside the 0-199 slice used by the original 200-question
development gate and Phase 7A-7D diagnostics.

## Probe Design

For each question, `full_context_probe` emits the exact baseline full context.
`balanced_subset_probe` then creates eight deterministic random masks and their
eight complements. Across those sixteen effect-fit probes, every selected passage
is included exactly eight times, so its known inclusion propensity is 0.5.

The runner also records an empty-context anchor. Full and empty anchors are useful
descriptive controls but do not enter the propensity-weighted estimator. Included
passage order is randomized from a separate stable stream so the inclusion effect
is averaged over position rather than being tied to the baseline rank order.

Total: 18 generations per question, 900 generations for 50 questions.

## Estimator

The primary estimator is five-fold, question-grouped AIPW with known propensity
0.5. The nuisance outcome model is ridge regression over non-gold numeric
covariates: Answer Scorer, retrieval rank, original selection position, passage
token count, other-context size, other-context Answer Scorer mass, other-context
token count, and other-context embedding redundancy.

Raw difference-in-means is retained as a model-free randomization estimate. AIPW
must agree with it rather than manufacture a different result. Bootstrap samples
are clustered by question. A complement-preserving within-question placebo tests
whether the reported signal can be reproduced after breaking the real assignment.

Gold answers are used only by the unchanged evaluator to produce EM/F1 outcomes.
They cannot enter mask creation, nuisance covariates, fold assignment, or plugin
configuration.

## Gate

All checks must pass:

1. Split-half influence Spearman is at least 0.20 and its clustered 95% CI lower
   bound is above 0.
2. AIPW and raw difference-in-means influence Spearman is at least 0.50.
3. The treatment-interaction term adds at least 0.05 held-out R2 beyond the frozen
   nuisance model.
4. At least 20% of questions have a within-question passage-effect range of 0.10
   F1 or more.
5. The complement-preserving placebo p-value is at most 0.05.
6. The run does not exceed 18 generations per question and emits compact output
   only.

Pass: authorize a separate Phase 8B plan for a cheap distilled surrogate. Fail:
stop the counterfactual-influence direction. A pass does not authorize a selector,
transfer run, full-data run, or manuscript claim.

## Plugin Boundary

`full_context_probe` is the null baseline transformer. `balanced_subset_probe`
appends diagnostic contexts without mutating the baseline. The existing generator
and evaluator remain stable boundaries. `doubly_robust_influence` consumes only
compact numeric outcomes after evaluation and cannot feed anything back into the
selector or generator.

The plugins are explicitly allowlisted. No filesystem discovery is permitted in a
formal experiment. Answer-identity enhancers, Mobius, cohesion, submodular, and DPP
are not composed into this attribution run.

## Planned Implementation Stage

After explicit approval, Phase 8A-1 will add:

- deterministic probe and estimator modules;
- a one-click model resolver that prefers an explicit override, then the legacy
  project path, then the pinned Hugging Face cache snapshot, and records the
  resolved model identity and config hash in the manifest;
- one configuration validator and smoke-manifest path;
- focused baseline-equivalence, balance, determinism, fold-isolation, placebo,
  privacy, and arithmetic tests;
- one collaborator wrapper that discovers a compatible Python environment and
  writes compact results under `exchange/five_ideas/phase8a_counterfactual_influence`.

Full Wiki-DPR execution remains collaborator-only.

## Cost Estimate

Phase 7D used 144 unique generations in 331 seconds. A purely linear projection for
900 generations is about 34.5 minutes. Allowing for retrieval, scoring, model load,
bootstrap analysis, and context-length variation, the practical first-run estimate
is 40-60 minutes on the RTX 3080 Ti. This estimate must be replaced by measured
timing after implementation smoke validation.
