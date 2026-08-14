# Diagnostics Analysis

- Source run: `20260814T041313Z`
- Configurations: qore_as_control, cohesion_fixed, cohesion_calibrated, cohesion_disabled
- Samples per configuration: 200
- Bootstrap: 5000 repetitions, seed 20260809

The analyzer ran locally on the original result.json files. This directory contains no question text, answers, predictions, passage text, or embeddings.

## Files

- `aggregate.csv`: recomputed scalar metrics.
- `paired_effects.csv`: question-level paired deltas and deterministic bootstrap 95% intervals.
- `per_question.csv`: scalar metrics, retrieval/gold flags, and selected ranks for every configuration.
- `qubo_diagnostics.csv`: QUBO-derived quality, redundancy, interaction, residual-complementarity, objective, and optimality statistics.
- `plugin_timing.csv`: passage-free per-question enhancer timing when emitted by the selector.
- `qubo_payload.jsonl.gz`: passage-free `a/b/w/x/pool_ranks` payload for exact offline re-scoring of the QUBO pool.
- `summary.json`: machine-readable validation, provenance, aggregate, and paired results.

## Interpretation guard

The current pilot fixes K=5. It reports the ingredients for a cohesion scale-law analysis, but cannot establish a delta*(K-1) law without a K sweep or held-out evaluation.
The recorded diagnostics energy/terms are retained for comparison; `qubo_diagnostics.csv` also recomputes the actual interaction objective from w, which is the solver matrix for enhancer runs.

## Publish

`git add exchange/five_ideas/development_gate/20260814T041313Z/analysis/ && git commit -m 'results: add compact diagnostics analysis' && git push`
