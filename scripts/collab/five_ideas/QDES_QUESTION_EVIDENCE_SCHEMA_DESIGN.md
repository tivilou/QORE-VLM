# Q-DES: Question Evidence Schema Feature Preflight

## Stage Status

Q-DES preflight is diagnostic-only. It tests whether a question-only typed
coverage signal is non-degenerate and separable from generic relevance on the
fixed Top-50 candidates. It does not authorize a selector modification,
Generator change, or Wiki-DPR mutation.

## Scope and Input

- Dataset/corpus and retrieval are frozen by the case-study artifact.
- Input is `research-web/apps/experiment-results/case-studies/
  rag-selector-50-detail.json`; its root field must be `cases`.
- The artifact must contain exactly 50 contiguous cases and 50 unique candidates
  ranked 1..50 in every case.
- Silver `top_50[].evidence.positive_consensus` labels are read only in the
  offline evaluation branch. They are never visible to the typed signal.
- The optional three-model panel file is schema-checked only and never feeds
  selection.

## Probe Design

The deterministic parser extracts operator (`who`, `where`, `when`, `why`,
`how`, `count`, and so on), answer type (`person`, `location`, `date`,
`quantity`, `entity`, and so on), a short subject anchor, and relation tokens.
There is no unconditional `what` fallback: an unparseable question is reported
as a failure.

For each passage, typed coverage is a fixed weighted combination of:

- subject-anchor overlap;
- relation-token overlap;
- answer-type cue coverage;
- operator cue coverage.

A whole-question lexical overlap, Answer Scorer, and retrieval score are reported
as generic controls. No embedding model or model call is required.

## Offline Metrics

The preflight reports passage-level micro and macro AUC for each signal, typed
gain over Answer Scorer and generic overlap, typed-vs-Answer-Scorer ordering
disagreement, non-zero/ non-degenerate coverage, and a diagnostic proxy Top-5
positive rate. The proxy ranking is not a production selector result.

## Decision Gate

All checks must pass before a later selector-modification design can be
considered:

1. Schema and fixed 50 x 50 accounting pass.
2. Parser success is at least 80%, and typed coverage is non-degenerate.
3. Typed micro AUC is at least 0.55 and improves over Answer Scorer by at least
   0.02; at least half of cases show material ordering disagreement.
4. Silver passage labels are present for all 50 cases.
5. The pure computation replays to the same digest and the no-feedback contract
   passes.

If any gate fails, Q-DES is disabled and no selector screen or collaborator
handoff is authorized. A pass is only a preflight signal, not an L1/L2 result.

## Privacy and Scope Contract

- Observation-only: no selector, retriever, Generator, evaluator, or panel call.
- Selection sees only question, candidate id/text/rank/retrieval score/Answer
  Scorer value.
- Gold answers, evidence labels, selector outputs, predictions, and generated
  answers are not consumed by selection and are not persisted in output.
- Output contains hashes, fixed positions, numeric diagnostics, and gates only.
- The runner performs a deterministic replay and records `gpu_used: false` and
  `model_used: false` in metadata.

## Implementation

- `applications/rag/qdes_preflight.py`: canonical schema validation, parser
  integration, typed signal, metrics, leakage audit, and replay digest.
- `scripts/collab/five_ideas/run_q_des_preflight_50.py`: CLI.
- `scripts/collab/five_ideas/run_qdes_preflight.sh`: environment-agnostic wrapper.
- `scripts/collab/five_ideas/question_evidence_schema_preflight.py`: compatibility
  entry point delegating to the canonical runner.

## Expected Artifacts

`exchange/five_ideas/qdes_preflight/<timestamp>/` contains compact
`result.json`, `summary.json`, and `run_metadata.json`. The files contain no raw
questions, passage text, answers, predictions, or provider responses.
