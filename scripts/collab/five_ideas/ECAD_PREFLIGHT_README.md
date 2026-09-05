# ECAD conflict preflight

This is a diagnostic-only, no-GPU preflight for Evidence Conflict-Aware
Diversification. It does not run a selector, retriever, Generator, evaluator,
or Wiki-DPR experiment.

Run from any project checkout with:

```bash
./scripts/collab/five_ideas/run_ecad_preflight.sh
```

The default run validates the detailed case-study `cases` schema and computes a
conservative lexical conflict proxy. The proxy is a negative control: it is
reported but can never pass the ECAD gate. A real pass requires a separately
generated `ecad_pairwise_nli.v1` artifact supplied with `--pairwise-nli`.

The pairwise artifact must contain exactly 50 cases, 1,225 unordered pairs per
case, and provenance fields `model_id`, `revision`, and `config_sha256`. Each
pair has `left_id`, `right_id`, and probabilities `contradiction`,
`entailment`, and `neutral` in [0, 1]. It must be generated from the same fixed
Top-50 artifact and is never fed back into the production selector by this
preflight.

Outputs are compact `result.json`, `summary.json`, and `run_metadata.json` in
`exchange/five_ideas/ecad_preflight/<timestamp>/`. They contain hashes,
numeric diagnostics, and gates only; no question, passage, answer, prediction,
or provider response is persisted.

The ECAD gate requires: a valid pairwise-NLI artifact, non-degenerate conflict
structure, conflict burden independent of Answer Scorer, at least 20% changed
Top-5 sets, at least 30% conflict reduction, and no more than a 5% silver
positive-retention drop. Passing this gate only permits a later design review;
it does not authorize a selector screen or collaborator handoff.
