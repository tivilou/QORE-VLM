# Q-DES: Question Evidence Schema Feature Preflight

## Quick Start

Run the one-click canonical no-GPU preflight analysis:

```bash
./scripts/collab/five_ideas/run_qdes_preflight.sh
```

This will load the detailed case-study artifact at
`research-web/apps/experiment-results/case-studies/rag-selector-50-detail.json`,
validate its root `cases` field and fixed 50 x 50 accounting, compute the
question-only typed signal, compare it with generic relevance controls, and
write compact results to `exchange/five_ideas/qdes_preflight/<timestamp>/`.
No selector, model, GPU, Generator, evaluator, or panel call is run.

## Requirements

- Python 3.8+ (the canonical preflight uses only the standard library)
- No GPU required (pure feature engineering, no model inference)
- Runtime: < 5 minutes

## Output Artifacts

All outputs are written to `exchange/five_ideas/qdes_preflight/<timestamp>/`:

- **result.json**: Compact parser, coverage, AUC, gate, and per-question diagnostics
- **summary.json**: Metrics and gate status
- **run_metadata.json**: Panel file hash, timestamp, Python version

Result size: ~6 KB (can commit to Git).

## Decision Gate

The preflight **passes** only when all explicit gates pass: real `cases` schema
and fixed 50 x 50 accounting, parser success, non-degenerate typed coverage,
typed-vs-generic separability, silver passage-label availability, and the
no-feedback contract. A failed gate disables the Q-DES candidate; it is not a
permission to run a selector screen.

**Pass**: Authorize a later Phase Q-DES selector-modification design.
**Fail**: Stop the Q-DES direction.

## Privacy Contract

- **Observation-only**: No selector, generator, or retrieval mutation
- **Compact output**: Only numeric features and summary statistics
- **No forbidden fields**: Raw questions, passage text, gold answers, and predictions are not persisted
- **Deterministic**: All features are deterministic given frozen Top-50 and QORE Top-5 selections

## Custom Panel File

To run with a different panel file:

```bash
./scripts/collab/five_ideas/run_qdes_preflight.sh --panel /path/to/panel.json
```

## Design Document

See `QDES_QUESTION_EVIDENCE_SCHEMA_DESIGN.md` for the hypothesis, feature
definitions, and decision gate specification. The legacy
`question_evidence_schema_preflight.py` filename delegates to the same
canonical implementation for compatibility.
