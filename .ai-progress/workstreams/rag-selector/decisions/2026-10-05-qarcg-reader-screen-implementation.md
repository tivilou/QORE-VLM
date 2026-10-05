# Q-ARCG Reader-integrated 100-case screen implementation

- Date: 2026-10-05
- Scope: `rag-selector`
- Status: active candidate; collaborator runner implemented, real-data run pending
- Supersedes: None
- Gate: collaborator screen must beat frozen Reader Top-k on the declared Silver diagnostic and beat the matched classical control for attribution
- Authorization: approved for collaborator execution only

## Decision

Implement a project-relative collaborator runner that trains only the Q-ARCG
head and its matched classical control on up to 512 `nq_open/train` questions.
The weak target is answer-string containment in retrieved Wiki-DPR text; it is
training-only, noisy, and never treated as official passage gold. The frozen
DPR Reader body, evaluation Top-50, production selector, Generator, and
evaluator remain outside the change.

The fixed 100-case detail artifact is evaluation-only. All three online arms
must finish ranking all cases before Silver oracle fields are opened for the
separate posthoc overlap diagnostic. The trace is exchange-only; compact files
contain aggregates, hashes, and provenance but no question/text/answer/selected
ID fields.

## Artifacts

- Runtime config: `configs/experiments/qarcg_reader_screen_100.json`
- Plugin plan: `configs/experiments/qarcg_reader_screen_100_plan.json`
- Runner: `scripts/collab/five_ideas/run_qarcg_reader_screen_100.py`
- One-command wrapper: `scripts/collab/five_ideas/run_qarcg_reader_screen_100.sh`
- No-model tests: `scripts/collab/five_ideas/test_qarcg_reader_screen.py`

## Validation

- Local screen contract tests: 4 passed.
- 3080Ti project Python/Torch: Q-ARCG + screen contracts 13/13 passed.
- Plugin-plan validator: 3 plugins validated.
- Wrapper Bash syntax and `--validate-only`: passed.
- No real task data, Wiki-DPR, DPR Reader checkpoint, Generator, or evaluator
  was loaded by this implementation validation.

## Next gate

The collaborator runs exactly:

```bash
bash scripts/collab/five_ideas/run_qarcg_reader_screen_100.sh
```

The result must be synced through GitHub for `summary.json`, `report.md`,
`run_metadata.json`, and `upload_manifest.json`; `selector_trace.json` must be
uploaded to the authenticated 18083 exchange. A pass is still L0 diagnostic
evidence because official passage gold and downstream generation are absent.

## Kill criteria

Stop Q-ARCG if split overlap, label leakage, null drift, residual-bound or
trace failures occur, or if the candidate fails the predeclared Reader and
matched-classical paired gates. A utility win without the matched-control win
does not support a quantum-specific claim.
