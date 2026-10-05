# Q-ARCG Reader screen implementation

- Date: 2026-10-05
- Workstream: `rag-selector`
- Outcome: implemented a collaborator-only real-data screen for the active Q-ARCG candidate.

## Changed

- Added the fixed config and research-plugin v2 plan for a 512-question weak-
  supervision training slice plus the registered 100 x 50 evaluation input.
- Added a project-relative runner that freezes the DPR Reader body, trains Q-
  ARCG and the matched classical control with the same cases/loss/seed, and
  separates online ranking from posthoc Silver diagnostics.
- Added automatic exchange manifest/upload for the selector trace and a
  one-command wrapper.
- Added no-model contract tests and recorded the new implementation decision.

## Verification

- Local screen tests: 4/4.
- 3080Ti tests: 13/13 including Q-ARCG Torch adapter and screen contracts.
- Plugin plan: 3 plugins validated.
- Python compilation, JSON parsing, Bash syntax, and `--validate-only`: pass.

## Boundary

No formal experiment was run. No Wiki-DPR/model download was performed during
this implementation. The next action is the collaborator command recorded in
the decision file.

## Git snapshot

- Existing unrelated dirty files were preserved.
- Agent-owned new screen files are not yet committed in this session; commit
  only the listed code/config/progress files after the final ownership check.
