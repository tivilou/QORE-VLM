# Session Outcome

- Workstream: rag-selector
- Recorded: 2026-10-13
- Topic: deterministic protected-boundary feasibility audit

## Outcome

Implemented and ran a deterministic post-hoc audit over the validated
59-question/228-pair training screen and 100-question evaluation screen. The
audit computes boundary-crossing demand, pair-relative demand, configured
two-sided residual capacity, observed correction, true pair inversion, and
set-valued Top-5 changes. It exports compact hash/index/numeric artifacts only.

## Findings

- 168/228 (73.7%) protected pairs are structurally unreachable under the
  current residual bound; 60 are reachable by the loose bound but were not
  learned.
- Quantum and classical heads invert 0/228 protected pairs. Quantum has 46
  harmful pair corrections versus 3 for the classical head.
- Set-valued Top-5 changes are 4/100 quantum and 5/100 classical; ordering-only
  swaps are excluded.
- The current candidate remains closed. The next idea must change the score
  formation/calibration interface rather than merely increase epochs or residual
  amplitude.

## Artifacts and verification

- `scripts/collab/five_ideas/audit_protected_boundary_feasibility.py`
- `.cache/protected_boundary_feasibility_audit/20261013T000000Z/audit.json`
- `.cache/protected_boundary_feasibility_audit/20261013T000000Z/report.md`
- `docs/rag-research-log/20261013T-protected-boundary-feasibility-audit.md`
- `python -m py_compile scripts/collab/five_ideas/audit_protected_boundary_feasibility.py`

The input case study SHA and all per-pair calculations are recorded in the
generated audit JSON. No dataset download, collaborator run, or selector change
was performed.

## Git snapshot

- Repository: `E:/Claude.Code/Paper.Writing/Q-DUET-VLM`
- Branch: `main`
- Commit: `Audit protected boundary feasibility` (hash captured by the final Git snapshot)
- Unrelated worktree changes were left unstaged.
