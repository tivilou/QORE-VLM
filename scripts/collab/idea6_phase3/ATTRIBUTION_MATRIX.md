# Idea 6 Attribution Matrix

This script runs the clean control matrix needed to separate:

1. the effect of the DPR Answer Scorer;
2. the incremental effect of Idea 6 complementarity;
3. the comparison with Top-K and MMR under the same quality signal.

## Run

From a clean worktree at the repository root:

~~~bash
bash scripts/collab/idea6_phase3/run_attribution_matrix.sh
~~~

Activate the project conda environment first. If its executable is not named python, set it explicitly:

~~~bash
PYTHON_BIN=/path/to/env/bin/python bash scripts/collab/idea6_phase3/run_attribution_matrix.sh
~~~

A 20-sample smoke test is available:

~~~bash
bash scripts/collab/idea6_phase3/run_attribution_matrix.sh --smoke
~~~

The full run uses 3610 validation questions, generation enabled, and one seed (42). The current pipeline produced identical per-question results for seeds 42, 43, and 44, so uncertainty should be estimated with question-level paired bootstrap rather than treating seeds as independent replicates.

The script refuses tracked modifications and merge conflicts by default. Use --allow-dirty only for debugging, never for paper numbers.

## Configurations

| Directory | Method | Answer Scorer | Complementarity |
|---|---|---:|---|
| qore_dpr/ | QORE | no | no |
| qore_as_control/ | QORE | yes | no |
| qore_as_idea6/ | QORE | yes | DPR pairwise |
| topk_as/ | Top-K | yes | no |
| mmr_as/ | MMR | yes | no |

Each configuration stores its command, log, and result.json. The run root stores the Git commit, status, diff stat, and an automatically generated summary.md.

## GitHub placement

Commit only:

- scripts/collab/idea6_phase3/run_attribution_matrix.sh
- scripts/collab/idea6_phase3/summarize_attribution_matrix.py
- scripts/collab/idea6_phase3/ATTRIBUTION_MATRIX.md

Do not commit exchange/idea6_attribution/ result JSON files or archives. The repository already ignores large result artifacts; keep the small summary.md and metadata for a selected run only if the collaborator explicitly submits a report.
