# Phase 7D Independence-Residual Diagnostic

This is a collaborator-only Wiki-DPR diagnostic. It tests whether answer
agreement and answer corroboration retain an association with QA F1 after
accounting for three pre-declared nuisance variables: Answer Scorer confidence
product, lexical token-Jaccard duplication, and embedding cosine redundancy.

The residual fit is unsupervised and is performed separately for each question
over all candidate passage pairs. Gold answers, generated answers, EM, and F1
are never used to fit the residual coefficients. Residual features are recorded
after fixed `qore_answer`, `topk_answer`, and `qore_dpr` controls are selected;
they are never passed to the selector or QUBO objective.

The frozen pilot uses validation questions 150-199, K=5, seed 42, one reference
Llama3-8B generator, and a deterministic 2,000-resample bootstrap. The output
contains only hashed question/prediction identifiers, compact feature values,
fit diagnostics, the manifest, and the gate summary. It does not contain raw
questions, passages, answers, or predictions.

From the repository root, the collaborator runs:

```bash
scripts/collab/five_ideas/run_phase7d_independence.sh
```

The wrapper chooses a usable `python`/`python3` from `PATH`; set `PYTHON_BIN`
only when the experiment environment is not first on `PATH`. A model-path
override is available without fixing a Python executable:

```bash
scripts/collab/five_ideas/run_phase7d_independence.sh \
  --generator reference=/path/to/reference/model
```

The collaborator should commit the generated Exchange directory and the
source/config commit on `five-ideas-development`. A gate failure means the
answer-identity objective remains unsupported; it is not a negative result for
the established Answer Scorer QORE control.
