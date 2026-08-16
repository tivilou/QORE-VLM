# Phase 7C Answer-Identity Ablation

This is a collaborator-only Wiki-DPR pilot for the answer-corroboration
mechanism. It uses a new held-out slice (validation questions 100-149), K=5,
seed 42, and one reference generator. It writes compact results under
`exchange/five_ideas/phase7c_answer_identity_ablation/<timestamp>/`.

From the repository root, run:

```bash
scripts/collab/five_ideas/run_phase7c_ablation.sh \
  --generator reference=/path/to/reference/model
```

The wrapper selects a usable `python`/`python3` from `PATH`; set `PYTHON_BIN`
only when the experiment environment is not first on `PATH`. The generator
override is optional when the committed model path is valid on the machine.

The pilot compares `qore_answer`, `topk_answer`, agreement-only QORE, and
corroboration-only QORE. `decisive_conflict` is recorded as a stricter
diagnostic and is not used as a QUBO penalty. Do not add raw questions,
passages, answers, or predictions to the Exchange output.
