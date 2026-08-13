# Phase 2 Cohesion Gate

This gate compares the fixed, calibrated, and explicitly disabled cohesion
enhancer against the matched Answer Scorer QORE control. It uses the same
dataset, retrieval pool, K, seed, and evaluator as Phase 1.

Full `wiki_dpr` runs are reserved for the collaborator. From the project root:

```bash
bash scripts/collab/five_ideas/run_cohesion_gate.sh
```

The script uses `PYTHON_BIN` when supplied and otherwise resolves `python3` or
`python` from the caller's environment. Results belong under
`exchange/five_ideas/development_gate/<timestamp>/`; only the passage-free
analysis package should be committed and pushed.

Promotion criteria are: no supported F1 harm, stable recall, an improvement in
F1 or redundancy, and latency close to the Answer Scorer control. Stop after
this gate and report the result before starting Phase 3.
