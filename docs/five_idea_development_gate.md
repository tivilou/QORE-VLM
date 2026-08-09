# Five-Idea Development Gate

Phase 1 uses one YAML manifest to define the matched configurations, shared
evaluator arguments, paired comparisons, and cache declaration. The default
manifest is `configs/experiments/five_idea_development_gate.yaml`; the smoke
manifest is the same contract with 20 questions.

Run a smoke gate:

```bash
bash scripts/collab/five_ideas/run_diagnostics_pilot.sh \
  --smoke --skip-generation --allow-dirty
```

Run the 200-question development gate from a clean worktree:

```bash
python scripts/collab/five_ideas/run_development_gate.py
```

Each run records the normalized manifest, git/Python provenance, command and
elapsed time for every configuration, evaluator logs, and a passage-free
`analysis/` directory. The analyzer writes aggregate metrics, paired effects,
per-question records, QUBO diagnostics, plugin timing, and a compressed QUBO
payload. `summary.json` always includes `cache_stats`; Phase 1 deliberately
reports `available: false` with null hit/miss counts because Answer Scorer
caching has not been implemented.

The gate is a development decision point, not a final paper claim. Do not
start cohesion or other new objective work until the five configurations pass
the matched quality, redundancy, and latency review.
