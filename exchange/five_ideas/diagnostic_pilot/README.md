# Five-Idea Diagnostics Pilot

Development-pilot runs for the five retrieval and selection configurations.
Raw `result.json` files contain dumped Wiki-DPR passages and remain local per
`exchange/README.md`. The runner now analyzes those files in place and writes
passage-free reports and QUBO payloads under the run's `analysis/` directory.

| Run | Samples | Seed | Status |
|---|---:|---:|---|
| `20260809T153657` | 200 | 42 | Complete (5/5); compact analysis pending collaborator run |

## One-command run

Future pilots automatically run the compact analyzer after all five
configurations finish:

```bash
bash scripts/collab/five_ideas/run_diagnostics_pilot.sh
```

To analyze the existing 20260809 run without rerunning any model or retrieval
experiment:

```bash
python scripts/collab/five_ideas/analyze_diagnostics_pilot.py \
  exchange/five_ideas/diagnostic_pilot/20260809T153657
```

Commit only the generated `analysis/` directory. Keep all five `result.json`
files local. The analyzer validates matched question IDs, retrieval hits, shared
configuration, aggregate metrics, and complete solver-faithful QUBO diagnostics
before emitting any report.
