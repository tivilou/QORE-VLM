# Diagnostics Pilot: 20260809T153657

Completed the five configured runs on 200 `nq_open` validation questions using
the `wiki_dpr` corpus, seed 42, `K=5`, and Llama 3 8B generation. Retrieval found
a gold passage in the Top-50 for 156/200 questions in every configuration.

| Configuration | Recall@5 | Redundancy | EM | F1 |
|---|---:|---:|---:|---:|
| `qore_dpr` | 0.3196 | 0.8090 | 0.2500 | 0.4406 |
| `qore_as_control` | 0.4344 | 0.7830 | 0.2750 | 0.4950 |
| `qore_as_idea6` | 0.4454 | 0.7876 | 0.2950 | 0.5092 |
| `topk_as` | 0.5098 | 0.8233 | 0.2800 | 0.4926 |
| `mmr_as` | 0.4566 | 0.7968 | 0.2750 | 0.4856 |

Recall@5 is conditional on a Top-50 retrieval hit, matching the evaluator's
reported metric. Standard deviations and progress output are retained in each
configuration's `log.txt`; exact invocations are in `command.txt`.

The raw `result.json` files are intentionally excluded from Git because
`--dump_passages` makes each file approximately 295-298 MB.

## Compact follow-up analysis

After pulling the analyzer commit, generate the passage-free analysis locally:

```bash
python scripts/collab/five_ideas/analyze_diagnostics_pilot.py \
  exchange/five_ideas/diagnostic_pilot/20260809T153657
```

Then commit and push only
`exchange/five_ideas/diagnostic_pilot/20260809T153657/analysis/`. The analyzer
does not retain question text, gold answers, predictions, passage text, or
embeddings.
