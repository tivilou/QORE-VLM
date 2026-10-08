# Semantic Reader implementation verification

Only synthetic fixtures and locally initialized tiny model classes were executed. No task data or pretrained model was downloaded. Production retrieval/selector/Generator/evaluator code is unchanged.

## Local commands and literal results

Environment: existing `lambeq`, Python executable `D:/Software/Dev/Language/Anaconda3/envs/lambeq/python.exe`, Torch `2.3.1+cpu`, Transformers `4.41.2`.

```text
python -m unittest applications.rag.tests.test_qarcg_reader scripts.collab.five_ideas.test_qarcg_reader_screen scripts.collab.five_ideas.test_qarcg_case_audit applications.rag.tests.test_quantum_semantic_reader -v
Ran 28 tests in 2.667s
OK
exit: 0

python scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py --preflight E:/Dir/CodexHome2/tmp/Q-DUET-VLM/semantic_reader_preflight
{"detached_values": "pass", "pooling_slices": "pass", "training_witness_replay": "pass", "all_evaluation_replay": "pass", "initial_null": "pass", "project_trace_structure": "pass", "project_trace_selection": "pass", "json_archive_agreement": "pass"}
exit: 0

python E:/Claude.Code/本地助手/ai-skills/research/research-plugin-architecture/scripts/validate_plugin_plan.py configs/experiments/quantum_semantic_reader_screen_100_plan.json
OK: 4 plugins validated
exit: 0

python E:/Claude.Code/本地助手/ai-skills/research/experiment-grounded-ideation/scripts/validate_mechanism_recovery.py .ai-progress/workstreams/rag-selector/refs/semantic_reader_recovery_20261008.json
OK: mechanism-recovery dossier validated
exit: 0
```

Local fixture JSON SHA256: `f69ca6ddaf7e1b4afa1449dcc202e07b5ac6ec390964975a6b8c19e9dc731ea2`.
Local detached NPZ SHA256: `9f24e1bdc56e63f59ba04ae09538daa17173683f9da782166d3a55c686d42ee4`.

## Canonical server commands and literal results

Checkout: `/home/Q-DUET-VLM/QORE-VLM-phase1-gate`; existing `py310` at `/usr/local/miniconda3/envs/py310/bin/python`, Torch `2.7.1+cu126`, Transformers `4.44.0`. Commands below ran in that checkout through strict key-authenticated SSH on the primary development role.

```text
python -m unittest applications.rag.tests.test_qarcg_reader scripts.collab.five_ideas.test_qarcg_reader_screen applications.rag.tests.test_quantum_semantic_reader -q
Ran 21 tests in 3.408s
OK
exit: 0

python scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py --validate-only
{"status": "valid", "stage": "exploratory_screen", "methods": ["frozen_reader_topk", "quantum_semantic", "classical_semantic", "quantum_scalar_control"]}
exit: 0

python scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py --preflight /tmp/qore-semantic-reader-preflight
{"detached_values": "pass", "pooling_slices": "pass", "training_witness_replay": "pass", "all_evaluation_replay": "pass", "initial_null": "pass", "project_trace_structure": "pass", "project_trace_selection": "pass", "json_archive_agreement": "pass"}
exit: 0

bash -n scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.sh
output: empty
exit: 0

python -m py_compile applications/rag/quantum_semantic_reader.py scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py
output: empty
exit: 0
```

Server fixture JSON SHA256: `6cb69577b773466382e918b99d0f0423d67bff8b2d303224e8e87298eb92e5f3`.
Server NPZ SHA256: `8e237346a3ac25d403ca209acb0b69a12fb33a5d32edc9325997d1f4b2a0914f`.
Archive bytes are not expected to match across Torch/platform versions. Each environment independently reopens and replays its own values, with numerical comparison tolerance `rtol=atol=1e-4`; JSON score vectors agree exactly with their own archives. This is the project v2 validator, not the old shared generic v1 validator.

## Behaviors and rollback

- Initial semantic/scalar residual scale is zero: all heads reproduce the raw Reader relevance vector exactly (`rtol=atol=0`).
- A nonzero trained/synthetic semantic residual remains bounded, responds to semantic input, has finite gradients, and can be replayed from saved parameters. This is engineering behavior, not a real-data improvement claim.
- Incorrect saved Top-5 and saturated-encoding numerical failures are covered by regression tests.
- Rollback: stop invoking the isolated screen wrapper and use the unchanged production Reader/Top-k path. Do not delete prior outputs or alter production code. The new module has no production selector import/registration side effect.

Real Wiki-DPR retrieval, 512-question training, 100-question evaluation and automatic artifact upload are pending collaborator execution; synthetic readiness is not evidence that those experiments succeeded.
