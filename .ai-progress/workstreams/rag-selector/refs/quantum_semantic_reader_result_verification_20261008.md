# Current screen audit verification (2026-10-08)

## Commands and status

Selected existing environments: local `lambeq` (`D:/Software/Dev/Language/Anaconda3/envs/lambeq/python.exe`), server `py310` (`/usr/local/miniconda3/envs/py310/bin/python`). No installs, model/corpus downloads, task reruns or repair experiments.

Combined local and remote test command, run in the project checkout:

```bash
python -m unittest applications.rag.tests.test_qarcg_reader scripts.collab.five_ideas.test_qarcg_reader_screen scripts.collab.five_ideas.test_qarcg_case_audit applications.rag.tests.test_quantum_semantic_reader scripts.collab.five_ideas.test_semantic_reader_audit -q
```

- Local literal summary: `Ran 33 tests in 4.086s`, `OK`; exit `0`.
- Canonical server literal summary: `Ran 33 tests in 5.590s`, `OK`; exit `0`.
- Local Python compilation of `scripts/collab/five_ideas/analyze_quantum_semantic_reader_screen.py`: no stdout, exit `0`.

Local independent offline audit:

```bash
python scripts/collab/five_ideas/analyze_quantum_semantic_reader_screen.py --run-dir E:/Dir/CodexHome2/tmp/Q-DUET-VLM/semantic_reader_20261008T101111Z --detail research-web/apps/experiment-results/case-studies/silver-oracle-top5-100-20260915T120525Z-detail.json --output .ai-progress/workstreams/rag-selector/refs/quantum_semantic_reader_20261008T101111Z_audit.json
```

Exit `0`; reported checks: `all_evaluation_replay`, `detached_values`, `initial_null`, `json_archive_agreement`, `pooling_slices`, `project_trace_selection`, `project_trace_structure`, `training_witness_replay`, each `pass`. Semantic heads each report `exact_baseline_score_vectors: 100`, `changed_sets: 0`, residual/score-correction `nonzero: 0`.

Canonical server independent audit, using already-uploaded artifacts only:

```bash
python scripts/collab/five_ideas/analyze_quantum_semantic_reader_screen.py --run-dir /srv/qore-collab-uploads/five_ideas/quantum_semantic_reader_screen_100/20261008T101111Z --detail /srv/qore-collab-uploads/five_ideas/selector_replay_100_historical_input/silver-oracle-top5-100-20260915T120525Z-detail.json --output /tmp/qore-semantic-reader-result-independent-audit.json
```

Exit `0`; same eight replay checks and zero-correction semantic diagnosis. Temporary remote audit may differ in low-order arithmetic for gradient/norm probes across Torch versions; no claim of byte-identical independent audit JSON. Authoritative trace/NPZ are immutable and hashed against service receipts.

Canonical skill validators:

- `validate_mechanism_recovery.py` on `semantic_reader_training_collapse_recovery_20261008.json`: `OK: mechanism-recovery dossier validated`, exit `0`.
- `validate_evidence_card.py` on `quantum_semantic_reader_20261008_evidence_card.json`: `OK: evidence card is valid at L0_diagnostic (2 warning(s))`, exit `0`. Warnings explicitly retain short-screen inconclusiveness and unresolved classical attribution. Strict mode does not pass and is not claimed to.

## Interpretation boundaries

- File hashes identify the actual mask-fixed semantic source matching commit `44c9f70`; the report's HEAD `5ce58ba` alone is insufficient to describe the dirty execution.
- All declared replay contracts pass. This confirms what executed, not effective task learning.
- Label-free decay-only updates operate on a fresh initialized parameter copy with zero gradients and no data/labels. They nearly reproduce parameter shrinkage but are not a complete replay of the actual task optimizer.
- First two usable training inputs and all eval inputs are saved; other usable train features, all-step gradients/moments and skipped-question hashes are absent. No unsupported causal link is reconstructed.
- Production code and training configuration remain unchanged. Stop invoking the isolated unchanged screen as the immediate rollback; outputs remain preserved. No new experiment command is handed to the collaborator until training repair is authorized and qualified.
