# Result inspection and bounded numerical replay

Input/source: canonical31a5c932de9b525cdf34eed688e9890226402cb9; local mirror47b42de. Raw nine artifacts at `/srv/qore-collab-uploads/five_ideas/semantic_reader_supervision_audit/20261009T045728Z/`; hashes/bytes in audit.json. No datasets/models downloaded, fitting, retrieval or Generator run here. Detached old-head replay only.

Exact independent command on canonical server:

```bash
cd /home/Q-DUET-VLM/QORE-VLM-phase1-gate
/usr/local/miniconda3/envs/py310/bin/python .ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_replay.py
```

Exit0; literal report fields `"source_identity": "pass"`, `"receipts": 9`, all11 original named replay fields `"pass"`, extra `"scalar_arithmetic_cross_environment_tolerance": "pass_at_1e-9"`. Weak totals87/87/87/89/87 and changed-case counts0/0/0/4/0. Full outputs captured in publication verification.json. Replayed NPZ stays on server; source raw hashes unchanged.

Historical `runner.validate_trace(run, cohort, checkpoint)` raises `ValueError: diagnostic arithmetic mismatch` when comparing saved/recomputed boundary/compression float dictionaries for exact equality. The independent replay.py preserves the production source but extracts that validator and replaces ONLY that comparison with recursive key/type/int identity and bounded float1e-9 comparison. No tensor/head/selection/provenance/hash tolerance is changed. It asserts the exact replacement target and reports140 differing scalars,max3.552713678800501e-15. This is a disclosed audit-only portability accommodation, not a statement that the original strict validator passed.

Local partial-review validation:73 distinct IDs all exist in sourceblind review,64 marked score-blind,9 exploratory; case IDs agree; every stored literal support_quote is contained in the source text; all labels marked official_gold=false. Source1600 labels leftnull. This verifies record linkage/quotes, not semantic truth or full coverage.

Only compact producer summary/report/run_metadata/upload_manifest and compact audit/review/analysis are versioned. Full case JSON/Markdown/review and NPZ remain exchange-only. The result does not establish support accuracy,L1/L2,generalization or quantum advantage.

Publication roles at `/tmp/qore-supervision-result-audit-publication-20261009/`: modified.tar.gz (changed artifact), patch.diff (baseline/modified diff), verification.json (literal command outputs/status), rollback.py (hash-guarded restoration, default check-only; --apply only after checking target hashes). Rollback executes against an isolated copy in validation, never rewinds live project progress. Original tracked files preserved and production code unchanged.
