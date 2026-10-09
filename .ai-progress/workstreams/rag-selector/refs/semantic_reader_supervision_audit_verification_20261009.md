# Train-only semantic supervision audit: verification and rollback

## Baseline / changed behavior

Canonical baseline is `5d4ad068105ea38f1cd934aef6da22e380499de4`, branch `five-ideas-development`, `/home/Q-DUET-VLM/QORE-VLM-phase1-gate`. Local mirror baseline `3aa97d0`. This adds an isolated observation exporter; no existing Reader, head, optimizer, selector, retrieval core, Generator, evaluator or old experiment config is edited. Original repair wrapper remains valid.

New path: pinned old training identities/checkpoints + existing collaborator caches -> fixed32 x50 Reader/old-head replay -> full raw case/NPZ and blinded review -> compact summary -> automatic namespaced upload. No optimizer/backward, evaluation-input read or generated answer is present. Prospective split is explicitly not heldout for the old heads.

## Exact commands and results

Local existing `D:/Software/Dev/Language/Anaconda3/envs/lambeq/python.exe`:

```text
python -m unittest scripts.collab.five_ideas.test_semantic_reader_supervision_audit scripts.collab.five_ideas.test_semantic_reader_repair_audit scripts.collab.five_ideas.test_semantic_reader_audit -q
Ran 36 tests
OK
exit 0
```

Same command with server `/usr/local/miniconda3/envs/py310/bin/python`: `Ran 36 tests`, `OK`, exit 0. No installs.

```text
bash -n scripts/collab/five_ideas/run_semantic_reader_supervision_audit.sh
stdout empty; exit 0
bash scripts/collab/five_ideas/run_semantic_reader_supervision_audit.sh --validate-only
{"status": "valid", "cases": 32, "training_allowed": false, "methods": ["frozen_reader_topk", "quantum_semantic", "classical_semantic", "quantum_scalar_control", "monotonic_compression_control"]}
exit 0
bash scripts/collab/five_ideas/run_semantic_reader_supervision_audit.sh --preflight /tmp/supervision-audit-server-final-preflight-20261009
{"project_v2_structure": "pass", "sample_split_identity": "pass", "complete_detached_values": "pass", "all_head_replay": "pass", "initial_null": "pass", "compression_null": "pass", "weak_metric_replay": "pass", "prior_two_witness_parity": "pass", "hidden_pooling_slices": "pass", "raw_input_hashes": "pass", "blinded_target_boundary": "pass"}
exit 0
```

Local preflight used `C:/Users/14630/AppData/Local/Temp/qore-supervision-audit-final-preflight-20261009`; same 11 checks pass, exit 0. This is a real synthetic trace fixture, not claimed real-data inference. Detached NPZ is reopened with `allow_pickle=False`; all coordinates/values and initial/final heads replay. Tampered question/text/rank/selection and pending/fabricated support judgments fail tests.

Shared canonical plugin validator:

```text
python E:/Claude.Code/本地助手/ai-skills/research/research-plugin-architecture/scripts/validate_plugin_plan.py configs/experiments/semantic_reader_supervision_audit_plan.json
OK: 6 plugins validated
exit 0
```

The trace uses the project-owned v2 structural/readiness validator, not an assertion that incompatible historical generic v1 validation accepted it. Source and fixture hashes are in `semantic_reader_supervision_audit_preflight_20261009.json`.

## Transport

Server command `/usr/local/miniconda3/envs/py310/bin/python /tmp/verify_supervision_audit_transport.py`:

```text
{"input_http_download": "pass", "source_files": 3, "input_manifest_sha256": "0464da7792e0fd2a0615c2e5a9058b641ef0d3b3300af6d0b39450b44fd39a81"}
{"directory_creation": "pass", "upload_receipts": "pass", "synthetic_only": true, "files": 9, "target_directory": "five_ideas/development_gate_smoke/semantic-supervision-audit-20261009T005814Z"}
exit 0
```

The service-owned secret is read privately into the process environment, not printed or recorded. This smoke uses only synthetic files in the development-gate namespace. Real collaborator output directories are created at runtime under `five_ideas/semantic_reader_supervision_audit/<UTC-run-id>`.

Original summary/trace/NPZ hashes were verified before extracting `/srv/qore-collab-uploads/five_ideas/semantic_reader_supervision_audit_input/repair_20261008T145652Z/`. Checkpoint arrays were reopened and exactly compared to their source; no evaluation arrays or raw evaluation labels/text were exported.

## Publication recovery roles

Server publication package is scoped to the explicitly owned code/config/progress/log/ignore files, preserving all unrelated tracked/untracked changes and historical raw results:

- Modified package: `/tmp/qore-supervision-audit-publication-20261009/modified.tar.gz`
- Exact baseline-to-modified patch: `/tmp/qore-supervision-audit-publication-20261009/patch.diff`
- Literal commands/stdout/stderr/exit statuses: `/tmp/qore-supervision-audit-publication-20261009/verification.json`
- Hash-guarded runnable rollback: `/tmp/qore-supervision-audit-publication-20261009/rollback.py`

Rollback defaults to check-only; `--apply` restores original tracked bytes and removes only the newly added owned files, after validating current and backup hashes. Actual restoration is tested on an isolated scratch copy, never the live code worktree. Immutable exchange inputs and original experiment results are retained; rollback does not delete corpus/model/artifact directories. Exact baseline validation and scratch rollback results are to be captured in the publication record before final completion.

## Handoff boundary

No real 32-case export has run here. Await collaborator outputs and independent blinded support judgments. No new training objective, supported-passages accuracy, L1/L2 or quantum advantage is claimed by this engineering checkpoint.

Progress doctor: `errors=0`, nine warnings for existing oversized/missing/indexed legacy progress records. Other workstream contents were not loaded or modified; broad progress cleanup is not part of this implementation.
