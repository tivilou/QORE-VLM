# Semantic Reader optimizer repair — verification

Date: 2026-10-08. This is a train-only engineering repair, not a new task result.

## Scope and source

- Local mirror: `E:/Claude.Code/Paper.Writing/Q-DUET-VLM`; prior HEAD `4a789c8`.
- Canonical checkout: `/home/Q-DUET-VLM/QORE-VLM-phase1-gate`, branch `five-ideas-development`; baseline `4cd8e74`.
- Changed training field: new config `training.optimizer_policy=adamw_projection_fanin_v1`. Historical default remains coupled Adam. Original Reader/head/config/plan stay unchanged.
- Data: existing archive SHA256 `4d18cca65cfc8db290020ccc1fea1c5d64773b95be488c9a68d78b9e807c581a`; only the first two saved usable training cases' numeric arrays are opened by qualification. All 64 updates and six head/policy comparisons are explicit in the compact report. No evaluation arrays or Silver are used.

## Exact command/output record

Complete baseline and modified commands, inputs, literal stdout/stderr and exit statuses are reopened in:

- Server: `/tmp/qore-semantic-reader-training-repair-20261008/verification.json`.
- Local backup: `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/semantic_training_repair_verification.json`.

Every one of its nine final commands exited 0. The first attempted deployment validation exited 1 solely for Windows/Linux source-line-ending identity; the guard was corrected explicitly, raw hashes retained, and final verification repeated. No task metrics or success gates were changed.

### Regression (baseline and repaired behavior)

```text
python -m unittest applications.rag.tests.test_semantic_reader_training applications.rag.tests.test_quantum_semantic_reader applications.rag.tests.test_qarcg_reader scripts.collab.five_ideas.test_qarcg_reader_screen scripts.collab.five_ideas.test_qarcg_case_audit scripts.collab.five_ideas.test_semantic_reader_audit -q
```

Local environment: existing `lambeq`, Torch 2.3.1 CPU. Server: existing `py310`, Torch 2.7.1, CPU qualification. No dependency installation.

Literal test result on both environments: `Ran 44 tests` and `OK`; exit 0. The emitted `training_health_failed=true` during this suite belongs to the synthetic failure-persistence regression, not the real repaired screen.

The original baseline is executed from the preserved pre-edit runner bytes, compiled with the canonical original `__file__` and `--preflight /tmp/qore-semantic-reader-training-repair-20261008/baseline_fixture`. All eight original value/pooling/witness/evaluation/null/JSON checks return `pass`. No original artifact or config is rewritten.

### Repaired wrapper and fixture

```bash
python scripts/collab/five_ideas/run_semantic_reader_training_repair_100.py --validate-only
python scripts/collab/five_ideas/run_semantic_reader_training_repair_100.py --preflight /tmp/qore-semantic-reader-training-repair-20261008/fixture
bash -n scripts/collab/five_ideas/run_semantic_reader_training_repair_100.sh
bash scripts/collab/five_ideas/run_semantic_reader_training_repair_100.sh --preflight /tmp/qore-semantic-reader-training-repair-20261008/wrapper_fixture
```

Config output: `status=valid`, four allowlisted methods. Repaired and wrapper fixtures return all nine checks `pass`, including `optimizer_witness_replay`. All exit 0. The three-update synthetic fixture's training-health status is honestly `deferred_below_64_updates`; the separate saved-training-case qualification supplies the learning-health check, not the fixture alone.

The shared `validate_plugin_plan.py configs/experiments/semantic_reader_training_repair_100_plan.json` reports literal `OK: 4 plugins validated`, exit 0. Python compilation and own-file diff checks pass.

### Train-only qualification

```bash
python scripts/collab/five_ideas/qualify_semantic_reader_training.py --input /srv/qore-collab-uploads/five_ideas/quantum_semantic_reader_screen_100/20261008T101111Z/semantic_values.npz --output /tmp/qore-semantic-reader-training-repair-20261008/independent_qualification
```

Output `passed=true`, exit 0. Historical coupled Adam and unscaled AdamW each fail the strengthened candidate-dependent correction guard on both heads; scaled AdamW passes both. Full moments/parameters and the final update/signals replay exactly within each CPU environment; cross-version values are not asserted byte-identical.

Local final values: `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/semantic_reader_optimizer_qualification_portable_20261008/qualification_values.npz`. Local compact report: `../refs/semantic_reader_training_qualification_20261008.json`. Server independent values/report: `/tmp/qore-semantic-reader-training-repair-20261008/independent_qualification/`.

## Four verified modification/recovery roles

All owned source originals were hashed/preserved before remote replacement. The installation compares every modified file against its manifest SHA256. Roles:

1. Modified artifact: `/tmp/qore-semantic-reader-training-repair-20261008/modified.tar.gz` (ten code/config/qualification paths plus manifest).
2. Patch: `/tmp/qore-semantic-reader-training-repair-20261008/patch.diff` (baseline-to-modified distribution diff).
3. Verification: `/tmp/qore-semantic-reader-training-repair-20261008/verification.json` (exact baseline/modified command evidence and role hashes).
4. Runnable rollback: `/tmp/qore-semantic-reader-training-repair-20261008/rollback.py`, with `operation.json` and `original/` backups.

The archive and patch were reopened. Live rollback runs check-only, verifies every current/backup hash, and reports literal `{"rollback": "check_passed", "files": 10}`, exit 0. Running the same rollback with `--root .../rollback_roundtrip --apply` on a disposable copy reports literal `{"rollback": "applied_verified", "files": 10}`, exit 0: original runner restored, newly added files absent. Live repaired code remains installed. Hash mismatches block later rollback rather than overwrite concurrent changes. Rollback covers the ten implementation/qualification paths, not separately appended progress/history documents.

## Completion and limits

Both verified behaviors: original exact-null/historical path remains replayable; repaired learning path has finite live gradients and nonconstant effective corrections, with optimizer-update replay and early-failure persistence. Original 100-question results are not changed. No repaired real-data screen, downstream answer score or quantum advantage is established. User controls running the separately registered collaborator wrapper.
