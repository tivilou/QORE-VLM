# Four full-case incremental support review verification

- Local baseline `e897981`, canonical baseline `94d323046d806a489737f44130a32615b54373e2`.
- Explicit167 model judgments, prior282 preserved unchanged;449 unique IDs/1600,1151 pending;cases1-4 each50/50. Quotes exact, source hashes and question/text links checked. No semantic label heuristics, new inference or fitting.
- Local replay and tamper checks10/10; exact commands/inputs/literal outputs/status in `semantic_supervision_20261009T045728Z_fullcase_01_04_local_verification.json`.
- Canonical publication stage `/tmp/qore-fullcase-review-publication-20261009-01-04/`; reopened modified archive, binary patch, literal baseline/modified command record and runnable hash-guarded rollback there. Server checks and scratch rollback must pass before commit/push.
- Four roles: `modified.tar.gz`, `patch.diff`, `verification.json`, `rollback.py`. Check-only rollback protects live committed files; actual rollback tested on isolated scratch copy.
- Neither this checkpoint nor old training-subset labels qualifies fresh supervision or L1/L2.
