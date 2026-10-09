# Boundary review publication verification

Inputs are unchanged collaborator case/score-blind review files for32x50 at `/srv/qore-collab-uploads/five_ideas/semantic_reader_supervision_audit/20261009T045728Z/`. Explicit primary-model labels are reviewed semantic input; scripts only validate/hash/join/count. The220-item scope is independently recreated; all282 labels link to source IDs and literal quotes; prior73 semantic records retained. No complete1600 claim.

Local6/6: full readback parity, reject missing scope, fabricated quote, duplicate/missing judgments and source SHA mismatch. Literal commands/outputs/exit statuses in local_verification.json. First readback detected cached prior-review CRLF vs immutable Git LF hashing; relinked ONLY prior_review_sha256 to actual immutable Git artifact; labels and aggregate data unchanged. Re-read and6/6 corrected checks passed. Original strict numerical validator defect remains separately disclosed in prior result verification; production source is not changed here.

Canonical command (existing artifacts only):

```bash
cd /home/Q-DUET-VLM/QORE-VLM-phase1-gate
/usr/local/miniconda3/envs/py310/bin/python .ai-progress/workstreams/rag-selector/refs/semantic_supervision_20261009T045728Z_boundary_tests.py --root . --source-dir /srv/qore-collab-uploads/five_ideas/semantic_reader_supervision_audit/20261009T045728Z --stage /tmp/qore-boundary-reviewed-record-checks-20261009
```

Canonical publication runs the above readback/tamper checks, exact output JSON equality and source/core hash parity. Reversible roles at `/tmp/qore-boundary-review-publication-20261009/`: modified.tar.gz, patch.diff, verification.json (exact baseline/current commands and outputs/status), rollback.py. Rollback is hash guarded, defaults check-only, and --apply is verified on isolated scratch; live original numerical/semantic source is never rewound during checks. Old immutable source and existing partial judgments preserved. Compact publication only; no Reader/model/corpus inference, dataset download, training or Generator call.
