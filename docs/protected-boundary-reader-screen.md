# Protected boundary Reader screen

Delivery checkpoint2026-10-12: ready for collaborator execution. The pinned
Reader preflight passed offline on the verified3080Ti;396 arrays,8 head output
replays and4 optimizer updates were checked. Both private inputs passed public
18083 GET byte/hash verification, and the canonical server's23 tests passed.
Only synthetic fixture targets were fitted during development; the real
59-question training and100-question evaluation remain unrun.

This is a fixed Top50 -> Top5 selector experiment. It trains a fresh four-qubit
semantic residual head and a matched classical head on **59 training
questions /228 adjudicated replacement pairs**, then scores the registered100
question panel without target masks. The DPR Reader body is frozen. No
Wiki-DPR retrieval, dataset download, Generator or EM/F1 evaluation is called.
The quantum circuit and0.25 residual bound are unchanged; the new variable is
support-qualified, question-normalized supervision with an immutable-slot loss.

Run from the collaborator's clone after updating `five-ideas-development` and
configuring the existing `QORE_EXCHANGE_TOKEN` environment variable:

```bash
bash scripts/collab/five_ideas/run_protected_boundary_reader_screen_100.sh
```

The wrapper discovers an existing Python>=3.10 with Torch, Transformers and
NumPy, including existing conda environments. No fixed Python/project/model
path is required. The pinned `facebook/dpr-reader-single-nq-base` model uses
the collaborator's Hugging Face cache; missing **model** files may be fetched,
but the script never downloads a dataset.

Two private inputs are needed. They are downloaded automatically when absent:

- `five_ideas/protected_boundary_reader_inputs/20261011-v1/frozen_support_bundle.zip`
  (26,121,092 bytes;SHA256
  `f723bbb6f9075c3e04fd3d8a7b2d3f8127b6e354422ee2a2f6e167bfe32248ce`).
  Original434-case export, final decisions/annotations/targets and all matching
  reviewer witnesses remain byte-identical. Their old training=false flags
  remain untouched; this screen's separate user-approved config supplies the
  execution boundary.87 validation rows are never fitted or relabeled.
- Existing registered100-case detail JSON,8,046,318 bytes;SHA256
  `669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731`.

Before real fitting, the same runner executes an actual pinned-Reader
preflight on four synthetic questions. Complete Reader hidden/pooling values,
CLS relevance weights, matched-head outputs, parameters, gradients and AdamW
moments must reopen and replay. Failure stops fitting. Development can run
only this bounded check using `--preflight-only --offline-model`.

The screen uses three fixed epochs, identical source order and parameter
budget, seed20261008, final epoch only and the repaired fan-in-scaled AdamW.
The loss protects every original selected slot except explicitly qualified
irrelevant negatives. This is a penalty, not a guarantee: report actual losses
after ordinary, unmasked Top5 selection. No Silver-based checkpoint choice or
residual amplification is allowed.

All100 cases count. Primary outcome is the number of positive-consensus
passages selected; direct-consensus retention is protected. Exact Silver-set
overlap is secondary. The diagnostic gate requires positive paired mean and
CI95 lower>0 versus frozen Reader, with nonnegative direct mean. A separate
comparison to the classical arm assesses quantum attribution. This exposed
panel and research judgments support **L0 only**, not generalization or
quantum advantage.

Outputs use
`exchange/five_ideas/protected_boundary_reader_screen_100/<UTC timestamp>/`.
The script automatically creates the matching18083 directory, uploads only
declared files and verifies byte/hash receipts. A required upload failure
exits nonzero and preserves `upload_manifest.json` for idempotent recovery.
Only its `github_files` may be committed: summary/report/run metadata,
manifest and upload receipts, all<=1MiB. Full texts, annotations, case-study
JSON/Markdown, full tensor archives and update traces are exchange-only.

`sample_trace.json` follows standard `sample-trace.v2` for the predeclared
first two samples in each lane. `case_study.json` /`case_study.md` cover all59
training and100 evaluation questions with fixedTop50, every score/Top5,
reference answers and explicit no-generation status. Training JSON also
includes actual supervision and per-question loss from every epoch. Full
hidden tensors are bounded to first candidate of the four fixed samples;
all159 base/pooled/head values and each epoch checkpoint remain available.
The private numeric coordinate reader checks the archive hash and permits
arbitrary valid coordinates; it reconstructs no missing state.
