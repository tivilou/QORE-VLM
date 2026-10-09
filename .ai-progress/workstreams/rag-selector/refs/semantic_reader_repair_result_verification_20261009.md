# Repaired semantic Reader result verification

Analysis date: 2026-10-09; experiment run `20261008T145652Z`.

## Source, receipts and split

- GitHub branch was `five-ideas-development` / `b2ca094aed4eccf266d90fff4adedb826081af22` at discovery; no new collaborator result commit. The result was complete on 18083. Four receipt-verified compact files are copied to `exchange/five_ideas/semantic_reader_training_repair_100/20261008T145652Z/` for focused publication. Raw trace/arrays remain exchange-only.
- Authoritative service directory: `/srv/qore-collab-uploads/five_ideas/semantic_reader_training_repair_100/20261008T145652Z/`.
- All six artifact/manifest receipts match SHA256, size and namespace. Raw selector trace: 11,607,670 bytes / `7e69b3efe0f733f15c68e2ea62bdd7c633e7f8122718bf0d6af90cb7b0fff258`; numeric archive: 81,799,848 bytes / `573209aa882b4d8122ffae26b2dc37bff214fb7882744e5dae4d3d3e22d76843`.
- Config/plan and all emitted source hashes match Git blobs at `b2ca094`; frozen input SHA256 `669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731`. Model/revision and Reader baseline are unchanged. All 434 usable weak training identities are disjoint from 100 evaluation hashes; skipped-question hashes remain unrecorded.

## Deterministic audit commands

Existing server `py310` environment; no installation, model loading, dataset download, fitting or new task experiment:

```bash
python scripts/collab/five_ideas/analyze_semantic_reader_training_repair.py --run-dir /srv/qore-collab-uploads/five_ideas/semantic_reader_training_repair_100/20261008T145652Z --detail /srv/qore-collab-uploads/five_ideas/selector_replay_100_historical_input/silver-oracle-top5-100-20260915T120525Z-detail.json --source-revision b2ca094 --prior-dir /srv/qore-collab-uploads/five_ideas/quantum_semantic_reader_screen_100/20261008T101111Z --output /tmp/semantic_reader_repair_20261008T145652Z_audit_final_v2.json
```

Exit 0. Literal validation fields each equal `pass`: `detached_values`, `pooling_slices`, `training_witness_replay`, `all_evaluation_replay`, `initial_null`, `optimizer_witness_replay`, `project_trace_structure`, `project_trace_selection`, `json_archive_agreement`. Eighteen sampled pre/post optimizer updates replay at declared tolerance, not producer-GPU byte identity. All 100 metrics, bootstrap intervals and reported gates recompute exactly.

Previous-run comparison returns `reader_score_vectors_exact=100`, `usable_train_identities_exact=true`, `first_two_train_inputs_exact=true`. All per-step data-gradient arrays have shape `1302×4`, finite nonnegative norms; check-window sums equal producer reports and all train-only health gates pass. Initial/epoch parameter norms and first-two signal health independently recompute.

Local `lambeq` separately rejoined all 100 case/candidate identities from the complete received trace and registered detail file, recomputed metrics/bootstrap, and reported literal `Independent local all-100 metrics/bootstrap/identity recomputation: pass`, exit 0. Complete raw NPZ replay was on the server, not local.

```text
python -m unittest scripts.collab.five_ideas.test_semantic_reader_repair_audit -q
```

Local `lambeq`: `Ran 7 tests`, `OK`, exit 0. Server `py310`: `Ran 7 tests`, `OK`, exit 0. Checks cover constant offsets, score-range boundary necessity, real swaps, ties, nonfinite inputs and architectural posthoc reachability. Audit alone never changes scoring or original results.

Shared validators: recovery dossier passes; evidence card is valid at `L0_diagnostic` with two expected warnings (screen inconclusive; unresolved classical attribution). Strict promotion is deliberately not asserted. These validators are evidence schemas, not task-success gates.

## Results and inference boundary

Reader/quantum semantic/classical semantic/scalar positive totals `203/204/203/203`; direct totals `138/139/138/139`; fixed-set totals `355/356/354/353`. Quantum positive paired `1/99/0`, delta `+0.01`, CI `[0,0.03]`; both declared gates false. All fourteen changed method/case sets are audited.

307 positive-capacity slots versus Reader 203 is an exposed Silver diagnostic ceiling, not a deployable selector or answer-success ceiling. The `18/75`, `63/78`, `110/112` unreachable candidate counts use the bounded-score necessary condition versus the weakest nonpositive Top-5 member; they are not independent questions or simultaneously feasible slots. Candidate correction-direction comparisons use the same 57 rescue-eligible cases, are rank-band descriptive, and do not establish causality.

## Raw cache integrity and storage incident

The E drive filled during transfer; its incomplete current-run cache was moved with checked literal paths to `C:/Users/14630/AppData/Local/Temp/qore-semantic-reader-repair-20261008T145652Z/`. Original server files were not modified. A later SCP session closed; completed compact files/trace and audit reports were individually retained. The incomplete NPZ failed the receipt hash and is explicitly named `semantic_values.npz.partial`. It is not accepted for replay or copied into Git. No unrelated local data was deleted. Future work should use server originals for numeric replay or a hash-verified resumed transfer to a drive with space.

Compact independent audit SHA256: `072ceaaf8b83b8158ced1207788578c5b1071ae62119184100b83f0748a9c04c`, stored as `semantic_reader_repair_20261008T145652Z_audit.json`. Full raw changed-case excerpts remain only in the non-Git cache and server `/tmp/semantic_reader_repair_20261008T145652Z_audit_final_v2_case_details.json`; progress stores compact numeric and semantic findings.

## Remaining

No new training mechanism, experiment, Generator output, quantum superiority or L1/L2 claim. Supervision-quality/boundary discrimination remains the next gated diagnosis. Prior missing Q-ARCG and span-fusion provenance are separate and unreconstructed.
