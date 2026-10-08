# Q-ARCG 100-case selector-trace audit

## Evidence and publication status

- Run directory: `five_ideas/qarcg_reader_screen_100/20261007T155432Z/`.
- Verified server artifact: `/srv/qore-collab-uploads/five_ideas/qarcg_reader_screen_100/20261007T155432Z/selector_trace.json`.
- Size: 996410 bytes; SHA-256: `0f7e624a97a3e47bf6458fc17c49d1e4941815034fd04fedb0c1d9962e820afe`, matching the service receipt.
- Schema: `sample-trace.v2`; artifact type: `qarcg_reader_screen_selector_trace`.
- Input SHA-256: `669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731` matches the registered fixed 100-case detail input.
- Reader: `facebook/dpr-reader-single-nq-base`; resolved revision `38f47a4986084c53447ba92ab0a83076b58d86a8`; config SHA-256 `383c5dbe735db108976bcaef70048348151def480bfe24ed8ff5243cd1ce4751`; CUDA; frozen body recorded by trace.
- GitHub branch/tree inspection found no Q-ARCG compact run artifacts on `main` or `five-ideas-development`. At inspection the development branch remained at `1751d8b7fec38fc7eb748fc39675c4835e98af29`. This is not verified as the collaborator's executed revision.
- Exchange directory contains only trace and upload receipt. Missing: `summary.json`, `report.md`, `run_metadata.json`, `upload_manifest.json`. Actual training coverage, losses, runtime code/config identity, and execution-time leakage checks remain unverified.

## Independently recomputed selection results

Silver overlap means membership intersection with the stored Silver Top-5, not answer correctness or official passage relevance.

| Arm | Total Silver members / 500 | Mean / 5 | Top-5 membership changed vs Reader | Better / equal / worse vs Reader |
|---|---:|---:|---:|---|
| Frozen Reader Top-k | 355 | 3.55 | 0 | Reference |
| Trained Q-ARCG | 349 | 3.49 | 13 cases, 14 replacements | 2 / 90 / 8 |
| Trained same-budget classical control | 353 | 3.53 | 3 cases, 3 replacements | 0 / 98 / 2 |

| Paired comparison | Mean delta / 5 | Percentile bootstrap 95% CI | Better / equal / worse |
|---|---:|---|---|
| Q-ARCG minus Reader | -0.06 | [-0.12, 0.00] | 2 / 90 / 8 |
| Classical minus Reader | -0.02 | [-0.05, 0.00] | 0 / 98 / 2 |
| Q-ARCG minus classical | -0.04 | [-0.10, 0.01] | 2 / 92 / 6 |

Recomputation used the existing `lambeq` Python environment (NumPy 1.23.5), no package installation. For each comparison, subtract per-case `silver_overlap`; bootstrap 2000 question-level samples with NumPy `default_rng`, seeds 20261006, 20261007, and 20261008 respectively; take quantiles 0.025 and 0.975. The lower bounds reproduce the runner's configured algorithm. Both declared positive paired gates fail; these intervals do not establish a statistically significant population degradation.

## Contract checks

- 100 unique question hashes, 50 unique candidate hashes per case, all three arms present.
- Every score vector has 50 finite numbers. Each selected set has five unique candidate hashes belonging to its corresponding Top-50. Every posthoc Silver set count is five.
- All recorded residual maxima satisfy `residual_bound * std(base_scores, ddof=0)` with tolerance `1e-5`; largest bound fraction is 0.920972. A raw residual-versus-coefficient comparison would be incorrect because the bound scales with the baseline score spread.
- Trace completeness is partial overall: ranking outputs are present; training examples, effective coverage, learning trajectory, per-candidate gates/features/observables, and checkpoint identity are absent. Full span-semantic failure attribution is not supported by this trace alone.
- Generator and evaluator are excluded by runner/config design; no generation result is emitted here. Runtime boundary declarations await compact metadata.

## Descriptive changed-case audit

All changed cases are included, not a selected favorable subset. Rank below is **Reader score rank**, not original retrieval rank. Q-ARCG improves cases 24 and 54, worsens 13, 19, 61, 69, 70, 73, 82, 90, and changes membership without overlap change in 55, 66, 79.

| Case | Reader -> Q overlap | Added Reader-score ranks | Removed Reader-score ranks |
|---:|---|---|---|
| 13 | 5 -> 4 | 6 | 5 |
| 19 | 5 -> 4 | 6 | 5 |
| 24 | 4 -> 5 | 6 | 5 |
| 54 | 3 -> 4 | 6 | 5 |
| 55 | 1 -> 1 | 6 | 5 |
| 61 | 3 -> 2 | 6 | 5 |
| 66 | 3 -> 3 | 9 | 5 |
| 69 | 5 -> 4 | 6 | 5 |
| 70 | 5 -> 4 | 6 | 5 |
| 73 | 5 -> 4 | 6 | 5 |
| 79 | 2 -> 2 | 6 | 5 |
| 82 | 4 -> 3 | 7 | 4 |
| 90 | 4 -> 3 | 6, 7 | 3, 5 |

No candidate originally below Reader-score rank 9 enters Top-5. At aggregate level two additional Silver memberships are outweighed by eight lost memberships. Five initially perfect cases lose one membership each (13, 19, 69, 70, 73).

Mean recorded case-level gate: Q-ARCG 0.817184036, range [0.7649601, 0.8459610]; classical 0.795189322, range [0.7703412, 0.8405573]. These are averages of 50 gates, not gate-open rates, and do not reveal candidate-level discrimination. Q-ARCG final-score corrections cover [-1.4596844, 0.8062257]; 2432 are negative and 2568 positive. Thus the head is active, not an unchanged zero-residual path. Limited membership changes despite active corrections suggest insufficient differential rescue, not necessarily insufficient overall correction magnitude.

## Interpretation and next action

- Measured: unlike the earlier unconditional span fusion (2.20/5), the anchored trained head stays close to Reader Top-k. It still does not deliver a selection win or quantum-control advantage.
- Hypothesis: weak answer-string labels may not teach evidence applicability; average gates are broadly high; four compact scalar features may discard evidence semantics. Training budget or convergence could also explain the result. None is established causally without the missing training records and candidate-level features.
- Stage: bounded screen; evidence ceiling `L0_diagnostic`; continuation `inconclusive` for the broader trainable mechanism, but this configuration is not promoted. The negative mean does not authorize retuning against the exposed 100-case Silver panel.
- Recommended next step: recover the already generated compact artifacts, verify training coverage/loss and code/config provenance, then join all 13 changed cases to the pinned detail/Reader trace for a loss-versus-rescue audit. No dataset download, new training run, quantum-depth increase, or formal evaluation is authorized by this audit.

## Follow-up correction, 2026-10-08

The full-panel label join is now complete. Six of the eight fixed-set overlap losses leave positive/direct evidence counts unchanged. Q-ARCG versus Reader retains 204/203 positive-consensus passages and 138/138 direct-consensus passages; the gain is inconclusive. The earlier reference to five initially perfect cases describes exact-set membership, not five correct evidence selections. See [semantic audit](qarcg_screen_20261007_semantic_case_audit.md) and [compact computed output](qarcg_screen_20261007_paired_case_audit.json). Training provenance remains missing.
