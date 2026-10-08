# Q-ARCG paired semantic audit and membership-metric correction

## Sources and matching

All 100 registered cases and all 13 Q-ARCG membership-change cases were joined by question SHA-256 and ordered candidate SHA-256. The 13-case subset is explicitly post-hoc, exploratory, and selected on membership change, not a confirmatory mechanism sample.

- Detail input SHA-256: `669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731`.
- Q-ARCG selector trace SHA-256: `0f7e624a97a3e47bf6458fc17c49d1e4941815034fd04fedb0c1d9962e820afe`.
- Historical frozen Reader candidate trace SHA-256: `13db49c56c0a5d61edf28156c5308e958f2c279e56867c6e773a135c81d56fa2`.
- Historical Reader and Q-ARCG baseline relevance logits differ by at most `0.0000567` across 5000 matched candidates. Historical Reader max-length/answer-span limits are `350/10`, matching the current config; model revision/config identity also match. Historical span signals remain cross-run observations, not recorded Q-ARCG per-candidate inputs or gates.
- The 18083 service was reachable. SSH file transfers were repeatedly closed, so the existing trace artifacts were downloaded through authenticated 18083; no dataset, model, or full tensor dump was downloaded.
- GitHub development branch remains at `1751d8b7fec38fc7eb748fc39675c4835e98af29`. Compact training output is still absent from the inspected exchange directory; the branch has not changed since the earlier tree audit.

## Necessary correction to the first audit

The earlier audit correctly calculated exact-set overlap but described all eight overlap losses too strongly as evidence losses. Exact membership is not the same as selecting useful evidence.

The fixed Silver reference contains 222 `direct`, 85 `partial`, 23 `uncertain`, and 170 `irrelevant` labels. In 61/100 questions fewer than five candidates have positive consensus; 13 questions have no positive-consensus candidate in Top-50. The reference nevertheless always contains five members. It attains the count ceiling in every question: total positive slots 307 and direct slots 222. Thus there is no detected ceiling-construction error, but the remaining fixed slots include nonpositive fillers, and membership within those slots is not evidence correctness. Other positive alternatives can also be semantically valid without being in the chosen five.

## Supplementary full-panel metrics

`positive_consensus` counts passages with at least two panel models voting positive (direct or partial). `direct_consensus` counts passages with direct consensus. `all_models_positive` requires all three models positive. These are stored Silver labels, not a new model evaluation or official gold. Metrics below are post-hoc supplements; they do not retroactively replace the preregistered gate.

| Arm | Exact Silver-set overlap | Positive consensus | Direct consensus | All three positive |
|---|---:|---:|---:|---:|
| Frozen Reader | 355 | 203 | 138 | 171 |
| Q-ARCG | 349 | 204 | 138 | 171 |
| Classical control | 353 | 203 | 138 | 171 |

The totals are over 500 selected slots. Q-ARCG positive-consensus delta is `+0.01/question`, bootstrap 95% CI `[-0.02,+0.05]`, cases `2/97/1` better/equal/worse. Direct delta is `0.00/question`, CI `[-0.03,+0.03]`, cases `1/98/1`. All-three-positive counts do not change in any question. Bootstrap uses 2000 question samples and seed 20261006; this is exploratory uncertainty, not a confirmatory test selected independently of the panel.

## Every changed case

| Case | Exact overlap delta | Positive delta | Direct delta | Content-linked observation |
|---:|---:|---:|---:|---|
| 13 | -1 | 0 | 0 | An unrelated novel is replaced by a plot fragment of the requested novel, but neither states the requested location. Both have zero positive votes. |
| 19 | -1 | -1 | 0 | A partial singer/series link is replaced by a band's collaboration with cast members; the latter does not support the theme-performer relation. |
| 24 | +1 | +1 | 0 | A character-roster passage is replaced by an outcome-related plot passage judged partial. This is positive evidence gain, not new direct evidence. |
| 54 | +1 | 0 | +1 | A passage with a brief origin reference is replaced by a direct explanation of the character's historical sources. |
| 55 | 0 | +1 | 0 | General organizational history is replaced by convention-related text that names the adopting body for optional protocols; panel support is partial, not direct support for the original convention's creator. |
| 61 | -1 | 0 | -1 | Explicit participation/voting eligibility is replaced by a broader description of an assembly. Direct evidence becomes partial. |
| 66 | 0 | 0 | 0 | One direct account of the antagonist is replaced by another. Different adaptation provenance is worth noting, but panel direct support stays constant. |
| 69 | -1 | 0 | 0 | One unrelated actor biography is replaced by another; neither establishes the requested character/cast relation. |
| 70 | -1 | 0 | 0 | Results from one other competition are replaced by results from another; neither supports the requested show's winners. |
| 73 | -1 | 0 | 0 | An unrelated song's authorship text is replaced by a different song's remix history. Neither supports the requested song. |
| 79 | 0 | 0 | 0 | Season-level background changes, without support for the character's exact transformation episode. |
| 82 | -1 | 0 | 0 | General/historical venue text is replaced by a particular historical game; both are uncertain for the question's unresolved year. |
| 90 | -1 | 0 | 0 | Two non-supporting basketball passages are replaced by two other non-supporting passages; team and season details still fail to establish the requested last missed tournament year. |

Of the eight exact-membership losses, six (13, 69, 70, 73, 82, 90) leave both positive and direct counts unchanged. The true panel-supported damage is case 19's lost partial support and case 61's direct-to-partial replacement. Case 55's positive gain was invisible to exact-set overlap.

## Signal pattern, not established causal explanation

Across all 14 replacements, removed passages have mean historical span-logit `12.1780`, span-margin `4.1558`, start/end entropies `0.2852/0.2556`; entrants have `7.3318`, `1.0156`, `1.0391/0.7807`. Their mean final-score corrections are `-0.6740` and `-0.3624`, respectively. Only case 73's singleton swap increases the margin; the multi-swap in case 90 also replaces both larger margins with smaller ones.

Across 5000 candidates, correction correlations with baseline relevance, historical span logit, and margin are approximately `-0.764`, `-0.688`, and `-0.541`. These pooled associations are confounded by question and input-feature correlation; they do not establish learned weights, causal gate behavior, or a universal anti-sharpness rule. The head is active but often demotes sharper/higher-ranked evidence more, which can help on case 54 and hurt on case 61. No checkpoint or per-candidate gates exist in the retrieved artifacts to test attribution further.

Content comparison suggests the missing distinction is support for the requested entity/relation/time, not generic span confidence. The implemented four-scalar head receives relevance/span/margin/disagreement statistics, not token-level relation semantics. Weak answer-string labels and incomplete convergence remain competing explanations; training output is needed before choosing between them.

## Next direction and boundaries

Retain the relevance anchor and bounded intervention, but stop treating either exact five-member imitation or unconditional span strengthening as the recovery target. First recover the existing training outputs. The provisional quantum-main direction is a question-conditioned semantic evidence head using Reader hidden/token representations instead of only four scalar signals, with matched classical control and independent training/validation labels. This is `exploratory_only`, not an implemented or novelty-validated idea; no new training, dataset download, circuit expansion, or exposed-panel tuning occurred.

Current hidden tensor traces cover only a small sample, so a missing changed-case activation or gate must remain missing; content analysis cannot manufacture it. A future experiment must prospectively declare useful/direct evidence retention, eligible-question handling, exact-set overlap as a secondary diagnostic, and downstream/independent-data claim boundaries.

## Reproduction and artifact boundary

Offline tool: `scripts/collab/five_ideas/analyze_qarcg_reader_screen_cases.py`. It takes the registered detail JSON, selector trace, and historical Reader trace; validates hashes/order/coverage; emits a compact anonymous summary. Optional `--private-cases` emits raw text outside GitHub.

Compact machine-readable output: [paired audit JSON](qarcg_screen_20261007_paired_case_audit.json). Local-only human/model case material is at `E:/Dir/CodexHome2/tmp/Q-DUET-VLM/qarcg_screen_20261007_private_cases.json`; it contains all 13 changed questions and the 28 added/removed passage observations. Do not commit this raw-text artifact. It is an offline joined audit, not a new end-to-end model case-study run.
