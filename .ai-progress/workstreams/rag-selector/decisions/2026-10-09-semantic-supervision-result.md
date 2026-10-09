# Supervision audit result: qualify support before quantum boundary training

- Date: 2026-10-09
- Scope: rag-selector
- Status: active
- Supersedes: None; records the result of the train-only audit
- Superseded By: None
- Authorization: result inspection and narrow evidence publication only; no new fitting or real-data inference on development servers
- Gate: finish declared support-review coverage with confidence/uncertainty; qualify corrected targets before preregistering a new fresh split training experiment

## Evidence and consequence

Source `five_ideas/semantic_reader_supervision_audit/20261009T045728Z`, producer revision `31a5c932de9b525cdf34eed688e9890226402cb9`. Nine receipts/hashes, source identity and11 full detached replay checks pass with an explicitly bounded1e-9 scalar-stat comparator; historical strict dict-equality validator fails at roundoff <=3.55e-15 and is unchanged.

Reader/Q-semantic/C-semantic/Q-scalar/monotonic weak selected totals87/87/87/89/87. Both semantic heads retain Reader Top-5 on all32 training cases. Scalar changes4 cases; two +1 substring changes are not certified support gains. These are not independent validation or support accuracy.

Single-model partial review:64 shuffled score-blind passages plus9 exploratory supplements,73/1600. Source review remains allnull. Concrete containment/support conflicts cover wrong entity, question-reference type, event/date and order constraints. Noise prevalence and its causal training effect are unproven. L0 diagnostic; do not close the quantum family or claim utility/advantage.

## Next candidate and stop gate

Continue support adjudication on existing cases, including registered boundary pairs, matched controls and all swaps; retain uncertain labels and original references. Then consider support-qualified, confidence-aware quantum semantic boundary/listwise supervision with Reader anchor and matched classical/scalar/compression controls. Future347/87 is only for freshly initialized training; never fit exposed100 Silver panel. No new training rerun, residual amplification, epochs/depth escalation before this gate. A partial review does not qualify a complete supervised cohort.

Stop progression if targets remain unreliable or selective correction is indistinguishable from compression; future fresh validation must improve certified evidence without destroying retained positives. Compare relative historical mechanisms as well as baseline gaps. Baseline/source qualification and historical raw-text parity limits remain explicit.

## Evidence links

- [Independent audit](../refs/semantic_supervision_20261009T045728Z_audit.json)
- [Partial support judgments](../refs/semantic_supervision_20261009T045728Z_partial_support_review.json)
- [Verification/replay](../refs/semantic_supervision_20261009T045728Z_verification.md)
- [Owner log](../../../../docs/rag-research-log/20261009T-045728Z-semantic-supervision-audit-result.md)
