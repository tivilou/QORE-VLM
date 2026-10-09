# Boundary review completed: qualify targets and preserve valid Reader selections

- Date: 2026-10-09
- Scope: rag-selector
- Status: active
- Supersedes: None; extends the partial supervision-result audit
- Superseded By: None
- Authorization: user started existing-material support adjudication and target design; no new fitting or corpus/model inference on development servers
- Gate: full32x50 target coverage/uncertainty and training protocol qualification before new quantum training; all434 weak masks remain uncertified

## Findings and durable target rule

All220 distinct items in frozen boundary/control/all-method Top5 union reviewed;209 new +73 preserved judgments =282/1600. Single reviewer, current scores withheld but not independent blinded review. Original source remains unchanged. Direct/partial Reader and Q/C semantic50/54;scalar50/53 plus one additional uncertainty, not certified harm/gain. Diagnostic population is seen training data, not independent validation.

Scope weak positives include19 irrelevant,2 contradictory; weak negatives include9 direct (some question-reference conflicts, not all certifiable targets). Registered60 dependent pairs contain only5 clean direct-vs-irrelevant with matched irrelevant control across3 questions. Q crosses0, and favors true over control on1; monotonic control can favor2 without changing selection. Nonselective compression remains unresolved; this is not family closure or utility proof.

Future supervision candidate must keep confirmed direct Reader selections, avoid forced direct-vs-direct comparisons, separate partial/uncertain/missing, isolate question-reference/temporal/version/formula corruption, and compare support-qualified quantum semantic boundary learning with same-budget classical, scalar and compression controls. Never auto-correct reference answers or upgrade single-model labels to gold.

## Next and stop

Finish remaining existing32x50 review and target qualification before declaring supervision ready. Define confidence/missing masks and unit-test target behavior with synthetic cases, then preregister collaborator-local full434 qualification/fresh347/87 training. Current diagnostic subset is not a complete supervision cohort. Exposed100 panel stays evaluation-only. No blind residual/depth/epoch escalation.

Stop if qualifying support remains inadequate or corrected signals only compress scores; promotion requires real fresh validation gains with positive retention and matched controls, not substring counts. Baseline/dataset qualification remains separate and incomplete.

## Links

- [Scope](../refs/semantic_supervision_20261009T045728Z_boundary_scope.json)
- [Labels](../refs/semantic_supervision_20261009T045728Z_boundary_review.json)
- [All pairs and selected counts](../refs/semantic_supervision_20261009T045728Z_boundary_join.json)
- [Chinese owner log](../../../../docs/rag-research-log/20261009T-boundary-support-review.md)
