# Critical Invariants: RAG Passage Selection with QORE
## Role and reading order
- Read this file and `state.md` first; follow indexed decisions. Keep detail in linked records.
## Non-negotiable principles
- After each audit, combine cumulative evidence; record the next idea, gate, kill condition and authorization before running.
- Diagnostic/oracle/gold-conditioned/short-screen results stay below L1/L2; separate utility, robustness, attribution and claim ceiling.
- Every experiment analysis gets an evidence-linked log and updates state/decision when the durable next action changes.
- Scope is fixed Top-50 -> Top-5; retrieval, production selector, Generator, evaluator, labels, order, and answer path stay frozen unless a binding decision changes it.
- Plugins are isolated, allowlisted, deterministic where declared, and compared with disabled/original behavior.
- Finite gradients, exact-null and artifact replay do not prove training health. Before a head-based task rerun, check data-gradient flow after null initialization, parameter/feature variation and effective score corrections; diagnose decay-driven collapse before attributing a null result to representation capacity.
- A nonzero uniform score shift is not candidate-dependent learning. Check within-case correction spread and wide-projection saturation; isolate optimizer/learning-path repair from claims of task utility. See [repair gate](decisions/2026-10-08-semantic-reader-optimizer-repair.md).
- Favorable correction of low-ranked positives is not support discrimination if wrong passages receive the same effect. Compare matched nonpositive controls and bounded-score reachability; healthy gradients/one rescue do not justify blind residual amplification. See [result](decisions/2026-10-09-semantic-reader-repair-result.md).
- Gold, evaluator/generation output, and panel labels are post-hoc diagnostics only; three-model labels are Silver L0.
- Distinguish exact Silver-set overlap from positive/direct evidence retention: fixed-size references can contain fillers or omit valid alternatives. New metrics must be declared prospectively; supplementary diagnostics do not rewrite old gates. See [detail](decisions/2026-10-08-silver-membership-vs-evidence-retention.md).
- Generator-native work uses one transport, exact mapping and zero-anchor fallback without retry/feedback.
- Use exact source/question/passage identity; fuzzy/manual/answer-string substitutions cannot manufacture coverage.
- Record dataset/model/revision/config hash/seed/split/code revision and compact provenance per artifact.
- GitHub accepts compact `result.json` only when <=`1,048,576` bytes; larger/raw results use authenticated exchange with manifest/hash/path.
- Collaborator commands are project-relative; wrappers discover local dependencies. Develop on the verified server; collaborators run locally.
## Durable cumulative knowledge
- Answer Scorer over QORE-DPR is the only stable positive carryover; its calibration is unsafe as the sole evidence criterion.
- Retrieval-depth/reranking, context routing, reader energy/transport, topology, ASAC, answer expansion, and parameter-only QORE/MMR/DPP rescues are closed.
- The 100-question panel and frozen-Generator oracle are L0: Silver Top-5 membership is not utility (oracle F1 uncertain, EM fell); neither establishes population utility or authorizes a selector.
- The 100-case fixed-Top-50 replay did not beat Top-k on Silver alignment: Top-k `3.55/5`, Support Graph `2.61`, Anchor--Residual `2.23`, Q-DES `0.88`; no Generator screen follows.
- Official gold alignment is blocked by source/version identity and usable-context coverage; it cannot classify the bottleneck.
- Q-DES preflight is L0/closed; typed signal lost to Answer Scorer.
- Anchor--Residual preflight is L0/closed; broad/direct `85/61` vs Top-k `106/70`.
- Support Graph preflight was an L0 candidate pass (`113/79` vs `106/70`), but only a separately gated screen was authorized; that screen is now closed.
- ECAD preflight was L0/blocked; the current candidate replay supplies provenance-locked pairwise NLI as an online signal, while all Silver labels remain post-hoc and the replay is still L0.
- TREG uses fixed DPR Reader residual gain `Reader(q,B4+p)-Reader(q,B4)`; ECAD uses 1,225 rank-ordered pair scores per case. Both receive an evidence-free view and must pass identity/order/probability validation.
- Rich portal snapshots are local-only until exact hash/source registration; never replace GitHub compact artifacts.
- External model consultation is advisory only: keep benchmark variant, exact API model ID, transport status, model echo and benchmark equivalence separate; never silently substitute an unavailable model. A quantum proposal must pass a mechanism audit and matched classical/ablation controls before implementation.
- Q-GSRR is closed after its feature-only gate: fixed rank-5 versus rank-6--10
  full/title/local-window audit found `2/100` canonical inversion cases, below
  the `15/100` minimum. Do not revive it by widening the band or tuning the
  token/sentence threshold against this artifact.
- Q-DUET-RAG is closed after its exact 100-case replay: Silver overlap `2.53/5`
  versus Top-k `3.55/5`, paired `3/36/61`. Its two-stage dominant/residual
  QUBO structure did not improve fixed Top-50 -> Top-5 alignment; do not move
  to QAOA, Generator screening, or Silver-driven parameter tuning.
- The classical scorer benchmark is a separate selector-only screen: compare
  each allowlisted non-generative scorer's direct Top-5 and matched QORE arm
  before selecting any scorer for quantum optimization. Silver/evidence remain
  post-hoc L0 diagnostics and cannot tune scorer or selector parameters.
- DPR Reader tensor tracing is observation-only: run the frozen Reader on the
  registered fixed 100 x 50 input, save compact signals for every candidate and
  a small full hidden/attention sample, then decide the quantum correction from
  the observed error pattern. Labels stay post-hoc and no selector, Generator,
  evaluator, or training call is allowed in this trace.
- The fixed DPR span/relevance screen is closed after the 100-case replay:
  Answer Scorer Top-k `3.55/5`, QORE `2.59/5`, classical fixed fusion `2.70/5`,
  and fixed quantum interaction `2.20/5` Silver overlap. The quantum arm and
  exact classical Born control select the same passages; the gate failed. Do
  not advance this fixed post-Reader map. A future candidate must change the
  mechanism, use a trainable Reader-integrated hybrid with a matched classical
  parameter-budget control, and pass an independent-data gate.
- Q-ARCG is the current exploratory Reader-integrated candidate. It keeps the
  DPR relevance logit as an exact null anchor and adds a four-qubit trainable
  applicability gate plus bounded residual from Reader-side span/support and
  disagreement features. Its same-budget classical control, exact
  zero-residual initialization, and Torch differentiability preflight passed;
  this is L0 engineering evidence only. Do not claim utility, L1/L2, or
  quantum advantage until a separately authorized real-data train/validation
  screen beats frozen Reader Top-k and the matched classical control.
- The Q-ARCG short-screen trace gives Reader/Q-ARCG/classical Silver overlap
  `3.55/3.49/3.53`; Q-ARCG changes 13 cases, with `2/90/8` better/equal/worse.
  This configuration is not promoted. The broader trained mechanism remains
  inconclusive pending compact training/provenance artifacts; do not tune the
  exposed Silver panel or treat this short run as full-budget falsification.
- Q-ARCG's full-panel posthoc audit gives positive consensus `204/203` versus
  Reader, direct `138/138`, and all-three-positive `171/171`: no reliable gain.
  Six of eight membership losses leave positive/direct counts unchanged. Keep
  the relevance anchor; the provisional next hypothesis changes the head's
  information input to question-conditioned semantic representations, not span
  coefficients. Previous training provenance remains missing. The subsequent
  user-approved semantic-head exploratory screen does not resolve that gap.
- The quantum semantic Reader screen is implemented for collaborator execution:
  frozen final Reader states -> question-conditioned pooling -> trainable
  classical projection -> four-qubit gated residual. Compare Reader, semantic
  quantum, matched semantic classical and scalar quantum on the same weak
  training cohort. Positive-consensus retention is primary, direct retention
  is protected, and exact Silver membership is secondary. This is an exposed-
  panel L0 screen, not end-to-end Reader fine-tuning or quantum advantage.
  Reopened hidden/pooling/head/checkpoint/selection replay is mandatory; raw
  NPZ/trace stay exchange-only and compact training reports are mirrored there.
- The semantic screen `20261008T101111Z` is blocked for training repair: Reader,
  semantic quantum and semantic classical totals are all positive/direct/member
  `203/138/355`; both semantic heads produce zero correction for all 5000
  candidates after projection/readout collapse. A coupled-Adam decay-only
  diagnostic nearly reproduces shrinkage. This is not an effective learned
  intervention test or closure of the quantum/semantic family. Stop unchanged
  reruns; qualify the learning path before spending more data/epochs. Scalar
  quantum changes 16 sets but has no net positive/direct gain (`203/138/348`).
- The approved Q-ARCG screen trains only the head on up to 512 `nq_open/train`
  answer-string-containment weak labels; this is not official passage gold.
  The fixed 100 x 50 detail artifact is evaluation-only, Silver is posthoc,
  Generator/evaluator are not called, and `selector_trace.json` is exchange-
  only. The compact result cannot carry raw question/text/answer/selected-ID
  fields. See [detail](decisions/2026-10-05-qarcg-reader-screen-implementation.md).
## Decision index
### Binding and current
- [BINDING] Cumulative-evidence next-step rule -> [detail](../../shared/decisions/cumulative-evidence-next-step-selection.md)
- [BINDING] Compact result publication/privacy threshold -> [detail](../../shared/decisions/2026-08-27-result-json-publication-threshold.md)
- [CLOSED] Selector-only scope and QORE-QES candidate -> [detail](decisions/2026-09-05-selector-only-qore-qes.md)
- [BLOCKED] Official NQ-test positive-context coverage/gold alignment -> [detail](decisions/2026-08-30-dpr-positive-context-coverage-block.md)
### Historical data and retrieval
- [CLOSED] Phase 9B retrieval ceiling; superseded by Phase 9D -> [detail](decisions/2026-08-18-phase9b-retrieval-ceiling.md)
- [CLOSED] Phase 9D retrieval-depth hypothesis -> [detail](decisions/2026-08-19-phase9d-clean-next-phase9e.md)
- [CLOSED/INVALID IDENTITY] DPR v2 zero-join preflight -> [detail](decisions/2026-08-29-dpr-v2-question-join-preflight.md)
- [CLOSED/INSUFFICIENT COVERAGE] Gold-label source repair gate -> [detail](decisions/2026-08-29-gold-label-source-repair-before-bottleneck.md)
- [SUPERSEDED] Official DPR NQ-test gold-info source -> [detail](decisions/2026-08-30-dpr-nq-test-gold-info-source.md)
### Historical context, selector, and Generator
- [SUPERSEDED/KILLED] Adaptive context-budget candidate -> [detail](decisions/2026-08-26-cumulative-next-step-adaptive-context.md)
- [CLOSED] Phase 10A context-budget routing -> [detail](decisions/2026-08-26-phase10a-result-stop-context-route.md)
- [SUPERSEDED] Phase 10B answer-hypothesis bridge -> [detail](decisions/2026-08-26-phase10b-answer-hypothesis-bridge.md)
- [CLOSED] Phase 10C reader-span energy -> [detail](decisions/2026-08-26-phase10c-reader-span-energy-kill.md)
- [SUPERSEDED] Phase 10D topology preregistration -> [detail](decisions/2026-08-26-phase10d-minimal-set-support-topology-preregistration.md)
- [CLOSED] Phase 10D topology screen -> [detail](decisions/2026-08-26-phase10d-topology-screen-kill.md)
- [CLOSED] ASAC novelty collision -> [detail](decisions/2026-08-27-asac-novelty-collision-stop.md)
- [SUPERSEDED] Generator-adaptation resume decision -> [detail](decisions/2026-08-28-rag-exploration-resume-evidence-to-answer-adaptation.md)
- [SUPERSEDED FOR SELECTOR SCOPE] Stop online RAG interventions -> [detail](decisions/2026-08-28-cumulative-rag-online-stop.md)
### Historical Generator-boundary chain
- [SUPERSEDED] Generator-boundary preflight gate -> [detail](decisions/2026-08-27-generator-boundary-preflight-gate.md)
- [SUPERSEDED] Synthetic boundary preflight pass -> [detail](decisions/2026-08-27-generator-boundary-synthetic-preflight-pass.md)
- [SUPERSEDED] Generator-native interface design -> [detail](decisions/2026-08-27-generator-native-interface-design.md)
- [SUPERSEDED] Generator-native boundary implementation -> [detail](decisions/2026-08-27-generator-native-boundary-implementation.md)
- [SUPERSEDED] Actual-model hook fixture -> [detail](decisions/2026-08-27-generator-native-hook-fixture-pass.md)
- [SUPERSEDED] Generator-native screen authorization -> [detail](decisions/2026-08-27-generator-native-screen-authorization.md)
- [CLOSED] Reader-localized Generator transport screen -> [detail](decisions/2026-08-28-generator-native-screen-kill.md)
- [ACTIVE] Project execution topology and collaborator handoff -> [detail](../../shared/decisions/2026-09-05-project-execution-topology.md)
- [SUPERSEDED] Portfolio priority -> [detail](decisions/2026-09-05-selector-portfolio-priority.md)
- [SUPERSEDED] Q-DES P0/ECAD next gate -> [detail](decisions/2026-09-05-qdes-p0-preflight-failure.md)
- [BLOCKED] ECAD conflict preflight -> [detail](decisions/2026-09-05-ecad-p0-preflight-failure.md)
- [BLOCKED] ECAD label provenance -> [detail](decisions/2026-09-05-ecad-label-provenance.md)
- [CLOSED/PREFLIGHT FAIL] Anchor--Residual -> [detail](decisions/2026-09-08-anchor-residual-preflight-failure.md)
- [CANDIDATE/PREFLIGHT PASS] Anchor-Conditioned Support Graph -> [detail](decisions/2026-09-08-support-graph-preflight-pass.md)
- [CLOSED] Support Graph fresh-50 screen implementation -> [detail](decisions/2026-09-08-support-graph-fresh-screen-implementation.md)
- [CLOSED] Silver-oracle Top-5 frozen-Generator ceiling -> [detail](decisions/2026-09-12-silver-oracle-top5-generator-ceiling.md)
- [CLOSED] 100-case selector replay portfolio -> [detail](decisions/2026-09-17-selector-replay-100-closure.md)
- [CLOSED] TREG and ECAD fixed-Top-50 selector replay -> [detail](decisions/2026-09-17-treg-ecad-selector-replay-100.md)
- [CANDIDATE] Retain Granularity-Stability Rank Rescue for a bounded preflight only -> [detail](decisions/2026-09-20-2026-09-20-ssgr-preflight-candidate.md)
- [CANDIDATE/L0 DESIGN ONLY] External quantum-selector consultation and F50-QWI gate -> [detail](decisions/2026-09-20-external-quantum-selector-consultation.md)
- [SUPERSEDED] Implement F50-QWI fixed-unitary selector audit -> [detail](decisions/2026-09-20-f50-qwi-implementation.md)
- [CLOSED] F50-QWI fixed-unitary selector replay -> [detail](decisions/2026-09-27-f50-qwi-replay-closure.md)
- [CANDIDATE / NOT AUTHORIZED] Quantum Granularity-Stability Gated Rank Rescue -> [detail](decisions/2026-09-27-qgsrr-candidate-recovery.md)
- [CLOSED/PREFLIGHT FAIL] Q-GSRR feature-only audit -> [detail](decisions/2026-09-27-qgsrr-feature-audit-failure.md)
- [CLOSED] DUET-inspired two-stage quantum-ready selector screen -> [detail](decisions/2026-09-27-q-duet-rag-screen.md)
- [CLOSED] Classical non-generative scorer benchmark on fixed Top-50 -> [detail](decisions/2026-09-28-classical-scorer-benchmark.md)
- [CLOSED] DPR Reader Span-Relevance Fusion Screen -> [detail](decisions/2026-10-01-dpr-reader-span-relevance-screen.md)
- [ACTIVE CANDIDATE / PREFLIGHT PASSED] Q-ARCG Reader-integrated residual -> [detail](decisions/2026-10-05-qarcg-reader-integrated-residual.md)
- [ACTIVE CANDIDATE / SCREEN IMPLEMENTED] Q-ARCG Reader 100-case screen -> [detail](decisions/2026-10-05-qarcg-reader-screen-implementation.md)
- [ACTIVE / SCREEN GATE FAILED, PROVENANCE PENDING] Q-ARCG result follow-up -> [detail](decisions/2026-10-08-qarcg-screen-trace-gate.md)
- [ACTIVE] Silver membership versus evidence retention -> [detail](decisions/2026-10-08-silver-membership-vs-evidence-retention.md)
- [BLOCKED / TRAINING REPAIR] Quantum semantic Reader head design -> [detail](decisions/2026-10-08-quantum-semantic-reader-screen.md)
- [ACTIVE / EVIDENCE REPAIR] Semantic training collapse and next gate -> [detail](decisions/2026-10-08-semantic-reader-training-collapse.md)
- [IMPLEMENTED / TRAIN-ONLY QUALIFIED] Semantic optimizer repair and controlled rerun -> [detail](decisions/2026-10-08-semantic-reader-optimizer-repair.md)
- [ACTIVE / SCREEN INCONCLUSIVE] Repaired semantic head: compression, one rescue, supervision gate -> [detail](decisions/2026-10-09-semantic-reader-repair-result.md)

- [IMPLEMENTED / EXPORT AUDITED; SUPPORT REVIEW PARTIAL] Train-only support/blinded boundary audit -> [detail](decisions/2026-10-09-train-only-supervision-audit.md)

- [ACTIVE / L0; 73/1600 PARTIAL REVIEW] Training containment/support conflicts and gated quantum boundary candidate -> [detail](decisions/2026-10-09-semantic-supervision-result.md)

- [ACTIVE / 1600 AUDIT ITEMS REVIEWED; FULL434 QUALIFICATION PENDING] Confidence-qualified support and protected quantum boundary target -> [detail](decisions/2026-10-09-support-qualified-boundary-review.md)

- [IMPLEMENTED / OFFLINE ONLY; REAL QUALIFICATION PENDING] Protected support boundary compiler -> [detail](decisions/2026-10-09-boundary-target-compiler.md)

## Information map and maintenance
- Current goal/status/next action/blocker: `state.md`.
- Rationale: `decisions/`; outcomes: `sessions/`; evidence: `refs/`; owner logs: `docs/rag-research-log/`.
- Update this index on decision lifecycle changes and update it plus `state.md` when policy changes.
- Code/tests are truth; never copy secrets, raw responses, transient logs or full results here.
