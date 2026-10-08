# Semantic Reader screen: diagnose and repair before another task run

- Date: 2026-10-08
- Status: active evidence-repair gate; semantic candidate blocked, broader family inconclusive
- Scope: rag-selector
- Source: `five_ideas/quantum_semantic_reader_screen_100/20261008T101111Z/`
- Authorization: inspect the collaborator result, verify and record analysis. No new training-path implementation or collaborator screen is authorized by this result-report turn.

## Verified result

All 100 cases and fixed candidate identities join the registered input. All file receipts, byte counts, input/config/plan and source hashes match. Saved hidden/pooling, initial and epoch/witness parameters, every evaluation score and JSON Top-5 replay successfully. Usable training coverage is 434/512 (84.77%), three epochs; 76 no-hit and two all-positive questions were excluded from weak training. No usable training question hash intersects evaluation.

Reader, quantum semantic and classical semantic all retain positive/direct/unanimous/membership totals `203/138/171/355`. Both semantic heads have exactly the Reader score vectors on all 100 cases and zero residual on all 5,000 candidates. Scalar quantum changes 16 sets, retains `203/138/170/348`; gains and losses cancel for positive/direct evidence. The declared selection and attribution gates fail. Neither scalar nor semantic arm improves this panel.

## Failure interpretation

The semantic projection's L2 norm falls from `1.1588575` to about `1.005e-10`, `1.336e-20`, `1.697e-30` across three quantum epochs. Classical projection collapses similarly. Float64 norm accumulation is required because float32 norm would prematurely report zero. Both residual scale/readout magnitudes approach zero, encodings cease differentiating candidates, and sampled task gradients are near zero. At exact-null initialization, only the residual scale receives a task gradient; projection/readout are blocked by the zero scale, while Adam's coupled decay still updates them. A zero-task-gradient, label-free Adam arithmetic diagnostic nearly reproduces epoch projection shrinkage and gate-bias movement.

This is strong evidence for a learning-path/regularization interaction, not proof from a full optimizer replay: all-step gradients and Adam states were not saved. Weak labels, weak early gradients and null-compatible learning still require qualification. The primary implementer owns the configuration and missing training-health tests; this is not a collaborator execution mistake. Finite/replayable outputs are necessary but insufficient evidence that learning occurred.

## Source and compatibility fix

The GitHub submission `44c9f70` contains a tokenizer-mask fix, not the compact result files at audit start. The producer reports HEAD `5ce58ba` but its semantic source hash equals the later `44c9f70` file exactly; runner/base-head/config/plan hashes also match. This recovers the executed files, not an assertion that the run started from a clean committed checkout. All four saved hidden samples contain a real SEP boundary, so the mask fix's midpoint fallback is not implicated for those samples. Do not approve that fallback for future malformed inputs merely because this run replayed.

## One recommended next step

Repair/qualify training before generating another idea or spending a larger task budget. First reuse the two saved train-only cases to compare decoupled versus coupled regularization and a viable exact-null learning path, separately checking projection/readout groups. Add assertions for preserved input-dependent encodings, nonzero data gradients after leaving null, finite values and effective float32 score corrections. A proposed correction is not a measured utility improvement. Keep quantum/semantic classical controls identical and retain relevance anchor and weak-label/data boundaries.

Only after that qualification should the owner approve a separately registered rerun of the same fixed 100-question comparison. Do not increase epochs, data scope, circuit depth or Silver-tuned coefficients to conceal this defect. If the qualified head then changes rankings without improving evidence, revisit supervision quality and error regimes using cumulative historical cases. This turn implements only offline audit tooling, not the training repair.

## Evidence boundary

L0 diagnostic; exposed Silver panel, one short run, no Generator/evaluator or downstream utility result. Stop this unchanged configuration, not RAG exploration or the broader quantum/semantic family. The recovery dossier deliberately uses `disposition=stop` for new method selection pending evidence repair; it is not a project-goal stop.

- Audit: `../refs/quantum_semantic_reader_20261008T101111Z_audit.json`
- Recovery: `../refs/semantic_reader_training_collapse_recovery_20261008.json`
- Evidence card: `../refs/quantum_semantic_reader_20261008_evidence_card.json`
- Owner log: `../../../../docs/rag-research-log/20261008T-101111Z-quantum-semantic-reader-training-collapse.md`
