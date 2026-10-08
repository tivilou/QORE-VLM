# Exploratory quantum semantic Reader screen

- Date: 2026-10-08
- Scope: rag-selector
- Status: blocked for training repair after 20261008T101111Z screen; broad semantic/quantum family inconclusive
- Supersedes: None
- Superseded By: None
- Gate: synthetic tensor/replay/interface contracts pass before collaborator screen; positive/direct evidence metrics prospectively frozen
- Authorization: user's "开始尝试" authorizes implementation and collaborator exploratory screen; not formal/full-data execution

## Design

Keep frozen DPR Reader relevance as the exact null anchor. Read its final joint question/passage encoder state. Pool CLS, question mean, question-conditioned passage attention, elementwise question/passage product and absolute difference. A trained classical projection maps these 5H features to four bounded inputs of the existing trainable four-qubit depth-two circuit. Local-Z and ring-ZZ readout controls a bounded gated residual. Reader body, retrieval of evaluation candidates, production selector, Generator and evaluator remain fixed.

Semantic quantum and semantic classical arms have identical projection initialization, trainable parameter count, input, seed, case order, optimizer and loss. A newly trained scalar quantum arm on the same cohort distinguishes richer representation from changing supervision or budget. For H=768 the semantic arms each have 15,399 trainable parameters, mostly in the classical projection; only 16 circuit parameters and 8 CNOTs. Exact classical simulation is allowed; quantum parameter/resource advantage is unestablished.

## Exploratory minimum experiment

Use the existing 512-question NQ-Open train weak-containment budget, three epochs, disjoint from the registered fixed 100x50 development panel. Keep this known noisy target unchanged to isolate representation, not to present it as official support labels. Training records from the previous screen remain missing, so this experiment does not prove scalar compression caused its null result. No dataset is downloaded or task experiment run on our development server.

Primary: mean positive-consensus evidence count over all 100 questions, including zero-positive cases. Protection: direct-consensus count. Secondary: all-three-positive and exact Silver-set intersection. Silver fields are opened only after every arm ranks every case. A diagnostic pass requires positive primary paired mean and bootstrap lower bound >0 against Reader, with nonnegative mean direct delta. Quantum-specific diagnostic attribution requires a separate positive paired interval versus the matched classical semantic arm. A formal independent-data utility/replication design is not authorized; no L1/L2 promotion follows this exposed-panel screen.

## Trace and publication

Save scores, encoded features, observables, gate and residual for all 100x50 candidates; complete pooled representations for all evaluation cases; complete hidden/token/mask/pooling-weight slice for the first retrieved candidate in the first two usable training and first two evaluation cases. Save all 50 pooled/base/scalar/weak-mask values for those two training cases, initial/epoch checkpoints and selected-sample preupdate checkpoints. Reopen detached NPZ, replay every evaluation output and training witness, and verify hidden-to-pool slices, finite/bounded values and exact null.

Use the one-command wrapper; automatically create timestamped 18083 directory, upload declared raw values/trace and mirror compact summary/report/metadata there. Compact artifacts also go through GitHub; raw data stays exchange-only. File/config/code/source hashes and actual training counts/losses are required. Old report absence is not silently repeated.

## Stop and rollback

Fail on split overlap, label leakage, nonfinite values, incomplete trace/replay, null drift, parameter-budget mismatch or upload receipt mismatch. A normal short-screen null is inconclusive and not promoted. Disable/remove only the new allowlisted runner/head to restore untouched production behavior; do not overwrite prior results or checkpoints. Do not tune against these 100 questions or expand circuit depth as an unregistered rescue.

## Deferred

Formal dataset qualification, literature novelty, complete natural-data run and independent confirmation remain pending. Quantum resource advantages, finite-shot execution and full Reader fine-tuning are not claimed. A discovery attempt through the Web tool failed; no literature-based novelty assertion is made.

## Verified implementation

Local combined regression: 28/28; primary server core/screen/semantic regression: 21/21. Actual randomly initialized Transformers DPRReader API/hook tests pass without downloading weights. Project v2 structural/semantic/value/checkpoint/selection replay passes on both environments; canonical four-plugin plan and recovery dossier pass. The old shared v1 trace validator is not claimed to validate this project v2 trace. Semantic encoding stays inside (-1,1) to avoid acos boundary gradients. Raw NPZ/trace are ignored under the new exchange namespace. Production code is untouched; rollback is to stop invoking the isolated runner, preserving outputs.

## Result addendum

The collaborator screen ran and replayed but both semantic heads collapsed to zero residual and exact Reader scores. Declared gates fail; no semantic or quantum utility gain is established. Preflight did not qualify real-distribution training health. Do not rerun this unchanged configuration. Follow [the training-repair gate](2026-10-08-semantic-reader-training-collapse.md); this addendum does not retrospectively change the config or success criteria.
