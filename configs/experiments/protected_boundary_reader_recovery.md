# Protected boundary supervision recovery

Candidate: `protected-boundary-reader-v1`. Implementation baseline: `b922182`
locally; canonical server `371d043`. The Reader semantic head and four-qubit
ansatz, exact null initialization, residual bound0.25, same-budget classical
control and fan-in-scaled AdamW remain fixed. This tests a new **training
signal**, not a new quantum circuit or demonstrated quantum advantage.

Failure evidence: weak answer containment labels included topic-only passages,
missed direct support and confused partial evidence with negatives. Optimizer
repair restored gradients but did not establish useful discrimination. Final
body adjudication closes the audit:59 training questions228 qualified
direct-versus-selected-irrelevant pairs,75 possible slots. Only60 pairs meet
the loose two-sided residual reachability bound; this does not prove rescue.

Mechanism: question-normalized softplus pair loss plus0.01 anchor penalty and
1.0 penalty for eligible positives outranking immutable selected slots. All
selected slots except explicitly qualified irrelevant negatives are immutable
**in the training loss**; online Top5 remains ordinary score ranking. The
penalty discourages damage but cannot guarantee protection from other
candidates. No label-aware mask may enter evaluation selection.

Discriminating measurements: task-gradient flow and within-question correction
spread; pair-gap changes and actual inversions; actual unmasked qualified
direct rescues, protected-direct/immutable-slot losses; positive correction
versus nearest-base-score primary-opinion irrelevant controls (with reuse,
not independently certified). Report all59 including difficult unreachable
pairs, without increasing the residual or excluding failures after fitting.

Minimum screen: train fresh quantum and same-budget classical semantic heads
on identical frozen59 in source order for3 epochs, seed20261008, learning
rate0.003, projection-only decay0.0001, dimension-scaled projection LR,
clip1.0. Final epoch only. No early stopping/checkpoint search on exposed100
Silver.87 producer-null validation rows stay unused.

Gate: all100 questions included; positive-consensus paired mean>0 and question
bootstrap CI95 lower>0 vs frozen Reader, direct-consensus paired mean>=0.
Quantum attribution is a separate positive/CI-lower>0 comparison vs matched
classical. Failure stops advancement, not all future quantum exploration.
Stop before panel inference if gradient/variation collapses or exact source,
score-order parity, numeric replay, model revision or required upload fails.

Preflight: two synthetic training and two synthetic evaluation questions run
through the actual pinned DPR Reader. Both heads receive same inputs, have
two synthetic updates and emit full real hidden/pooling/scores plus detached
gradient/optimizer evidence. Reopen values and schema-v2 lineage before real
fitting. This is engineering verification, never task-effectiveness evidence.

Targets are research judgments, not official gold; inherited primary exposure,
same-parent A/B and redacted Claude identity remain explicit.100-panel
exposure prevents L1/L2 claims. No dataset download, retrieval or Generator
call; real experiments belong to collaborator hardware.
