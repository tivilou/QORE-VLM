# Quantum Reader rotation crossed screen contract

- Date: 2026-10-13
- Scope: rag-selector
- Status: implemented / collaborator-only execution pending
- Supersedes: the un-crossed execution placeholder in `2026-10-13-quantum-reader-feature-rotation.md`
- Authorization: the user authorized starting implementation; real Wiki-DPR fitting remains delegated to the collaborator.

## Decision

Run a fixed 100-case Top-50 to Top-5 screen with the DPR Reader frozen. The
screen uses the immutable 59-question/228-pair support bundle for training and
the registered 100-case detail artifact for evaluation. Retrieval, Generator,
evaluator, candidate order, and all evaluation labels stay outside the online
forward, loss input, and selection path.

The crossed arms are:

1. frozen Reader Top-k;
2. the previous semantic head with the previous boundary loss;
3. the previous semantic head with the new Top-5 retention loss;
4. the new quantum feature rotation with the previous loss;
5. the new quantum feature rotation with the new loss (primary candidate); and
6. a parameter-matched classical coordinate rotation with the new loss.

Every fitted arm gets a fresh head, identical question order, seed, three
epochs, optimizer budget, gradient cap, and complete 50-candidate scoring. The
new head is a zero-initialized, question-conditioned real-unitary rotation of
the Reader CLS representation before the original relevance classifier. It
uses 10 data qubits, two layers and 200 trainable angles. The classical arm
uses the same angle count and conditioning projection with seeded coordinate
matchings. Both are exact classical simulations; no quantum advantage claim is
permitted.

## Prospective gates

- Primary utility: the primary candidate's per-question positive-consensus
  count has paired mean greater than zero and a bootstrap 95% lower bound above
  zero versus frozen Reader Top-k.
- Protection: its direct-consensus paired mean is nonnegative versus frozen
  Reader Top-k.
- Attribution: its positive-consensus paired mean and bootstrap lower bound are
  positive versus the matched classical rotation.
- Kill the candidate if any Reader score/span/null/trace integrity check fails,
  direct support declines, or the utility gate fails. Do not rescue a failure by
  widening residual bounds, adding epochs, tuning Silver labels, or switching
  to a different panel.

## Artifacts and handoff

- Config: `configs/experiments/quantum_reader_rotation_screen_100.json`
- Plugin plan: `configs/experiments/quantum_reader_rotation_screen_100_plan.json`
- Runner: `scripts/collab/five_ideas/run_quantum_reader_rotation_screen_100.py`
- One-command wrapper: `scripts/collab/five_ideas/run_quantum_reader_rotation_screen_100.sh`
- Compact GitHub artifacts: `summary.json`, `report.md`, `run_metadata.json`.
- Exchange-only artifacts: `selector_trace.json`, `rotation_values.npz`, and
  `training_targets.private.json`, all under
  `five_ideas/quantum_reader_rotation_screen_100/<timestamp>/`.

The collaborator runs the wrapper from the project root after configuring only
their local `QORE_EXCHANGE_TOKEN`. The wrapper discovers local Python and model
cache; it does not encode this workspace or server paths. No local Wiki-DPR
download or real fitting was performed while implementing this contract.

## Evidence ceiling

Until the collaborator result is uploaded and independently reopened, this is
an implemented L0 diagnostic design. A passing 100-case screen would support a
bounded candidate result, not L1/L2, full-data utility, or quantum advantage.
