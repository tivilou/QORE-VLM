# F50-QWI design dossier

- Candidate: `f50_qwi`
- Scope: fixed Wiki-DPR Top-50 to Top-5 selector boundary only.
- Evidence tier: L0 diagnostic; Silver overlap is a post-hoc alignment signal.
- Authorization: implement the CPU mechanism audit and 100-case replay only after the Q1-Q9 design grill confirmation.

## Hypothesis

A fixed, full-50 single-particle Hermitian walk may change passage marginal
probabilities enough to exceed the stored Top-k Silver-alignment diagnostic.
The mechanism is intentionally not a joint five-passage quantum state and is
explicitly recorded as classically exactly simulable.

## Frozen mechanism

- Six qubits: valid states `|0>` through `|49>`, isolated invalid states `|50>` through `|63>`.
- Retrieval and Answer Scorer values are independently min-max normalized per Top-50 and averaged 0.5/0.5.
- Initial state is the softmax of that score at temperature 1.0.
- `H = 1.0 * diag(z) + 0.85 * C`, with centered off-diagonal cosine `C` divided by its spectral norm and evolution time `tau=1.0`.
- Exact spectral matrix exponential, exact Born probabilities, no sampling, tie break by retrieved rank then passage ID.
- Controls: matched real positive-affinity diffusion, dephased unitary population transfer, and deterministic SHA-256 passage-identity phase scramble.

## Gates and ceiling

All 100 cases must pass ordered identity, Hermitian, block-leakage, nonnegative
normalization, zero-time identity, deterministic replay, and no-label boundary
checks within a 600-second CPU budget. Only a strict Born mean Silver overlap
greater than stored Top-k `3.55/5` unlocks a later complete selector replay.
That condition does not authorize full-data handoff and cannot establish L1/L2
or quantum advantage.
