# Phase 2: Cohesion Calibration

`cohesion` is an additive QUBO enhancer. It leaves retrieval, quality
scoring, candidate pooling, solving, and evaluation unchanged.

The calibrated mode uses only deterministic pool statistics:

```text
m_q = max(a[K-1] - a[K], 0)
c_q = mean(max(b_ij, 0)), i < j
delta_q = eta * m_q / ((K - 1) * (c_q + eps))
```

The plugin adds `delta_q * b` to the current interaction matrix. `fixed` uses
the configured `weight`, and `disabled` returns the input matrix exactly. The
plugin emits `delta_q`, `m_q`, `c_q`, `K`, and correction norm in the enhancer
diagnostic trace.

The required Phase 2 ablations are defined in:

- `configs/experiments/cohesion_calibration.yaml`
- `configs/experiments/cohesion_fixed.yaml`
- `configs/experiments/cohesion_calibrated.yaml`
- `configs/experiments/cohesion_disabled.yaml`

The full `wiki_dpr` gate must be run by the collaborator. Local validation is
limited to plugin contracts, synthetic selector fixtures, and the existing
tracked test suite.
