# Phase 1 Implementation Plan: Core QUBO Framework

## Goal
Build the `qore/` module that can:
1. Construct a QUBO matrix from quality + redundancy signals
2. Solve it with multiple backends (brute-force, simulated annealing, greedy)
3. Demonstrate on synthetic data that "QUBO-SA > greedy top-K" especially under high redundancy
4. Establish timing benchmarks

## Dependencies to Install
```
dwave-neal   # SimulatedAnnealingSampler
dimod        # BQM/QUBO data structures
```

## File Structure
```
qore/
├── __init__.py              # expose main API
├── qubo.py                  # build_qubo_matrix(a, b, K, lam) -> Q
├── signals.py               # cosine_redundancy, rbf_redundancy, normalize
├── block_decompose.py       # partition large problems into sub-QUBOs
├── solvers/
│   ├── __init__.py          # unified solve() dispatcher
│   ├── brute.py             # exact enumeration (N ≤ 20)
│   ├── anneal.py            # dwave-neal SA
│   └── greedy.py            # top-K by quality only (baseline)
└── tests/
    ├── __init__.py
    ├── test_qubo.py          # Q matrix math correctness
    ├── test_solvers.py       # all solvers agree on small N
    └── test_synthetic.py     # SA vs greedy on controlled scenarios + timing
```

## Module Designs

### qore/qubo.py
```python
def build_qubo_matrix(a, b, K, lam=2.0):
    """
    Build QUBO matrix Q such that min x^T Q x gives optimal subset.
    
    Args:
        a: (N,) quality scores (higher = more important)
        b: (N, N) redundancy matrix (symmetric, zero diag; higher = more redundant)
        K: int, budget (number of items to select)
        lam: float, penalty weight for budget constraint
    
    Returns:
        Q: (N, N) upper-triangular QUBO matrix
    """
    # Diagonal: -a_i + lam*(1 - 2K)
    # Off-diagonal (i<j): b_ij + 2*lam
```

### qore/signals.py
```python
def cosine_redundancy(features):
    """Pairwise cosine similarity matrix, clipped to [0, 1]."""

def rbf_redundancy(features, sigma=1.0):
    """RBF kernel similarity matrix."""

def normalize(a):
    """Min-max normalize to [0, 1]."""
```

### qore/block_decompose.py
```python
def decompose(a, b, K, num_blocks, strategy="proportional"):
    """Split N items into num_blocks groups, allocate budget proportionally."""

def recompose(block_solutions, block_indices, N):
    """Merge block solutions into a single global binary vector."""
```

### qore/solvers/brute.py
```python
def solve(Q, K):
    """Exact: enumerate all C(N,K) subsets, return minimum energy solution."""
    # Only feasible for N ≤ 20
```

### qore/solvers/anneal.py
```python
def solve(Q, K, num_reads=100, seed=None):
    """Simulated annealing via dwave-neal."""
    # Convert Q to BQM, sample, return lowest-energy feasible solution
```

### qore/solvers/greedy.py
```python
def solve(a, K):
    """Greedy top-K: select K items with highest quality score a_i."""
    # Ignores redundancy - this is the baseline to beat
```

### qore/solvers/__init__.py
```python
def solve(Q=None, a=None, b=None, K=None, method="anneal", **kwargs):
    """Unified dispatcher."""
```

## Synthetic Experiment Design (test_synthetic.py)

### Scenario 1: Low Redundancy
- N=50 items with random features (d=64), well-spread
- b_ij values are low → greedy ≈ QUBO (both should work well)
- Purpose: sanity check, verify QUBO doesn't hurt when redundancy is low

### Scenario 2: High Redundancy (key demonstration)
- N=50 items, but arranged in 5 clusters of 10 similar items each
- Within-cluster b_ij ≈ 0.8-0.95, between-cluster b_ij ≈ 0.1-0.3
- K=10 budget
- Greedy will pick all top-scored items (likely from 1-2 clusters) → redundant selection
- QUBO-SA will spread selections across clusters → better coverage
- Purpose: demonstrate the core value proposition

### Scenario 3: Scaling
- N = 20, 30, 40, 50, 75, 100
- Measure SA solve time for each N
- Show the timing curve stays well under 10ms for N ≤ 100

### Metrics
- **Energy**: E(x) value (lower = better, directly measures our objective)
- **Coverage**: number of distinct clusters represented in selection
- **Information retention**: 1 - reconstruction_error (how well can dropped items be reconstructed from kept ones)
- **Solve time**: wall-clock milliseconds

### Expected Results
- Scenario 1: QUBO-SA ≈ greedy (within noise)
- Scenario 2: QUBO-SA >> greedy on coverage and information retention
- Scenario 3: SA solve time < 5ms for N=100

## Verification Criteria
1. All solvers agree with brute-force on N=12, K=4 (exact correctness)
2. QUBO-SA achieves lower energy than greedy in Scenario 2 (≥95% of runs)
3. QUBO-SA selects from more clusters than greedy in Scenario 2
4. SA solve time < 10ms for N ≤ 100
5. All tests pass with pytest
