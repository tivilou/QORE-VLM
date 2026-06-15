"""
Synthetic experiments: demonstrate QUBO-SA beats greedy top-K under redundancy.

This module serves as both a test suite and a reproducible experiment.
Run with: pytest qore/tests/test_synthetic.py -v -s
"""

import time

import numpy as np
import pytest

from qore.qubo import build_qubo_matrix, energy, energy_decomposed
from qore.signals import cosine_redundancy, normalize
from qore.solvers import solve


# ---------------------------------------------------------------------------
# Helpers for generating synthetic data
# ---------------------------------------------------------------------------

def make_clustered_data(
    n_clusters: int = 5,
    items_per_cluster: int = 10,
    dim: int = 64,
    within_noise: float = 0.05,
    seed: int = 42,
):
    """
    Generate synthetic data with controlled cluster structure.

    Items in the same cluster are nearly identical (high redundancy).
    Items in different clusters are diverse (low redundancy).

    Returns:
        features: (N, dim) feature matrix
        a: (N,) quality scores (cluster centers get higher base quality)
        cluster_ids: (N,) integer cluster membership
    """
    rng = np.random.default_rng(seed)
    N = n_clusters * items_per_cluster

    # Generate cluster centers far apart
    centers = rng.standard_normal((n_clusters, dim)) * 3.0

    features = np.zeros((N, dim))
    cluster_ids = np.zeros(N, dtype=int)

    for c in range(n_clusters):
        start = c * items_per_cluster
        end = start + items_per_cluster
        noise = rng.standard_normal((items_per_cluster, dim)) * within_noise
        features[start:end] = centers[c] + noise
        cluster_ids[start:end] = c

    # Quality: base quality varies by cluster, with some within-cluster variation
    base_quality = rng.uniform(0.5, 1.0, size=n_clusters)
    a = np.zeros(N)
    for c in range(n_clusters):
        start = c * items_per_cluster
        end = start + items_per_cluster
        a[start:end] = base_quality[c] + rng.uniform(-0.05, 0.05, items_per_cluster)

    return features, a, cluster_ids


def make_uniform_data(N: int = 50, dim: int = 64, seed: int = 42):
    """
    Generate data with no cluster structure (low redundancy).

    Returns:
        features: (N, dim) feature matrix
        a: (N,) quality scores
    """
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((N, dim))
    a = rng.uniform(0.3, 1.0, size=N)
    return features, a


def cluster_coverage(x: np.ndarray, cluster_ids: np.ndarray) -> int:
    """Count how many distinct clusters are represented in the selection."""
    selected = np.where(x == 1)[0]
    return len(set(cluster_ids[selected]))


def information_retention(
    x: np.ndarray, features: np.ndarray
) -> float:
    """
    Measure how well the selected subset covers the full set.

    For each dropped item, find its distance to the nearest kept item.
    Lower average distance = better coverage = higher retention.
    Returns a score in [0, 1] where 1 = perfect coverage.
    """
    selected = np.where(x == 1)[0]
    dropped = np.where(x == 0)[0]

    if len(dropped) == 0:
        return 1.0

    kept_features = features[selected]
    dropped_features = features[dropped]

    # Pairwise distances from dropped to kept
    # Use cosine distance for consistency with our redundancy metric
    kept_norm = kept_features / np.maximum(
        np.linalg.norm(kept_features, axis=1, keepdims=True), 1e-12
    )
    dropped_norm = dropped_features / np.maximum(
        np.linalg.norm(dropped_features, axis=1, keepdims=True), 1e-12
    )

    similarity = dropped_norm @ kept_norm.T  # (n_dropped, n_kept)
    max_sim = similarity.max(axis=1)  # best match for each dropped item
    retention = max_sim.mean()  # average retention

    return float(retention)


# ---------------------------------------------------------------------------
# Experiment: High Redundancy (QUBO should clearly win)
# ---------------------------------------------------------------------------

class TestHighRedundancy:
    """Scenario 2: clustered data where greedy wastes budget on redundancy."""

    @pytest.fixture
    def scenario(self):
        features, a, cluster_ids = make_clustered_data(
            n_clusters=5, items_per_cluster=10, within_noise=0.03, seed=42
        )
        b = cosine_redundancy(features)
        K = 10
        lam = 2.0
        return {
            "features": features,
            "a": normalize(a),
            "b": b,
            "K": K,
            "lam": lam,
            "cluster_ids": cluster_ids,
        }

    def test_qubo_better_energy(self, scenario):
        """QUBO-SA should achieve lower QUBO energy than greedy."""
        a, b, K, lam = scenario["a"], scenario["b"], scenario["K"], scenario["lam"]
        Q = build_qubo_matrix(a, b, K, lam)

        x_greedy = solve(a, b, K, lam=lam, method="greedy")
        x_sa = solve(a, b, K, lam=lam, method="anneal", num_reads=200, seed=42)

        e_greedy = energy(x_greedy, Q)
        e_sa = energy(x_sa, Q)

        print(f"\n  Energy — greedy: {e_greedy:.4f}, SA: {e_sa:.4f}, "
              f"improvement: {e_greedy - e_sa:.4f}")
        assert e_sa <= e_greedy, "SA should achieve equal or lower energy than greedy"

    def test_qubo_better_coverage(self, scenario):
        """QUBO-SA should select from more clusters than greedy."""
        a, b, K, lam = scenario["a"], scenario["b"], scenario["K"], scenario["lam"]
        cluster_ids = scenario["cluster_ids"]

        x_greedy = solve(a, b, K, lam=lam, method="greedy")
        x_sa = solve(a, b, K, lam=lam, method="anneal", num_reads=200, seed=42)

        cov_greedy = cluster_coverage(x_greedy, cluster_ids)
        cov_sa = cluster_coverage(x_sa, cluster_ids)

        print(f"\n  Coverage — greedy: {cov_greedy}/5 clusters, "
              f"SA: {cov_sa}/5 clusters")
        assert cov_sa >= cov_greedy, "SA should cover at least as many clusters"

    def test_qubo_better_retention(self, scenario):
        """QUBO-SA should have higher information retention."""
        a, b, K, lam = scenario["a"], scenario["b"], scenario["K"], scenario["lam"]
        features = scenario["features"]

        x_greedy = solve(a, b, K, lam=lam, method="greedy")
        x_sa = solve(a, b, K, lam=lam, method="anneal", num_reads=200, seed=42)

        ret_greedy = information_retention(x_greedy, features)
        ret_sa = information_retention(x_sa, features)

        print(f"\n  Retention — greedy: {ret_greedy:.4f}, SA: {ret_sa:.4f}")
        assert ret_sa >= ret_greedy - 0.01, "SA should have comparable or better retention"

    def test_greedy_picks_redundant(self, scenario):
        """Verify that greedy actually falls into the redundancy trap."""
        a, b, K = scenario["a"], scenario["b"], scenario["K"]
        cluster_ids = scenario["cluster_ids"]

        x_greedy = solve(a, b, K, lam=scenario["lam"], method="greedy")
        cov = cluster_coverage(x_greedy, cluster_ids)

        # With 5 clusters of 10 items and K=10, greedy should NOT cover all 5
        # (it should concentrate on high-quality clusters)
        print(f"\n  Greedy covers {cov}/5 clusters (expected < 5)")
        assert cov < 5, "Greedy should miss some clusters due to redundant picks"


# ---------------------------------------------------------------------------
# Experiment: Low Redundancy (QUBO should not hurt)
# ---------------------------------------------------------------------------

class TestLowRedundancy:
    """Scenario 1: diverse data where greedy is already near-optimal."""

    @pytest.fixture
    def scenario(self):
        features, a = make_uniform_data(N=50, dim=64, seed=42)
        b = cosine_redundancy(features)
        K = 10
        lam = 2.0
        return {"features": features, "a": normalize(a), "b": b, "K": K, "lam": lam}

    def test_qubo_no_worse_than_greedy(self, scenario):
        """When redundancy is low, QUBO should not hurt vs greedy."""
        a, b, K, lam = scenario["a"], scenario["b"], scenario["K"], scenario["lam"]
        features = scenario["features"]

        x_greedy = solve(a, b, K, lam=lam, method="greedy")
        x_sa = solve(a, b, K, lam=lam, method="anneal", num_reads=100, seed=42)

        ret_greedy = information_retention(x_greedy, features)
        ret_sa = information_retention(x_sa, features)

        print(f"\n  Low-redundancy retention — greedy: {ret_greedy:.4f}, SA: {ret_sa:.4f}")
        # Allow tiny margin for SA randomness
        assert ret_sa >= ret_greedy - 0.05


# ---------------------------------------------------------------------------
# Timing Benchmark
# ---------------------------------------------------------------------------

class TestTiming:
    """Scenario 3: measure SA solve time at various scales."""

    @pytest.mark.parametrize("N", [20, 30, 50, 75, 100])
    def test_solve_time(self, N):
        """
        SA solve time benchmark.

        In production, block decomposition keeps sub-QUBOs at n=24-32 with
        num_reads=20-50. Here we test with num_reads=30 (realistic for latency-
        sensitive paths) and assert that time scales reasonably.
        Threshold: < 20ms * N for this environment.
        """
        np.random.seed(0)
        features = np.random.randn(N, 32)
        a = normalize(np.random.rand(N))
        b = cosine_redundancy(features)
        K = max(2, N // 5)
        lam = 2.0
        Q = build_qubo_matrix(a, b, K, lam)

        # Warm-up
        _ = solve(a, b, K, lam=lam, method="anneal", num_reads=10, seed=0)

        # Timed run with production-like num_reads
        num_reads = 30
        start = time.perf_counter()
        x = solve(a, b, K, lam=lam, method="anneal", num_reads=num_reads, seed=1)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert x.sum() == K
        # Generous threshold — this is a "not absurdly slow" sanity check, not a
        # precise benchmark. Timing is sensitive to machine load (especially when
        # the full suite runs in parallel), so we allow a wide margin to avoid
        # flaky failures. SA is ~O(N^2) per read.
        threshold_ms = 50 * N  # 1s for N=20, 5s for N=100
        print(f"\n  N={N}, K={K}, reads={num_reads}: {elapsed_ms:.1f}ms "
              f"(threshold: {threshold_ms}ms)")
        assert elapsed_ms < threshold_ms, (
            f"SA took {elapsed_ms:.1f}ms for N={N}, expected < {threshold_ms}ms"
        )


# ---------------------------------------------------------------------------
# Full comparison report (not a test, run manually with -s flag)
# ---------------------------------------------------------------------------

class TestFullReport:
    """Generate a comparison table for the paper / documentation."""

    def test_comparison_table(self):
        """Print a full comparison across redundancy levels."""
        print("\n" + "=" * 70)
        print("QORE Synthetic Experiment: SA-QUBO vs Greedy Top-K")
        print("=" * 70)

        configs = [
            ("Low redundancy (uniform)", make_uniform_data, {}),
            ("High redundancy (5 clusters)", make_clustered_data,
             {"n_clusters": 5, "items_per_cluster": 10, "within_noise": 0.03}),
            ("Very high redundancy (3 clusters)", make_clustered_data,
             {"n_clusters": 3, "items_per_cluster": 17, "within_noise": 0.01}),
        ]

        for name, gen_fn, kwargs in configs:
            if gen_fn == make_uniform_data:
                features, a = gen_fn(N=50, seed=42)
                cluster_ids = None
            else:
                features, a, cluster_ids = gen_fn(seed=42, **kwargs)

            a = normalize(a)
            b = cosine_redundancy(features)
            N = len(a)
            K = 10
            lam = 2.0
            Q = build_qubo_matrix(a, b, K, lam)

            x_greedy = solve(a, b, K, lam=lam, method="greedy")
            x_sa = solve(a, b, K, lam=lam, method="anneal", num_reads=200, seed=42)

            e_greedy = energy(x_greedy, Q)
            e_sa = energy(x_sa, Q)
            ret_greedy = information_retention(x_greedy, features)
            ret_sa = information_retention(x_sa, features)

            print(f"\n{'─' * 70}")
            print(f"  {name} (N={N}, K={K})")
            print(f"{'─' * 70}")
            print(f"  {'Metric':<25} {'Greedy':>10} {'SA-QUBO':>10} {'Δ':>10}")
            print(f"  {'─' * 55}")
            print(f"  {'Energy':<25} {e_greedy:>10.3f} {e_sa:>10.3f} "
                  f"{e_greedy - e_sa:>+10.3f}")
            print(f"  {'Info retention':<25} {ret_greedy:>10.4f} {ret_sa:>10.4f} "
                  f"{ret_sa - ret_greedy:>+10.4f}")

            if cluster_ids is not None:
                cov_g = cluster_coverage(x_greedy, cluster_ids)
                cov_s = cluster_coverage(x_sa, cluster_ids)
                n_cl = len(set(cluster_ids))
                print(f"  {'Cluster coverage':<25} {cov_g:>7}/{n_cl:>2} "
                      f"{cov_s:>7}/{n_cl:>2} {cov_s - cov_g:>+10d}")

            # Decomposed energy
            d_greedy = energy_decomposed(x_greedy, a, b, K, lam)
            d_sa = energy_decomposed(x_sa, a, b, K, lam)
            print(f"  {'Quality term':<25} {d_greedy['quality']:>10.3f} "
                  f"{d_sa['quality']:>10.3f}")
            print(f"  {'Redundancy term':<25} {d_greedy['redundancy']:>10.3f} "
                  f"{d_sa['redundancy']:>10.3f}")
            print(f"  {'Penalty term':<25} {d_greedy['penalty']:>10.3f} "
                  f"{d_sa['penalty']:>10.3f}")

        print(f"\n{'=' * 70}")
        print("  Conclusion: SA-QUBO should show clear gains under high redundancy")
        print("  while matching greedy under low redundancy.")
        print("=" * 70)
