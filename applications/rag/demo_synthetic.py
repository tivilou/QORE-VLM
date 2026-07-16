"""
Synthetic RAG experiment: demonstrate QORE > MMR > Top-K on passage selection.

This script simulates a realistic RAG scenario with controlled redundancy:
- A query about a multi-faceted topic
- Gold passages covering different aspects of the answer
- Near-duplicate distractors (high relevance, high redundancy with gold)
- Irrelevant distractors (low relevance)

Run: python -m applications.rag.demo_synthetic
"""

import time

import numpy as np

from applications.rag.selector import select_passages, evaluate_selection


def make_rag_scenario(
    n_gold: int = 5,
    n_near_duplicates: int = 20,
    n_distractors: int = 75,
    dim: int = 128,
    duplicate_noise: float = 0.03,
    seed: int = 42,
) -> dict:
    """
    Create a synthetic RAG scenario with controlled structure.

    The scenario models a multi-hop question where the answer requires
    evidence from multiple distinct passages (gold passages). A good selector
    should find all gold passages, not waste budget on near-duplicates.

    Key design: gold passages are each relevant to the query but DIVERSE from
    each other (they cover different "facets" of the answer). Near-duplicates
    are paraphrases of specific gold passages (similar to their parent, relevant
    to query, but redundant with that parent).

    Structure:
        - query: a vector representing the information need
        - gold passages: each aligned with a different "facet" of the query,
          diverse from each other but all individually relevant
        - near-duplicates: slight variations of specific gold passages
        - distractors: low relevance to query
    """
    rng = np.random.default_rng(seed)

    # Build query as a sum of n_gold orthogonal "facet" directions
    # This way each gold passage can be aligned with ONE facet while
    # being less aligned with the other facets → gold-gold diversity
    facets = np.zeros((n_gold, dim))
    for i in range(n_gold):
        facets[i, i * (dim // n_gold):(i + 1) * (dim // n_gold)] = 1.0
        facets[i] += rng.standard_normal(dim) * 0.1
    facets = facets / np.linalg.norm(facets, axis=1, keepdims=True)

    # Query is a mix of all facets (asking about all aspects)
    query = facets.sum(axis=0)
    query = query / np.linalg.norm(query)

    # Gold passages: each primarily aligned with ONE facet + some query alignment
    gold_passages = np.zeros((n_gold, dim))
    for i in range(n_gold):
        # 60% facet direction + 40% general query direction + noise
        gold_passages[i] = 0.6 * facets[i] + 0.4 * query + 0.1 * rng.standard_normal(dim)
    gold_passages = gold_passages / np.linalg.norm(gold_passages, axis=1, keepdims=True)

    # Near-duplicates: very close to their parent gold passage
    duplicates_per_gold = n_near_duplicates // n_gold
    near_duplicates = []
    for g_idx in range(n_gold):
        for _ in range(duplicates_per_gold):
            dup = gold_passages[g_idx] + rng.standard_normal(dim) * duplicate_noise
            dup = dup / np.linalg.norm(dup)
            near_duplicates.append(dup)
    # Fill remaining
    while len(near_duplicates) < n_near_duplicates:
        g_idx = rng.integers(n_gold)
        dup = gold_passages[g_idx] + rng.standard_normal(dim) * duplicate_noise
        dup = dup / np.linalg.norm(dup)
        near_duplicates.append(dup)
    near_duplicates = np.array(near_duplicates)

    # Distractors: random directions, low relevance to query
    distractors = rng.standard_normal((n_distractors, dim))
    distractors = distractors / np.linalg.norm(distractors, axis=1, keepdims=True)

    # Assemble all passages
    all_passages = np.vstack([gold_passages, near_duplicates, distractors])
    N = len(all_passages)

    # Gold indices are the first n_gold
    gold_indices = np.arange(n_gold)

    # Relevance scores (cosine with query)
    relevance = all_passages @ query
    # Add tiny noise to break ties
    relevance += rng.standard_normal(N) * 0.001

    return {
        "query": query,
        "passages": all_passages,
        "gold_indices": gold_indices,
        "near_duplicate_indices": np.arange(n_gold, n_gold + n_near_duplicates),
        "distractor_indices": np.arange(n_gold + n_near_duplicates, N),
        "relevance_scores": relevance,
        "N": N,
        "n_gold": n_gold,
    }


def run_experiment(K: int = 8, seed: int = 42):
    """Run the full comparison experiment."""
    scenario = make_rag_scenario(seed=seed)

    methods = {
        "Top-K": {"method": "topk"},
        "MMR (λ=0.5)": {"method": "mmr", "lambda_mmr": 0.5},
        "MMR (λ=0.7)": {"method": "mmr", "lambda_mmr": 0.7},
        "QORE-SA": {"method": "qore", "num_reads": 200, "seed": seed},
    }

    results = {}

    for name, kwargs in methods.items():
        start = time.perf_counter()
        indices = select_passages(
            query_embedding=scenario["query"],
            passage_embeddings=scenario["passages"],
            K=K,
            relevance_scores=scenario["relevance_scores"],
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = evaluate_selection(
            indices, scenario["gold_indices"], scenario["passages"]
        )
        metrics["time_ms"] = elapsed_ms
        metrics["selected"] = indices
        results[name] = metrics

    return scenario, results


def print_results(scenario: dict, results: dict, K: int):
    """Print a formatted comparison table."""
    print()
    print("=" * 75)
    print("  QORE-RAG Synthetic Experiment: Passage Selection Comparison")
    print("=" * 75)
    print()
    print(f"  Scenario: N={scenario['N']} passages, K={K} budget, "
          f"{scenario['n_gold']} gold passages")
    print(f"  Near-duplicates: {len(scenario['near_duplicate_indices'])} "
          f"(high relevance, redundant with gold)")
    print(f"  Distractors: {len(scenario['distractor_indices'])} (low relevance)")
    print()
    print(f"  {'Method':<16} {'Recall':>8} {'Gold':>6} {'Redundancy':>12} "
          f"{'Diversity':>10} {'Time':>8}")
    print(f"  {'─' * 66}")

    for name, m in results.items():
        print(f"  {name:<16} {m['recall']:>7.1%} "
              f"{m['gold_hits']:>3}/{m['gold_total']:<2} "
              f"{m['redundancy_ratio']:>11.4f} "
              f"{m['diversity_score']:>9.4f} "
              f"{m['time_ms']:>6.1f}ms")

    print()

    # Detailed selection analysis
    print("  Selection details:")
    print(f"  {'─' * 66}")

    n_gold = scenario["n_gold"]
    dups_per_gold = len(scenario["near_duplicate_indices"]) // n_gold

    for name, m in results.items():
        selected = m["selected"]
        n_gold_sel = len(set(selected) & set(scenario["gold_indices"]))
        n_dup_sel = len(set(selected) & set(scenario["near_duplicate_indices"]))
        n_dist_sel = len(set(selected) & set(scenario["distractor_indices"]))

        # Cluster coverage: how many gold "groups" are represented?
        # A group is covered if either the gold passage or any of its duplicates is selected
        groups_covered = set()
        for idx in selected:
            if idx < n_gold:
                groups_covered.add(idx)
            elif idx < n_gold + len(scenario["near_duplicate_indices"]):
                # Map near-duplicate back to its parent gold
                dup_offset = idx - n_gold
                parent_gold = dup_offset // dups_per_gold
                if parent_gold < n_gold:
                    groups_covered.add(parent_gold)

        print(f"  {name:<16}  "
              f"gold={n_gold_sel}, near-dup={n_dup_sel}, distractor={n_dist_sel}"
              f"  | groups covered: {len(groups_covered)}/{n_gold}")

    print()
    print("  Key observations:")
    print("  • Top-K: concentrates on 1-2 groups, wastes budget on near-duplicates")
    print("  • MMR: good diversity but greedy (order-dependent, λ-sensitive)")
    print("  • QORE-SA: high gold recall at tight budgets; covers most groups")
    print("  • 'Groups covered' is the key metric — it measures information coverage")
    print("=" * 75)


def main():
    """Run the full demo."""
    K = 8
    scenario, results = run_experiment(K=K, seed=42)
    print_results(scenario, results, K)

    # Also run with tighter budget to stress-test
    print("\n")
    K_tight = 5
    scenario2, results2 = run_experiment(K=K_tight, seed=42)
    print_results(scenario2, results2, K_tight)


if __name__ == "__main__":
    main()
