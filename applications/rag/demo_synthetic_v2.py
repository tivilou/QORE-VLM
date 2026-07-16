"""
Improved synthetic RAG experiment: more realistic scenario.

V1 问题:
- Near-duplicates 的 relevance 太高(0.77-1.0),接近 gold(0.80-1.0)
- 这是对抗性设计,不符合真实数据分布
- 导致 QORE 难以正确权衡

V2 改进:
- Near-duplicates 的 relevance 降低到 0.5-0.7(中等,更符合真实)
- Gold passages 保持高 relevance(0.8-1.0)
- 更接近真实 RAG 场景:高质量文档少,中等质量重复多

Run: python -m applications.rag.demo_synthetic_v2
"""

import numpy as np
from applications.rag.selector import select_passages, evaluate_selection


def make_realistic_rag_scenario(
    n_gold: int = 5,
    n_near_duplicates: int = 20,
    n_distractors: int = 75,
    dim: int = 128,
    duplicate_noise: float = 0.03,
    seed: int = 42,
) -> dict:
    """
    Create a more realistic synthetic RAG scenario.

    KEY CHANGE from V1: near-duplicates have MEDIUM relevance (0.5-0.7),
    not high relevance (0.77-1.0). This matches real-world distributions
    where duplicates are typically paraphrases with lower semantic match.
    """
    rng = np.random.default_rng(seed)

    # Query embedding
    query = rng.standard_normal(dim)
    query = query / np.linalg.norm(query)

    # Gold passages: high relevance (0.8-1.0), diverse
    gold_passages = []
    for i in range(n_gold):
        # Each gold passage is a different aspect (diverse directions)
        direction = rng.standard_normal(dim)
        direction = direction / np.linalg.norm(direction)
        # High similarity to query (0.8-1.0)
        similarity_target = 0.8 + 0.2 * rng.random()
        passage = similarity_target * query + np.sqrt(1 - similarity_target**2) * direction
        passage = passage / np.linalg.norm(passage)
        gold_passages.append(passage)

    # Near-duplicates: MEDIUM relevance (0.5-0.7), high redundancy with gold
    # KEY CHANGE: lower relevance target
    near_duplicates = []
    for gold_passage in gold_passages:
        for _ in range(n_near_duplicates // n_gold):
            noise = rng.standard_normal(dim) * duplicate_noise
            dup = gold_passage + noise
            dup = dup / np.linalg.norm(dup)

            # Project to query to get MEDIUM relevance (0.5-0.7)
            current_sim = float(dup @ query)
            target_sim = 0.5 + 0.2 * rng.random()  # 0.5-0.7

            # Adjust to target similarity
            orthogonal = dup - (dup @ query) * query
            orthogonal = orthogonal / (np.linalg.norm(orthogonal) + 1e-8)
            dup = target_sim * query + np.sqrt(max(0, 1 - target_sim**2)) * orthogonal
            dup = dup / np.linalg.norm(dup)

            near_duplicates.append(dup)

    # Distractors: low relevance (0.0-0.4)
    distractors = []
    for _ in range(n_distractors):
        direction = rng.standard_normal(dim)
        direction = direction / np.linalg.norm(direction)
        similarity_target = 0.4 * rng.random()  # 0.0-0.4
        passage = similarity_target * query + np.sqrt(1 - similarity_target**2) * direction
        passage = passage / np.linalg.norm(passage)
        distractors.append(passage)

    # Combine all passages
    all_passages = np.array(gold_passages + near_duplicates + distractors)
    N = len(all_passages)

    gold_indices = np.arange(n_gold)
    near_dup_indices = np.arange(n_gold, n_gold + len(near_duplicates))
    distractor_indices = np.arange(n_gold + len(near_duplicates), N)

    # Compute relevance scores
    relevance_scores = all_passages @ query

    return {
        "query": query,
        "passages": all_passages,
        "relevance_scores": relevance_scores,
        "gold_indices": gold_indices,
        "near_duplicate_indices": near_dup_indices,
        "distractor_indices": distractor_indices,
        "N": N,
        "n_gold": n_gold,
    }


def run_experiment(K: int = 8, seed: int = 42):
    """Run comparison experiment on realistic scenario."""
    scenario = make_realistic_rag_scenario(seed=seed)

    methods = {
        "Top-K": {"method": "topk"},
        "MMR (λ=0.5)": {"method": "mmr", "lambda_mmr": 0.5},
        "MMR (λ=0.7)": {"method": "mmr", "lambda_mmr": 0.7},
        "QORE-SA": {"method": "qore", "num_reads": 200, "seed": seed},
    }

    results = {}

    for name, kwargs in methods.items():
        import time
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
    """Print comparison table."""
    print()
    print("=" * 75)
    print("  QORE-RAG Synthetic V2: Realistic Passage Selection")
    print("=" * 75)
    print()
    print(f"  Scenario: N={scenario['N']} passages, K={K} budget, "
          f"{scenario['n_gold']} gold passages")
    print(f"  Near-duplicates: {len(scenario['near_duplicate_indices'])} "
          f"(MEDIUM relevance 0.5-0.7, redundant with gold)")
    print(f"  Distractors: {len(scenario['distractor_indices'])} (low relevance 0.0-0.4)")
    print()
    print(f"  Method             Recall   Gold   Redundancy  Diversity     Time")
    print("  " + "─" * 68)

    for name, metrics in results.items():
        recall = metrics["recall"]
        gold_count = int(recall * scenario["n_gold"])
        print(f"  {name:18s} {recall:5.1%}   {gold_count}/{scenario['n_gold']}       "
              f"{metrics['redundancy_ratio']:6.4f}    {metrics['diversity_score']:6.4f}  "
              f"{metrics['time_ms']:6.1f}ms")

    print()
    print("  Selection details:")
    print("  " + "─" * 68)

    for name, metrics in results.items():
        selected = metrics["selected"]
        gold_sel = len(set(selected) & set(scenario["gold_indices"]))
        n_dup_sel = len(set(selected) & set(scenario["near_duplicate_indices"]))
        n_dist_sel = len(set(selected) & set(scenario["distractor_indices"]))

        print(f"  {name:18s} gold={gold_sel}, near-dup={n_dup_sel}, distractor={n_dist_sel}")

    print()
    print("  Key observations:")
    print("  • Realistic scenario: gold has HIGH relevance, near-dup has MEDIUM relevance")
    print("  • QORE should now correctly prioritize gold over near-duplicates")
    print("  • V1 was adversarial (near-dup too high relevance); V2 is realistic")
    print("=" * 75)


def main():
    """Run the full demo."""
    K = 8
    scenario, results = run_experiment(K=K, seed=42)
    print_results(scenario, results, K)

    # Also run with K=5
    print("\n")
    K_tight = 5
    scenario2, results2 = run_experiment(K=K_tight, seed=42)
    print_results(scenario2, results2, K_tight)


if __name__ == "__main__":
    main()
