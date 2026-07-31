"""Example: Using the enhancer plugin system.

This script demonstrates how to use the refactored QORE selector with
pluggable enhancers for clean idea composition.
"""

import numpy as np
from applications.rag.selector import select_passages
from qore.enhancers import list_enhancers

print("=" * 70)
print("QORE Enhancer Plugin System - Usage Examples")
print("=" * 70)
print()

# Check available enhancers
print("Available enhancers:")
for name in list_enhancers():
    print(f"  - {name}")
print()

# Create dummy data
N = 50  # 50 candidate passages
K = 5   # Select 5
d = 128  # Embedding dimension

np.random.seed(42)
query_emb = np.random.randn(d)
passage_embs = np.random.randn(N, d)

# Normalize embeddings
query_emb = query_emb / np.linalg.norm(query_emb)
passage_embs = passage_embs / np.linalg.norm(passage_embs, axis=1, keepdims=True)

print("=" * 70)
print("Example 1: Using baseline (standard QUBO)")
print("=" * 70)

indices_baseline = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=K,
    method="qore",
    enhancers=["baseline"],
    enhancer_configs={"baseline": {"gamma": 1.0}},
    seed=42,
)

print(f"Selected passage indices: {indices_baseline}")
print()

print("=" * 70)
print("Example 2: Using Idea 6 (complementarity)")
print("=" * 70)
print("NOTE: Requires answer_scorer, passages, and question in real usage.")
print("Skipping Idea 6 demo due to missing context dependencies.")
print()

print("=" * 70)
print("Example 3: Backward compatibility (legacy API)")
print("=" * 70)

# Old code still works!
indices_legacy = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=K,
    method="qore",
    gamma=1.0,  # Legacy parameter
    seed=42,
)

print(f"Selected passage indices: {indices_legacy}")
print(f"Same as baseline? {np.array_equal(indices_baseline, indices_legacy)}")
print()

print("=" * 70)
print("Example 4: Comparing methods")
print("=" * 70)

# Top-K (baseline comparison)
from applications.rag.selector import select_passages

indices_topk = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=K,
    method="topk",
)

# MMR (diversity baseline)
indices_mmr = select_passages(
    query_embedding=query_emb,
    passage_embeddings=passage_embs,
    K=K,
    method="mmr",
    lambda_mmr=0.5,
)

print(f"QORE (baseline):  {indices_baseline}")
print(f"Top-K:            {indices_topk}")
print(f"MMR:              {indices_mmr}")
print()

# Check overlap
overlap_topk = len(set(indices_baseline) & set(indices_topk))
overlap_mmr = len(set(indices_baseline) & set(indices_mmr))

print(f"Overlap with Top-K: {overlap_topk}/{K}")
print(f"Overlap with MMR:   {overlap_mmr}/{K}")
print()

print("=" * 70)
print("✅ Examples completed!")
print("=" * 70)
print()
print("Next steps:")
print("1. See docs/ENHANCER_PLUGIN_SYSTEM.md for full documentation")
print("2. See docs/ENHANCER_DEVELOPER_GUIDE.md to add new ideas")
print("3. See configs/experiments/ for experiment configurations")
print("4. Run PYTHONPATH=. python scripts/test_enhancers.py for tests")
