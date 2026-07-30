"""Generate synthetic training data for Idea 7 MVP test.

Creates a minimal dataset with known gold passages to verify that:
1. Soft QUBO can be trained end-to-end
2. Gradients flow correctly
3. Recall improves over epochs
"""

import json
import numpy as np
from pathlib import Path


def generate_synthetic_sample(n_passages=20, n_gold=2, embed_dim=128, seed=None):
    """Generate one synthetic training sample.

    Args:
        n_passages: total number of candidate passages
        n_gold: number of gold (relevant) passages
        embed_dim: embedding dimension
        seed: random seed

    Returns:
        dict with embeddings and gold_indices
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random embeddings
    embeddings = np.random.randn(n_passages, embed_dim).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)

    # Gold indices (first n_gold passages)
    gold_indices = list(range(n_gold))

    # Make gold passages slightly more similar to each other
    # (simulating that they talk about the same topic)
    if n_gold > 1:
        mean_gold = embeddings[gold_indices].mean(axis=0)
        for i in gold_indices:
            embeddings[i] = 0.7 * embeddings[i] + 0.3 * mean_gold
            embeddings[i] /= (np.linalg.norm(embeddings[i]) + 1e-12)

    return {
        "question": f"Synthetic question {seed}",
        "embeddings": embeddings.tolist(),
        "gold_indices": gold_indices,
        "texts": [f"Passage {i}" for i in range(n_passages)],
    }


def main():
    output_dir = Path("exchange/idea7_synthetic_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 20 training samples
    samples = []
    for i in range(20):
        sample = generate_synthetic_sample(
            n_passages=20,
            n_gold=2,
            embed_dim=128,
            seed=42 + i,
        )
        samples.append(sample)

    # Save as JSON
    output_path = output_dir / "synthetic_train.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"✅ Generated {len(samples)} synthetic samples")
    print(f"   Saved to: {output_path}")
    print(f"\nSample structure:")
    print(f"  - n_passages: {len(samples[0]['embeddings'])}")
    print(f"  - embed_dim: {len(samples[0]['embeddings'][0])}")
    print(f"  - n_gold: {len(samples[0]['gold_indices'])}")
    print(f"\nTo train:")
    print(f"  python -m scripts.idea7.train_soft_qubo_simple \\")
    print(f"    --result_json {output_path} \\")
    print(f"    --epochs 50 \\")
    print(f"    --output_dir exchange/idea7_mvp")


if __name__ == "__main__":
    main()
