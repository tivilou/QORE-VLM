"""Block decomposition for scaling QUBO to large N."""

import numpy as np
from typing import List, Tuple


def decompose(
    a: np.ndarray,
    b: np.ndarray,
    K: int,
    num_blocks: int,
    strategy: str = "proportional",
    partition: np.ndarray | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
    """
    Decompose a large (N, N) QUBO into smaller independent sub-problems.

    Args:
        a: (N,) quality scores.
        b: (N, N) redundancy matrix.
        K: Total budget.
        num_blocks: Number of blocks to split into.
        strategy: Budget allocation strategy.
            - "proportional": allocate K to blocks proportional to their total quality.
            - "uniform": allocate K / num_blocks to each block (rounded).
        partition: (N,) integer array assigning each item to a block [0, num_blocks).
            If None, items are assigned to blocks by contiguous chunks.

    Returns:
        List of (a_block, b_block, K_block, indices) tuples, one per block.
        indices is the array of original indices belonging to this block.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    N = len(a)

    if partition is None:
        # Default: contiguous chunks of roughly equal size
        partition = np.zeros(N, dtype=np.int64)
        splits = np.array_split(np.arange(N), num_blocks)
        for block_id, indices in enumerate(splits):
            partition[indices] = block_id

    blocks = []
    quality_sums = []

    for block_id in range(num_blocks):
        mask = partition == block_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        a_block = a[indices]
        b_block = b[np.ix_(indices, indices)]
        quality_sums.append(a_block.sum())
        blocks.append((a_block, b_block, indices))

    # Allocate budget
    total_quality = sum(quality_sums)
    result = []

    for i, (a_block, b_block, indices) in enumerate(blocks):
        if strategy == "proportional":
            if total_quality > 0:
                K_block = max(1, round(K * quality_sums[i] / total_quality))
            else:
                K_block = max(1, K // len(blocks))
        elif strategy == "uniform":
            K_block = max(1, K // len(blocks))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Clamp K_block to block size
        K_block = min(K_block, len(indices) - 1)
        K_block = max(1, K_block)

        result.append((a_block, b_block, K_block, indices))

    # Adjust total budget allocation to match K exactly
    allocated = sum(r[2] for r in result)
    diff = K - allocated
    if diff != 0:
        # Distribute remainder to largest blocks (or take from smallest)
        sorted_idx = sorted(range(len(result)), key=lambda i: len(result[i][3]), reverse=True)
        for i in sorted_idx:
            if diff == 0:
                break
            a_b, b_b, k_b, idx_b = result[i]
            if diff > 0 and k_b < len(idx_b) - 1:
                result[i] = (a_b, b_b, k_b + 1, idx_b)
                diff -= 1
            elif diff < 0 and k_b > 1:
                result[i] = (a_b, b_b, k_b - 1, idx_b)
                diff += 1

    return result


def recompose(
    block_solutions: List[np.ndarray],
    block_indices: List[np.ndarray],
    N: int,
) -> np.ndarray:
    """
    Merge block-level binary solutions into a single global solution vector.

    Args:
        block_solutions: List of (n_block,) binary vectors, one per block.
        block_indices: List of (n_block,) arrays of original indices.
        N: Total number of items.

    Returns:
        x: (N,) binary vector representing the global selection.
    """
    x = np.zeros(N, dtype=np.int32)
    for sol, indices in zip(block_solutions, block_indices):
        x[indices] = sol.astype(np.int32)
    return x
