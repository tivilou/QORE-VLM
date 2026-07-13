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
    # b may be None: budget allocation below is purely quality-based, so the
    # redundancy matrix is never needed to DECIDE the split. Passing b=None lets
    # the caller compute each block's redundancy lazily from that block's own
    # features, avoiding a global O(N^2) dense matrix (the dominant cost at long
    # context). When b is provided we still slice per-block submatrices as before.
    b = np.asarray(b, dtype=np.float64) if b is not None else None
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
        b_block = b[np.ix_(indices, indices)] if b is not None else None
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

        # Clamp K_block to [1, block size]. A block MAY keep all its members
        # (K_block == len(indices)) — that is "retain this whole block", handled
        # by the caller without invoking the QUBO solver (which requires
        # K <= N-1). Clamping to len(indices)-1 here was a bug: it capped the
        # total reachable budget at N - num_blocks, so when the caller wanted to
        # drop fewer than num_blocks items the allocation could never sum to K
        # and eviction silently under-kept (e.g. N=509, K=508 -> only 481 kept).
        K_block = min(K_block, len(indices))
        K_block = max(1, K_block)

        result.append((a_block, b_block, K_block, indices))

    # Adjust total budget allocation to match K exactly. This MUST be multi-pass:
    # a single sweep adds at most 1 per block, but the rounding gap can exceed the
    # number of blocks with spare room (e.g. gap 18 spread over 7 blocks each with
    # 1-4 free slots). A single pass would leave the budget short and eviction
    # would under-keep. We repeat sweeps until the gap closes or no block can
    # absorb more. Growth is allowed up to len(idx_b) (a full block, handled
    # solver-free by the caller); shrink stays >= 1 so no block empties.
    sorted_idx = sorted(range(len(result)), key=lambda i: len(result[i][3]), reverse=True)
    diff = K - sum(r[2] for r in result)
    while diff != 0:
        progressed = False
        for i in sorted_idx:
            if diff == 0:
                break
            a_b, b_b, k_b, idx_b = result[i]
            if diff > 0 and k_b < len(idx_b):
                result[i] = (a_b, b_b, k_b + 1, idx_b)
                diff -= 1
                progressed = True
            elif diff < 0 and k_b > 1:
                result[i] = (a_b, b_b, k_b - 1, idx_b)
                diff += 1
                progressed = True
        if not progressed:
            # No block can absorb the remaining gap (K infeasible for this
            # partition, e.g. K > N or K < num_blocks). Leave as-is.
            break

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
