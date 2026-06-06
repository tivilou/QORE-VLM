"""
QORECache: Drop-in replacement for DynamicCache with QUBO-optimized eviction.

Usage:
    from applications.kv_cache import QORECache

    cache = QORECache(max_capacity=1024, trigger_every=128)
    outputs = model.generate(input_ids, past_key_values=cache, max_new_tokens=500)
"""

import torch
import numpy as np
from transformers.cache_utils import DynamicCache
from typing import Optional, Dict, Any, Tuple

from qore import solve as qore_solve
from qore.signals import cosine_redundancy, normalize
from .signals_kv import key_norm_quality, pairwise_key_similarity


class QORECache(DynamicCache):
    """
    KV cache with QUBO-optimized eviction policy.

    When the cache exceeds `max_capacity`, QORE selects which KV entries to
    keep by solving a QUBO that balances individual importance (attention-based
    quality) against pairwise redundancy (key vector similarity).

    This is a drop-in replacement for DynamicCache — no model changes needed.
    """

    def __init__(
        self,
        max_capacity: int = 1024,
        trigger_every: int = 128,
        num_sink_tokens: int = 4,
        num_layers: Optional[int] = None,
        solver_method: str = "anneal",
        num_reads: int = 30,
        lam: float = 2.0,
        redundancy_method: str = "cosine",
        score_layer: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            max_capacity: Maximum number of KV entries to retain after eviction.
            trigger_every: Minimum tokens between evictions (amortizes cost).
            num_sink_tokens: Number of initial "sink" tokens to always keep
                (attention sinks — first tokens get disproportionate attention).
            num_layers: Total number of transformer layers. If None, auto-detected
                after the second forward pass.
            solver_method: QORE solver ("anneal" or "greedy" for fast baseline).
            num_reads: SA reads per solve (more = better solution, slower).
            lam: QUBO penalty weight.
            redundancy_method: "cosine" or "rbf" for key similarity.
            score_layer: Which layer's keys to use for scoring (0 = first layer).
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.num_sink_tokens = num_sink_tokens
        self.solver_method = solver_method
        self.num_reads = num_reads
        self.lam = lam
        self.redundancy_method = redundancy_method
        self.score_layer = score_layer
        self.seed = seed

        self._tokens_since_eviction = 0
        self._attention_accumulator = None  # cumulative attention scores
        self._eviction_count = 0
        self._num_layers = num_layers  # None = auto-detect
        self._last_layer_idx = -1  # track forward pass progress

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new KV states, triggering eviction if needed.

        Eviction is triggered when:
        1. Cache size exceeds max_capacity, AND
        2. At least trigger_every tokens have been added since last eviction

        Eviction is synchronized: computed once (on score_layer), then the same
        keep_indices are applied to ALL layers.
        """
        # Standard DynamicCache append
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Track tokens added (only count on layer 0 to avoid multi-counting)
        if layer_idx == 0:
            self._tokens_since_eviction += key_states.shape[-2]

        # Detect total number of layers: when layer_idx wraps back to a value
        # <= last seen, we know the previous pass covered all layers.
        if self._num_layers is None:
            if layer_idx <= self._last_layer_idx and self._last_layer_idx > 0:
                # A new pass started → previous pass had _last_layer_idx + 1 layers
                self._num_layers = self._last_layer_idx + 1
        self._last_layer_idx = layer_idx

        # Check eviction condition (only trigger on the last layer to ensure
        # all layers have been updated before we evict)
        if (self._num_layers is not None and
                layer_idx == self._num_layers - 1 and
                self.get_seq_length() > self.max_capacity and
                self._tokens_since_eviction >= self.trigger_every):
            self._evict()
            self._tokens_since_eviction = 0
            self._eviction_count += 1

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict(self):
        """
        Run QORE eviction: select which KV entries to keep across all layers.

        Strategy:
        1. Always keep the first `num_sink_tokens` (attention sinks)
        2. From the remaining entries, use QORE to select `max_capacity - num_sink_tokens`
        3. Apply the same keep_indices to ALL layers
        """
        seq_len = self.get_seq_length()
        if seq_len <= self.max_capacity:
            return

        n_sink = min(self.num_sink_tokens, seq_len)
        n_candidates = seq_len - n_sink
        n_keep = self.max_capacity - n_sink

        if n_keep >= n_candidates:
            return  # nothing to evict

        # Get key states from the scoring layer for signal construction
        # Shape: [batch, num_heads, seq_len, head_dim]
        score_layer = min(self.score_layer, len(self.key_cache) - 1)
        keys = self.key_cache[score_layer]

        # Work with first batch element (batch_size=1 for generation)
        # Average across heads for a single importance vector
        keys_2d = keys[0, :, n_sink:, :]  # [num_heads, n_candidates, head_dim]

        # Build quality signal: key norm as importance proxy
        a = key_norm_quality(keys_2d)

        # Build redundancy signal from a subset of heads (for speed)
        b = pairwise_key_similarity(
            keys_2d, method=self.redundancy_method, max_heads=4
        )

        # Solve QUBO
        a_np = a.cpu().numpy().astype(np.float64)
        b_np = b.cpu().numpy().astype(np.float64)
        a_np = normalize(a_np)

        # Use block decomposition for large candidate pools
        if n_candidates > 64:
            keep_indices = self._solve_with_blocks(a_np, b_np, n_keep)
        else:
            x = qore_solve(
                a_np, b_np, n_keep,
                lam=self.lam,
                method=self.solver_method,
                num_reads=self.num_reads,
                seed=self.seed,
            )
            keep_indices = np.where(x == 1)[0]

        # Convert to absolute positions (offset by sink tokens)
        keep_positions = np.concatenate([
            np.arange(n_sink),  # always keep sinks
            keep_indices + n_sink,  # QORE-selected positions
        ])
        keep_positions = np.sort(keep_positions)
        keep_tensor = torch.tensor(keep_positions, dtype=torch.long, device=keys.device)

        # Apply eviction to ALL layers
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_tensor, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_tensor, :]

    def _solve_with_blocks(
        self, a: np.ndarray, b: np.ndarray, n_keep: int
    ) -> np.ndarray:
        """Solve large problems via block decomposition."""
        from qore.block_decompose import decompose, recompose

        N = len(a)
        num_blocks = max(2, N // 32)

        blocks = decompose(a, b, n_keep, num_blocks=num_blocks)

        solutions = []
        indices_list = []
        for a_block, b_block, k_block, block_indices in blocks:
            x_block = qore_solve(
                a_block, b_block, k_block,
                lam=self.lam,
                method=self.solver_method,
                num_reads=self.num_reads,
                seed=self.seed,
            )
            solutions.append(x_block)
            indices_list.append(block_indices)

        x_global = recompose(solutions, indices_list, N)
        return np.where(x_global == 1)[0]

    @property
    def eviction_count(self) -> int:
        """Number of evictions performed so far."""
        return self._eviction_count
