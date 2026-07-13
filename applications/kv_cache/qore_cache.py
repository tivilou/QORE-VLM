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
from .attention_accumulator import AttentionAccumulatorMixin, assert_single_batch


class QORECache(AttentionAccumulatorMixin, DynamicCache):
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
        quality: str = "attention",
        seed: Optional[int] = None,
        qaoa_p: int = 2,
        qaoa_maxiter: int = 50,
        quantum_backend: str = "tensorcircuit",
        quantum_n_qubits: int = 6,
        quantum_n_layers: int = 2,
        vqc_encoder=None,
    ):
        """
        Args:
            max_capacity: Maximum number of KV entries to retain after eviction.
            trigger_every: Minimum tokens between evictions (amortizes cost).
            num_sink_tokens: Number of initial "sink" tokens to always keep
                (attention sinks — first tokens get disproportionate attention).
            num_layers: Total number of transformer layers. If None, auto-detected
                after the second forward pass.
            solver_method: QORE solver. "anneal" (default) or "greedy" for the
                fast baselines; "qaoa_tc"/"qaoa_pl"/"qaoa_qk" for QAOA on each
                sub-QUBO (ablation — small blocks only, slow).
            num_reads: SA reads per solve (more = better solution, slower).
            lam: QUBO penalty weight.
            redundancy_method: bᵢⱼ signal. "cosine"/"rbf" (classical key
                similarity) or "quantum" (fidelity quantum kernel — ablation).
            score_layer: Which layer's keys to use for scoring (0 = first layer).
            qaoa_p: QAOA circuit depth p (only for solver_method="qaoa_*").
            qaoa_maxiter: QAOA classical optimizer iterations.
            quantum_backend: "tensorcircuit"/"pennylane"/"qiskit" for quantum
                kernel and VQC paths.
            quantum_n_qubits / quantum_n_layers: quantum circuit size for the
                quantum kernel and VQC encoder.
            vqc_encoder: optional pre-built/trained VQCEncoder. When provided
                (or quality="vqc"), the VQC produces BOTH aᵢ and bᵢⱼ from one
                circuit (fully-quantum KV pipeline, ablation).
            quality: Quality signal aᵢ source. "attention" (default) uses real
                cumulative attention captured via forward hooks (H2O heavy-hitter,
                matches the paper); requires AttentionCapture + eager attention.
                "keynorm" uses the key-vector L2 norm as a fast proxy (no hooks).
                If "attention" is set but no attention was captured, falls back
                to key norm automatically.
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
        self.quality = quality
        self.seed = seed
        self.qaoa_p = qaoa_p
        self.qaoa_maxiter = qaoa_maxiter
        self.quantum_backend = quantum_backend
        self.quantum_n_qubits = quantum_n_qubits
        self.quantum_n_layers = quantum_n_layers
        self.vqc_encoder = vqc_encoder

        self._init_attention_state()
        self._tokens_since_eviction = 0
        self._eviction_count = 0
        self._num_layers = num_layers  # None = auto-detect

        # Track the original absolute position of each slot. Each element is a
        # [batch=1, seq] or [seq] tensor holding the absolute positions baked
        # into the RoPE phase of the cached keys. After eviction the retained
        # positions are compacted. This enables RoPE re-rotation: keys enter the
        # cache pre-RoPE-encoded, so after physical eviction we must re-rotate
        # each retained key from its original phase to its new compact position
        # (physical slot == phase, fixing the query-key relative geometry).
        self.original_positions: List[Optional[torch.Tensor]] = []

        # Lazy-init: fetched on first eviction from the model's rotary_emb.
        self._inv_freq: Optional[torch.Tensor] = None
        self._last_layer_idx = -1  # track forward pass progress

    def set_model(self, model):
        """Fetch RoPE inv_freq from the model for key re-rotation on eviction.

        Call this once before generation if you want correct post-eviction RoPE
        phase. Without it, keys retain their baked absolute phase and query-key
        relative geometry breaks after eviction (degenerate generation).
        """
        from .rope_utils import get_inv_freq
        self._inv_freq = get_inv_freq(model)
        if self._inv_freq is None:
            import warnings
            warnings.warn(
                "Could not extract inv_freq from model. RoPE re-rotation disabled; "
                "post-eviction generation may degrade due to position mismatch.",
                stacklevel=2,
            )

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

        # Track absolute positions: keys entering the cache were RoPE-encoded at
        # positions [L_before, L_before+1, ..., L_before+n_new-1], where
        # L_before = this layer's seq_len before this update. After eviction we
        # re-rotate the retained keys to compact positions, so we need to remember
        # each slot's original absolute phase. Expand position tracking to match
        # num layers lazily (we may not know num_layers at first).
        while len(self.original_positions) <= layer_idx:
            self.original_positions.append(None)
        n_new = key_states.shape[-2]
        L_before = keys.shape[-2] - n_new  # seq_len before this append
        new_pos = torch.arange(L_before, L_before + n_new, device=keys.device)
        if self.original_positions[layer_idx] is None:
            self.original_positions[layer_idx] = new_pos
        else:
            self.original_positions[layer_idx] = torch.cat(
                [self.original_positions[layer_idx], new_pos], dim=0
            )

        # Track tokens added (only count on layer 0 to avoid multi-counting)
        if layer_idx == 0:
            self._tokens_since_eviction += n_new

        # Detect total number of layers: when layer_idx wraps back to a value
        # <= last seen, we know the previous pass covered all layers.
        if self._num_layers is None:
            if layer_idx <= self._last_layer_idx and self._last_layer_idx > 0:
                # A new pass started → previous pass had _last_layer_idx + 1 layers
                self._num_layers = self._last_layer_idx + 1
        self._last_layer_idx = layer_idx

        # Check eviction condition. Constraints:
        # - last layer only: all layers have appended this step before we evict
        # - decode only (query length == 1): during PREFILL the attention for
        #   this layer runs AFTER update() returns, over a full-length query;
        #   truncating the KV here would make attn_weights (q=prompt_len) and
        #   keys (kv=max_capacity) mismatch. Standard KV-eviction compresses
        #   after prefill, on the first decode step. key_states.shape[-2] is the
        #   number of new tokens: prompt_len during prefill, 1 during decode.
        is_decode_step = key_states.shape[-2] == 1
        if (self._num_layers is not None and
                layer_idx == self._num_layers - 1 and
                is_decode_step and
                self.get_seq_length() > self.max_capacity and
                self._tokens_since_eviction >= self.trigger_every):
            # The model built this step's causal mask from the pre-eviction
            # length, and the layers before this one already attended over the
            # full KV. Snapshot this layer's full tensors to return for its own
            # attention (mask-consistent), then evict — the truncation takes
            # effect on the NEXT forward, whose mask is rebuilt from the new
            # (smaller) get_seq_length().
            keys_ret = self.key_cache[layer_idx]
            values_ret = self.value_cache[layer_idx]
            self._evict()
            self._tokens_since_eviction = 0
            self._eviction_count += 1
            return keys_ret, values_ret

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

        # Degenerate budget: capacity <= sink tokens leaves no room for QUBO
        # selection (K would be <= 0, which the solver rejects). Keep only the
        # first max_capacity positions (sinks take priority) — matches how the
        # H2O/Window baselines behave in this configuration. Guarding here
        # avoids a ValueError from build_qubo_matrix (K must be in [1, N-1]).
        if n_keep <= 0:
            keep_tensor = torch.arange(
                min(self.max_capacity, seq_len),
                dtype=torch.long, device=self.key_cache[0].device,
            )
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_tensor, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_tensor, :]
            self.prune_attention(keep_tensor)
            return

        # Get key states from the scoring layer for signal construction
        # Shape: [batch, num_heads, seq_len, head_dim]
        score_layer = min(self.score_layer, len(self.key_cache) - 1)
        keys = self.key_cache[score_layer]

        # Eviction uses batch-0 signals for all rows — valid only at batch 1.
        assert_single_batch(self.key_cache, "QORECache")

        # Work with first batch element (batch_size=1 for generation)
        # Average across heads for a single importance vector
        keys_2d = keys[0, :, n_sink:, :]  # [num_heads, n_candidates, head_dim]

        # VQC ablation: one circuit yields BOTH aᵢ and bᵢⱼ, so it can't be split
        # into the quality-then-per-block-redundancy path. It's a small-N ablation
        # (fidelity over all candidates), so solve the whole pool directly.
        if self.quality == "vqc":
            a_np, b_np = self._vqc_signals(keys_2d)
            x = qore_solve(a_np, b_np, n_keep, lam=self.lam,
                           method=self.solver_method, **self._solver_kwargs())
            keep_indices = np.where(x == 1)[0]
            keep_positions = np.sort(np.concatenate([
                np.arange(n_sink), keep_indices + n_sink]))
            keep_tensor = torch.tensor(keep_positions, dtype=torch.long, device=keys.device)
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_tensor, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_tensor, :]
            self.prune_attention(keep_tensor)
            return

        # Quality signal aᵢ is O(N) and always built over all candidates.
        a_np = self._build_quality(keys_2d, n_sink, n_candidates)

        # Choose the direct-vs-block threshold by solver. QAOA simulates one
        # qubit per candidate, so 2^n_candidates state — anything beyond ~16
        # qubits is intractable. Force block decomposition (small blocks) for
        # QAOA regardless of pool size; SA scales fine to the full 64.
        direct_threshold = self._max_direct_candidates()
        if n_candidates > direct_threshold:
            # Large pool: decompose by quality, then compute redundancy ONLY
            # within each block (O(sum block^2) << O(N^2)). Avoids ever
            # materializing the global N×N similarity matrix — the dominant
            # cost that made QORE ~7x slower than full at long context.
            keep_indices = self._solve_with_blocks(a_np, keys_2d, n_keep)
        else:
            # Small pool: one solve; redundancy over the whole (small) pool.
            b_np = self._build_redundancy(keys_2d)
            x = qore_solve(
                a_np, b_np, n_keep,
                lam=self.lam,
                method=self.solver_method,
                **self._solver_kwargs(),
            )
            keep_indices = np.where(x == 1)[0]

        # Convert to absolute positions (offset by sink tokens)
        keep_positions = np.concatenate([
            np.arange(n_sink),  # always keep sinks
            keep_indices + n_sink,  # QORE-selected positions
        ])
        keep_positions = np.sort(keep_positions)
        keep_tensor = torch.tensor(keep_positions, dtype=torch.long, device=keys.device)

        # Record for bug 7.2 fix: the last layer's post-hook fires AFTER _evict
        # but carries pre-eviction-length attention. add_attention will map it.
        self._last_keep_indices = keep_tensor.clone()

        # Apply eviction to ALL layers
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_tensor, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_tensor, :]

        # Re-rotate retained keys: they were RoPE-encoded at their ORIGINAL
        # absolute positions (baked phase), but now sit in compact slots 0,1,2,..
        # We must shift their phase to match the compact position so Query's
        # monotonic position aligns with Key's phase (physical slot == phase).
        # This fixes the query-key relative geometry RoPE encodes.
        if self._inv_freq is None:
            # Lazy-fetch once from the model. We're inside an attention forward,
            # so the model is accessible via the keys' device or from the caller's
            # context. Best-effort: try standard layouts (model-level / per-attn).
            # If it fails, re-rotation is skipped (fallback to broken positions,
            # same as before this fix — a warning would alert, but we can't inject
            # model here cleanly; the symptom is degenerate generation post-eviction).
            from .rope_utils import get_inv_freq
            # The keys are on the model's device; we can't directly access the model
            # here, but the typical usage is generate_with_eviction which holds the
            # model. For now, store a model ref on first update or accept it as an
            # init param. Temporary: assume inv_freq can be inferred from keys.device
            # and a standard RoPE formula. Let's make a method to receive it.
            pass  # Will be set externally via set_model() or passed in __init__

        if self._inv_freq is not None:
            from .rope_utils import rerotate_keys
            for layer_idx in range(len(self.key_cache)):
                k = self.key_cache[layer_idx]  # [batch=1, heads, seq, dim]
                orig_pos = self.original_positions[layer_idx][keep_positions]
                new_pos = torch.arange(len(keep_positions), device=k.device)
                # rerotate_keys expects [heads, seq, dim]; k is [1, heads, seq, dim]
                k_rotated = rerotate_keys(k[0], self._inv_freq, orig_pos, new_pos)
                self.key_cache[layer_idx] = k_rotated.unsqueeze(0)
                # Update position tracking: compacted keys now sit at 0,1,2,...
                self.original_positions[layer_idx] = new_pos

        # Keep the attention accumulator aligned with the retained positions.
        self.prune_attention(keep_tensor)

    def _max_direct_candidates(self) -> int:
        """Largest candidate pool solved directly (no block decomposition).

        QAOA simulates 2^n amplitudes (one qubit per candidate), so it must run
        on small blocks; SA handles the full pool. Return the per-solve size cap.
        """
        if self.solver_method.startswith("qaoa"):
            return 14  # 2^14 statevector — the practical QAOA simulation ceiling
        return 64

    def _block_size(self) -> int:
        """Target sub-problem size for block decomposition, solver-aware.

        Must not exceed _max_direct_candidates(): for QAOA every candidate in a
        block is a qubit, so an oversized block reintroduces the 2^N blow-up the
        block decomposition exists to prevent.
        """
        return 12 if self.solver_method.startswith("qaoa") else 32

    def _solve_with_blocks(
        self, a: np.ndarray, keys_2d, n_keep: int
    ) -> np.ndarray:
        """Solve large problems via block decomposition.

        Redundancy is built PER BLOCK from that block's own key vectors — the
        global N×N similarity matrix is never materialized. Budget allocation in
        decompose() is quality-only, so passing b=None is correct; each block's
        b is computed lazily just before its solve.
        """
        from qore.block_decompose import decompose, recompose

        N = len(a)
        # ceil division: guarantees the LARGEST block <= block_size. Using
        # floor (N // block_size) undersizes num_blocks so np.array_split makes
        # blocks of ceil(N/num_blocks) > block_size — which for QAOA silently
        # exceeds the qubit ceiling (e.g. N=35, size=12 -> 2 blocks of 18).
        block_size = self._block_size()
        num_blocks = max(2, -(-N // block_size))  # ceil(N / block_size)

        # b=None: decompose partitions by quality only; we fill redundancy below.
        blocks = decompose(a, None, n_keep, num_blocks=num_blocks)

        solutions = []
        indices_list = []
        for a_block, _b_unused, k_block, block_indices in blocks:
            block_n = len(block_indices)
            if k_block >= block_n:
                # Retain the whole block: the QUBO solver requires K <= N-1, so
                # a full-keep block must bypass it (selecting all is trivial).
                solutions.append(np.ones(block_n, dtype=np.int32))
                indices_list.append(block_indices)
                continue
            # Redundancy for THIS block only: slice its key vectors (contiguous
            # candidate positions) and build the small block_n × block_n matrix.
            block_keys = keys_2d[:, block_indices, :]  # [heads, block_n, dim]
            b_block = self._build_redundancy(block_keys)
            x_block = qore_solve(
                a_block, b_block, k_block,
                lam=self.lam,
                method=self.solver_method,
                **self._solver_kwargs(),
            )
            solutions.append(x_block)
            indices_list.append(block_indices)

        x_global = recompose(solutions, indices_list, N)
        return np.where(x_global == 1)[0]

    def _solver_kwargs(self) -> dict:
        """Assemble solver-specific kwargs for qore_solve.

        Anneal takes num_reads; QAOA variants take p/maxiter. Passing num_reads
        to a QAOA solver (or p to anneal) would raise, so we route by method.
        """
        if self.solver_method.startswith("qaoa"):
            kwargs = {"p": self.qaoa_p, "maxiter": self.qaoa_maxiter}
        elif self.solver_method == "greedy":
            kwargs = {}
        else:  # anneal (default)
            kwargs = {"num_reads": self.num_reads}
        if self.seed is not None and self.solver_method != "greedy":
            kwargs["seed"] = self.seed
        return kwargs

    def _build_quality(self, keys_2d, n_sink: int, n_candidates: int) -> np.ndarray:
        """Quality signal aᵢ over the candidate positions (O(N), no pairwise).

        Real cumulative attention when available (H2O heavy-hitter, the paper's
        signal); key-norm proxy as fallback / when quality="keynorm".
        """
        if self.quality == "attention" and self.has_attention():
            attn = self.attention_scores()[n_sink:n_sink + n_candidates]
            a = attn.to(keys_2d.device)
            if a.shape[0] != n_candidates:
                a = key_norm_quality(keys_2d)
        else:
            a = key_norm_quality(keys_2d)
        return normalize(a.cpu().numpy().astype(np.float64))

    def _build_redundancy(self, keys_2d) -> np.ndarray:
        """Redundancy matrix bᵢⱼ for the given key vectors.

        Called per-block by _solve_with_blocks (so its cost is O(block^2), never
        O(N^2)), or over the whole pool when N is small enough to solve directly.
        """
        if self.redundancy_method == "quantum":
            return self._quantum_kernel_redundancy(keys_2d)
        b = pairwise_key_similarity(
            keys_2d, method=self.redundancy_method, max_heads=4
        )
        return b.cpu().numpy().astype(np.float64)

    def _key_features(self, keys_2d) -> np.ndarray:
        """Mean-over-heads key vectors as (n_candidates, head_dim) features."""
        return keys_2d.mean(dim=0).cpu().numpy().astype(np.float64)

    def _quantum_kernel_redundancy(self, keys_2d) -> np.ndarray:
        """bᵢⱼ = fidelity quantum kernel over key vectors (ablation)."""
        from qore.kernels import quantum_kernel
        features = self._key_features(keys_2d)
        return quantum_kernel(
            features,
            backend=self.quantum_backend,
            n_qubits=self.quantum_n_qubits,
            n_layers=self.quantum_n_layers,
        )

    def _vqc_signals(self, keys_2d):
        """VQC encoder produces both aᵢ and bᵢⱼ from one circuit (ablation)."""
        from qore.vqc.encoder import VQCEncoder
        if self.vqc_encoder is None:
            self.vqc_encoder = VQCEncoder(
                n_qubits=self.quantum_n_qubits,
                n_layers=self.quantum_n_layers,
                backend=self.quantum_backend,
                seed=self.seed,
            )
        features = self._key_features(keys_2d)
        signals = self.vqc_encoder.encode_and_measure(features)
        a_np = normalize(np.asarray(signals["quality"], dtype=np.float64))
        b_np = np.asarray(signals["redundancy"], dtype=np.float64)
        np.fill_diagonal(b_np, 0.0)
        np.clip(b_np, 0.0, 1.0, out=b_np)
        return a_np, b_np

    @property
    def eviction_count(self) -> int:
        """Number of evictions performed so far."""
        return self._eviction_count
