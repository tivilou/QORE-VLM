# Phase 2.5: HuggingFace Cache API Technical Note

> Investigation of how to integrate QORE KV-cache eviction into HuggingFace
> transformers **without** copying generation_utils.py (unlike DUET-VLM's approach).

## Key Finding

**We can subclass `DynamicCache` and override `update()`.** No need to modify
`generate()` or copy any HuggingFace internals. The cache object is passed
directly as `past_key_values` to the model.

## HF Cache Architecture (transformers 4.44.0)

### Class Hierarchy
```
Cache (base, nn.Module)
├── DynamicCache          ← default, grows unbounded
├── StaticCache           ← pre-allocated, no eviction
├── SinkCache             ← attention sinks + sliding window eviction
├── SlidingWindowCache    ← fixed window, positional eviction
├── QuantizedCache        ← quantized KV for memory saving
├── HybridCache           ← combines static + sliding
└── OffloadedCache        ← CPU offloading
```

### Key Interface: `update()`
```python
def update(self, key_states, value_states, layer_idx, cache_kwargs=None)
    -> (full_keys, full_values)
```
Called once per layer per generation step. Returns the FULL key/value tensors
(including history) that the attention mechanism uses.

### How `generate()` uses it
1. If `generation_config.cache_implementation` is set → uses registered class from
   `NEED_SETUP_CACHE_CLASSES_MAPPING`
2. Otherwise → creates `DynamicCache()` instance
3. Cache is passed as `past_key_values` to `model.forward()` on every step
4. **We can pass our own cache object directly** via `model.generate(past_key_values=our_cache)`

## Our Strategy: `QORECache(DynamicCache)`

```python
class QORECache(DynamicCache):
    def __init__(self, max_capacity: int, trigger_every: int = 128, ...):
        super().__init__()
        self.max_capacity = max_capacity
        self.trigger_every = trigger_every
        self.steps_since_eviction = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1. Normal append (same as DynamicCache)
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 2. Check if eviction needed (only on layer 0 to avoid per-layer overhead)
        if layer_idx == 0:
            self.steps_since_eviction += key_states.shape[-2]

        # 3. Trigger eviction when capacity exceeded and interval reached
        if (layer_idx == 0 and
            self.get_seq_length() > self.max_capacity and
            self.steps_since_eviction >= self.trigger_every):
            self._evict()
            self.steps_since_eviction = 0

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict(self):
        """Run QORE eviction across all layers."""
        # Build quality signal from attention patterns (accumulated)
        # Build redundancy from key vector similarity
        # Solve QUBO → get keep_indices
        # Slice all layers' caches to keep_indices
        pass
```

## Critical Details

### 1. Eviction must be synchronized across layers
When we evict, we must remove the SAME positions from ALL layers' caches.
The QUBO can be solved based on one layer's keys (e.g., layer 0 or middle layer),
then the same `keep_indices` applied to all layers.

### 2. Attention score accumulation
For the quality signal `a_i`, we need cumulative attention scores. Options:
- **Hook-based**: Register a forward hook on attention layers to accumulate scores
- **Approximation**: Use key norm as a proxy (cheaper, no hooks needed)
- **H2O-style**: The model can be patched to expose attention weights cheaply

For Phase 3, we'll start with key-norm proxy (simplest, no model surgery).

### 3. RoPE position handling
SinkCache re-rotates keys when evicting because positional encoding is baked into
key vectors. Two approaches:
- **Ignore**: Some recent work (H2O, SnapKV) shows that keeping already-rotated keys
  at wrong positions still works reasonably. Start here.
- **Re-rotate**: If needed, copy SinkCache's `_apply_key_rotary_pos_emb` logic.
  More correct but adds complexity.

### 4. Redundancy computation cost
Computing pairwise cosine similarity on key vectors: O(N² × d).
For N=2048, d=128 (head_dim): 2048² × 128 = 537M ops. On GPU this is fast (~1ms).
Only triggered every T=128 steps, so amortized cost is negligible.

### 5. Block decomposition: per-head
Each attention head has independent KV cache. Natural decomposition:
- Solve one QUBO per head (n = cache_size, parallelizable)
- Or solve one shared QUBO using averaged key similarity across heads

## Comparison with DUET-VLM's Approach

| Aspect | DUET-VLM | QORE (planned) |
|--------|----------|----------------|
| Integration point | Copied entire generation_utils.py (4673 lines) | Subclass DynamicCache (~100 lines) |
| When eviction happens | Mid-forward (between specific layers) | In cache.update() (between generation steps) |
| What's evicted | Visual tokens only (by position) | Any KV entry (by importance) |
| Model modification needed | Yes (custom pdrop_forward) | No (just pass custom cache) |
| Score computation | Uses next-layer Q/K projection | Uses accumulated attention or key norms |
| Compatibility | LLaMA only (custom forward) | Any model using HF Cache interface |

## Conclusion

The HF Cache API is well-designed for our use case. We can implement QORE KV-cache
eviction as a **drop-in replacement** for DynamicCache with zero changes to the model
or generation code. This is a major advantage over DUET-VLM's approach.

**Estimated implementation effort for Phase 3**: ~200 lines for `QORECache`, plus
~100 lines for signal extraction (attention hooks or key-norm proxy).

## Files to Create in Phase 3
```
applications/kv_cache/
├── __init__.py
├── qore_cache.py       # QORECache(DynamicCache) with eviction logic
├── signals_kv.py       # Quality (attention accumulator) + Redundancy (key similarity)
├── config.py           # Capacity, trigger interval, solver params
├── baselines/
│   ├── h2o_cache.py    # H2O: heavy-hitter eviction
│   ├── random_cache.py # Random eviction baseline
│   └── window_cache.py # Sliding window (like SinkCache)
└── tests/
    ├── test_cache.py   # Unit tests
    └── demo_long.py    # Demo on a long text generation task
```
