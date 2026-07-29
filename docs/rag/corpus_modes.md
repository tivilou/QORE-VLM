# RAG Corpus Modes

## Overview

The RAG evaluation system supports four interchangeable corpus modes, each solving the same problem — "give each query a candidate passage pool that provably contains its gold evidence" — with different cost/fidelity tradeoffs.

All modes expose the same interface (`CorpusManager`), so switching modes is a config change, not a code change.

---

## Mode Comparison

| Mode | Corpus Size | Gold Coverage | Realism | Setup Cost | Use Case |
|------|-------------|---------------|---------|------------|----------|
| **aligned** | ~40K (tunable) | ✅ Guaranteed | Low | Medium | Quick experiments |
| **wiki_dpr** | 21M (full) | ⚠️ Probabilistic | High | Low | Production evaluation |
| **faiss** | 21M (full) | ⚠️ Probabilistic | High | High | Offline large-scale |
| **precomputed** | Dataset-specific | ⚠️ Dataset-dependent | Medium | None | Benchmark reproduction |

---

## Mode Details

### 1. Aligned Mode

**Concept**: Build a controlled pool of (all gold passages) + random distractors.

**Characteristics**:
- **Corpus**: Gold passages from questions + random distractors (default 36,000)
- **Total size**: ~40K passages (200 questions × ~5 gold + 36K distractors)
- **Gold coverage**: 100% guaranteed — every question's gold is in the pool
- **Realism**: Low — distractors are randomly sampled, not retrieval-based

**Configuration**:
```python
config = {
    "output_dir": "path/to/cache",      # Cache directory
    "n_distractors": 36000,              # Number of random distractors
    "wiki_dpr_config": "psgs_w100.nq.compressed",  # Source for distractors
    "seed": 42,                          # RNG seed for reproducibility
    "rebuild": False,                    # Set True to ignore cache
}
```

**Use Case**: 
- Quick parameter tuning experiments
- Isolating selection algorithm performance (no retrieval noise)
- Resource-constrained environments

**Trade-offs**:
- ✅ Fast iteration (corpus cached after first build)
- ✅ Guaranteed answerable (all gold present)
- ✅ Small corpus (~40K vs 21M)
- ❌ Not realistic (random negatives, not hard negatives)
- ⚠️ First build is slow (streams 108K passages from wiki_dpr)

---

### 2. Wiki_dpr Mode

**Concept**: Use HuggingFace's pre-built FAISS index over the full 21M Wikipedia corpus.

**Characteristics**:
- **Corpus**: Full 21M passages from facebook/wiki_dpr
- **Index**: Compressed IVFPQ FAISS index (loaded from HF datasets)
- **Gold coverage**: Probabilistic (depends on retrieval quality)
- **Realism**: High — real retrieval over full corpus

**Configuration**:
```python
config = {
    "wiki_dpr_config": "psgs_w100.nq.compressed",  # HF dataset config
    "wiki_dpr_cache_dir": "/path/to/cache",         # HF cache directory
    "wiki_dpr_nprobe": 64,                          # FAISS nprobe parameter
}
```

**Use Case**:
- Final evaluation for publication
- Realistic end-to-end RAG benchmarking
- When you need full-corpus retrieval but don't want to manage FAISS yourself

**Trade-offs**:
- ✅ Most realistic (full corpus + FAISS retrieval)
- ✅ No manual FAISS index building
- ✅ Stable and well-tested
- ✅ Fast after initial HF download
- ⚠️ First run downloads ~75GB from HuggingFace
- ⚠️ Higher memory usage (~few GB RAM for index)

---

### 3. FAISS Mode

**Concept**: Build and query your own FAISS index over pre-computed embeddings.

**Characteristics**:
- **Corpus**: Full 21M passages
- **Index**: Custom FAISS index (you control the parameters)
- **Gold coverage**: Probabilistic (depends on retrieval quality)
- **Realism**: High — real retrieval over full corpus

**Configuration**:
```python
config = {
    "faiss_embeddings_path": "path/to/embeddings.npy",  # (21M, 768) array
    "faiss_passages_path": "path/to/passages.pkl",      # List of 21M texts
    "faiss_mmap": True,                                  # Memory-map embeddings
}
```

**Use Case**:
- Custom FAISS index parameters (e.g., different quantization)
- Offline large-scale evaluation
- When you already have pre-computed embeddings

**Trade-offs**:
- ✅ Full control over index parameters
- ✅ Can use custom embeddings
- ❌ Requires pre-built embeddings (~65GB)
- ❌ Manual index building step
- ❌ Highest setup cost

---

### 4. Precomputed Mode

**Concept**: Use retrieval candidates pre-packaged with the dataset.

**Characteristics**:
- **Corpus**: Whatever the dataset provides (e.g., BM25 top-100)
- **Gold coverage**: Dataset-dependent
- **Realism**: Medium — depends on dataset's retrieval method

**Configuration**:
```python
config = {
    "custom_path": "path/to/dataset.json",  # Dataset with pre-computed candidates
}
```

**Use Case**:
- Reproducing published benchmark results
- When the dataset ships with DPR/BM25 top-k
- Fastest option (no corpus building)

**Trade-offs**:
- ✅ Fastest (no corpus building needed)
- ✅ Exactly reproduces benchmark conditions
- ⚠️ Gold coverage depends on dataset
- ⚠️ Not all datasets provide precomputed candidates

---

## Choosing a Mode

### For Quick Experiments / Parameter Tuning
→ **aligned** mode
- Fast iteration after first build
- Guaranteed gold coverage
- Good for QORE algorithm development

### For Final Evaluation / Publication
→ **wiki_dpr** mode
- Most realistic
- Full corpus retrieval
- Standard benchmark

### For Custom Embeddings / Offline Evaluation
→ **faiss** mode
- Full control
- Highest setup cost
- Best for research experiments

### For Benchmark Reproduction
→ **precomputed** mode
- Fastest
- Exact reproduction
- Only if dataset provides candidates

---

## Implementation

All modes implement the `CorpusManager` interface:

```python
class CorpusManager(ABC):
    def build(self, questions: list[dict]) -> Corpus:
        """Build/load the corpus for these questions."""
        
    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> tuple:
        """Retrieve (indices, scores) of top_k passages."""
        
    def gold_mapping(self) -> dict:
        """Return question_id -> gold corpus indices mapping."""
```

**Usage**:
```python
from applications.rag.data import make_corpus_manager

# Create corpus manager
manager = make_corpus_manager("wiki_dpr", config)

# Build corpus
corpus = manager.build(questions)

# Retrieve for a query
indices, scores = manager.retrieve(query_emb, top_k=50)

# Check gold coverage
gold_indices = corpus.gold_for(question_id)
```

---

## Common Issues

### Aligned mode is slow
- **Cause**: Streams 108K passages from wiki_dpr on first build
- **Solution**: Use cache (set `output_dir`), or switch to wiki_dpr mode

### Wiki_dpr downloads large files
- **Cause**: First run downloads ~75GB from HuggingFace
- **Solution**: Pre-download with `datasets-cli download facebook/wiki_dpr`

### FAISS mode fails to load
- **Cause**: Missing embeddings.npy or passages.pkl
- **Solution**: Run `scripts/rag/build_faiss_corpus.py` first

### Precomputed mode: "no candidates field"
- **Cause**: Dataset doesn't provide precomputed candidates
- **Solution**: Use aligned or wiki_dpr mode instead

---

## See Also

- `applications/rag/data/corpus_manager.py` - Base interface
- `applications/rag/data/aligned.py` - Aligned mode implementation
- `applications/rag/data/wiki_dpr_corpus.py` - Wiki_dpr mode implementation
- `scripts/rag/eval_rag_refactored.py` - Usage example
