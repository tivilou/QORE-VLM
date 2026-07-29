# RAG Troubleshooting Guide

This guide covers common issues when running RAG experiments and how to resolve them.

---

## Table of Contents

1. [Corpus Building Issues](#corpus-building-issues)
2. [Module Import Errors](#module-import-errors)
3. [Parameter Errors](#parameter-errors)
4. [Memory Issues](#memory-issues)
5. [Performance Issues](#performance-issues)

---

## Corpus Building Issues

### Aligned mode appears frozen

**Symptoms**:
```
Building corpus (mode=aligned)...
[No further output, CPU ~0%]
```

**Cause**: Aligned mode streams 108K passages from wiki_dpr, which is slow, and has no progress logging.

**Solutions**:

1. **Switch to wiki_dpr mode** (recommended for production):
   ```bash
   --corpus_mode wiki_dpr \
   --wiki_dpr_config psgs_w100.nq.compressed \
   --wiki_dpr_cache_dir /root/.cache/huggingface/datasets
   ```

2. **Wait it out**: First build takes ~10-30 minutes depending on network. Subsequent runs use cache.

3. **Use smaller n_distractors**:
   ```bash
   --corpus_mode aligned \
   --n_distractors 1000  # Instead of default 36000
   ```

**Fixed in**: Commit `1ab79b7` added progress logging to aligned mode.

---

### Wiki_dpr: "config not found"

**Symptoms**:
```
ValueError: Config 'psgs_w100.nq.exact' not found
```

**Cause**: Using wrong config name. The correct config is `compressed`, not `exact`.

**Solution**:
```bash
--wiki_dpr_config psgs_w100.nq.compressed  # ✓ Correct
--wiki_dpr_config psgs_w100.nq.exact       # ✗ Wrong
```

---

### Wiki_dpr: First run downloads large files

**Symptoms**:
```
Downloading data: 0.00/544M [00:00<?, ?B/s]
[Takes a long time]
```

**Cause**: First run downloads ~75GB of data from HuggingFace.

**Solution**: 
- **Normal behavior**. Subsequent runs use cache.
- Pre-download if needed:
  ```bash
  datasets-cli download facebook/wiki_dpr psgs_w100.nq.compressed
  ```

---

### FAISS mode: "embeddings.npy not found"

**Symptoms**:
```
FileNotFoundError: path/to/embeddings.npy
```

**Cause**: FAISS mode requires pre-built embeddings file.

**Solution**:
```bash
# Build embeddings first
python scripts/rag/build_faiss_corpus.py

# Then run evaluation
python -m scripts.rag.eval_rag_refactored --corpus_mode faiss ...
```

---

## Module Import Errors

### "No module named 'scripts'"

**Symptoms**:
```
ModuleNotFoundError: No module named 'scripts'
```

**Cause**: PYTHONPATH not set correctly.

**Solution**:
```bash
# Run from repository root
cd /path/to/QORE-VLM
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m scripts.rag.eval_rag_refactored ...
```

**Fixed in**: Commit `80ad6c7` updated `run_p2_experiments.sh` to auto-set PYTHONPATH.

---

### "No module named 'applications'"

**Symptoms**:
```
ModuleNotFoundError: No module named 'applications'
```

**Cause**: Not running from repository root.

**Solution**:
```bash
cd /path/to/QORE-VLM  # Go to repo root
python -m scripts.rag.eval_rag_refactored ...
```

---

## Parameter Errors

### "unrecognized arguments: --delta"

**Symptoms**:
```
error: unrecognized arguments: --delta
```

**Cause**: Using old version of `eval_rag_refactored.py`.

**Solution**:
```bash
git pull origin main  # Update to latest code
```

**Added in**: Commit `80ad6c7`

---

### "unrecognized arguments: --complementarity_method"

**Symptoms**:
```
error: unrecognized arguments: --complementarity_method
```

**Cause**: Using old version of `eval_rag_refactored.py`.

**Solution**:
```bash
git pull origin main  # Update to latest code
```

**Added in**: Commit `80ad6c7`

---

### "complementarity_method requires --use_answer_scorer"

**Symptoms**:
```
ValueError: complementarity_method='dpr' requires --use_answer_scorer
```

**Cause**: Using complementarity without answer scorer.

**Solution**:
```bash
--complementarity_method dpr \
--use_answer_scorer  # Add this flag
```

---

## Memory Issues

### "CUDA out of memory"

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Cause**: Model or batch too large for GPU.

**Solutions**:

1. **Reduce max_samples**:
   ```bash
   --max_samples 100  # Instead of 200
   ```

2. **Use smaller model** (if applicable):
   ```bash
   --encoder_type sentence  # Instead of dpr
   ```

3. **Clear GPU cache**:
   ```python
   torch.cuda.empty_cache()
   ```

---

### "Cannot allocate memory" (CPU)

**Symptoms**:
```
MemoryError: Cannot allocate memory
```

**Cause**: Loading full 21M corpus into RAM.

**Solutions**:

1. **Use wiki_dpr mode** (compressed index):
   ```bash
   --corpus_mode wiki_dpr
   ```

2. **Enable memory mapping** (FAISS mode):
   ```bash
   --corpus_mode faiss \
   --faiss_mmap  # Memory-map embeddings
   ```

3. **Use aligned mode** (smaller corpus):
   ```bash
   --corpus_mode aligned \
   --n_distractors 10000  # Smaller corpus
   ```

---

## Performance Issues

### Evaluation is very slow

**Possible causes and solutions**:

1. **Using aligned mode first build**:
   - Wait for cache to build (one-time cost)
   - Or switch to wiki_dpr mode

2. **Large max_samples**:
   ```bash
   --max_samples 100  # Reduce for testing
   ```

3. **No GPU available**:
   ```bash
   # Check CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Top_k_retrieval too large**:
   ```bash
   --top_k_retrieval 50  # Instead of 1000
   ```

---

### Answer scoring is slow

**Cause**: Answer scorer runs DPR reader model on many passages.

**Solutions**:

1. **Reduce top_k_retrieval**:
   ```bash
   --top_k_retrieval 50  # Score fewer candidates
   ```

2. **Disable answer scorer** (if not needed):
   ```bash
   # Remove --use_answer_scorer flag
   ```

3. **Use cross_encoder backend** (faster):
   ```bash
   --use_answer_scorer \
   --answer_scorer_backend cross_encoder
   ```

---

## Quick Diagnostic Checklist

If something goes wrong, check these in order:

### 1. Environment
```bash
# Check you're in the right place
pwd  # Should show /path/to/QORE-VLM

# Check Python path
python -c "import sys; print(sys.path)"  # Should include repo root

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Code version
```bash
# Pull latest changes
git pull origin main

# Check if parameters exist
python -m scripts.rag.eval_rag_refactored --help | grep delta
python -m scripts.rag.eval_rag_refactored --help | grep complementarity
```

### 3. Corpus mode
```bash
# Test with simplest mode first
python -m scripts.rag.eval_rag_refactored \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --max_samples 1 \
    --method topk \
    --K 5 \
    --skip_generation \
    --wiki_dpr_config psgs_w100.nq.compressed
```

### 4. Cache
```bash
# Check HuggingFace cache
ls ~/.cache/huggingface/datasets/facebook___wiki_dpr/

# Check aligned corpus cache (if using aligned mode)
ls -lh /path/to/cache/corpus_*.{pkl,npy,json}
```

---

## Getting Help

If you're still stuck:

1. **Check logs**: Look for the last successful step before failure
2. **Minimal reproduction**: Try with `--max_samples 1` and `--method topk`
3. **Check documentation**: `docs/rag/corpus_modes.md` for corpus mode details
4. **Search issues**: Search GitHub issues for similar problems

---

## Common Command Templates

### Quick test (1 sample)
```bash
python -m scripts.rag.eval_rag_refactored \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --max_samples 1 \
    --method topk \
    --K 5 \
    --skip_generation \
    --wiki_dpr_config psgs_w100.nq.compressed \
    --wiki_dpr_cache_dir ~/.cache/huggingface/datasets
```

### Full evaluation with QORE
```bash
python -m scripts.rag.eval_rag_refactored \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --max_samples 200 \
    --method qore \
    --K 5 \
    --lam 2.0 \
    --gamma 0.5 \
    --delta 0.0 \
    --seed 42 \
    --wiki_dpr_config psgs_w100.nq.compressed \
    --wiki_dpr_cache_dir ~/.cache/huggingface/datasets \
    --output_dir results/
```

### With complementarity (idea 6)
```bash
python -m scripts.rag.eval_rag_refactored \
    --corpus_mode wiki_dpr \
    --dataset nq_open \
    --max_samples 200 \
    --method qore \
    --K 5 \
    --lam 2.0 \
    --gamma 0.5 \
    --delta 0.3 \
    --complementarity_method dpr \
    --use_answer_scorer \
    --seed 42 \
    --wiki_dpr_config psgs_w100.nq.compressed \
    --wiki_dpr_cache_dir ~/.cache/huggingface/datasets \
    --output_dir results/
```

---

## Version History

- **2026-07-28**: Added aligned mode progress logging (commit `1ab79b7`)
- **2026-07-28**: Added `--delta` and `--complementarity_method` parameters (commit `80ad6c7`)
- **2026-07-28**: Fixed PYTHONPATH in P2 experiment scripts (commit `80ad6c7`)
