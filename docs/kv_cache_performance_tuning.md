# KV Cache Performance Tuning Guide

## Quality-Speed Tradeoff

QORE offers configurable quality-speed tradeoff for KV cache eviction through the `num_reads` parameter (simulated annealing sampling iterations).

### Performance Profiles

| Mode | num_reads | Expected F1 | Latency (relative to H2O) | Use Case |
|------|-----------|-------------|---------------------------|----------|
| **Quality** (default) | 100 | 0.1949 | ~6× | Paper results, highest accuracy needed |
| **Balanced** | 50 | ~0.19 (-2% est.) | ~3-4× | Production, most use cases |
| **Fast** | 30 | ~0.18 (-5% est.) | ~2-3× | Prototyping, latency-critical |

*Note: F1 and latency estimates based on preliminary testing. Actual values may vary by dataset.*

---

## Usage

### Option 1: Use Speed Mode Presets (Recommended)

**Quality mode** (default, best F1):
```bash
python -m scripts.kv_cache.eval_kv_cache \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset longbench \
  --policy qore \
  --speed_mode quality \
  --max_capacity 1024
```

**Balanced mode** (recommended for production):
```bash
python -m scripts.kv_cache.eval_kv_cache \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset longbench \
  --policy qore \
  --speed_mode balanced \
  --max_capacity 1024
```

**Fast mode** (latency-critical applications):
```bash
python -m scripts.kv_cache.eval_kv_cache \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset longbench \
  --policy qore \
  --speed_mode fast \
  --max_capacity 1024
```

### Option 2: Direct num_reads Control (Advanced)

For fine-grained control, set `num_reads` directly:

```bash
python -m scripts.kv_cache.eval_kv_cache \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset longbench \
  --policy qore \
  --num_reads 75 \
  --max_capacity 1024
```

**Note**: `--speed_mode` overrides `--num_reads` if both are specified.

---

## When to Use Each Mode

### Quality Mode (num_reads=100)
✅ Use when:
- Reporting paper/benchmark results
- Accuracy is critical
- Latency is not a constraint
- Baseline comparison with other methods

❌ Avoid when:
- Real-time inference needed
- Limited compute budget

### Balanced Mode (num_reads=50) ⭐ Recommended
✅ Use when:
- Deploying to production
- Good balance of quality and speed needed
- Most general use cases

This is the **sweet spot** for most applications.

### Fast Mode (num_reads=30)
✅ Use when:
- Prototyping/development
- Latency is critical (e.g., interactive applications)
- Willing to accept slight quality loss

❌ Avoid when:
- Final evaluation/benchmarking
- Quality cannot be compromised

---

## Performance Comparison

Example results on LongBench (10 samples):

```
Mode      num_reads  F1       Latency(ms)  Speedup vs Quality
────────────────────────────────────────────────────────────
Quality   100        0.1949   18023        1.0×
Balanced  50         ~0.19    ~9000        ~2.0×
Fast      30         ~0.18    ~5400        ~3.3×
H2O       -          0.1412   2907         6.2×
```

**Key insights**:
- Balanced mode provides 2× speedup with minimal quality loss
- Even Fast mode significantly outperforms H2O baseline in quality
- QORE remains competitive in latency while offering superior F1

---

## Technical Details

### What is num_reads?

`num_reads` controls the number of samples in simulated annealing (SA) optimization:
- **Higher** → More exploration → Better solutions → Slower
- **Lower** → Less exploration → Risk of local optima → Faster

### Why does it affect quality?

QORE solves a QUBO optimization problem to select which KV cache entries to evict:
- More SA samples → higher chance of finding the global optimum
- Fewer samples → may converge to suboptimal solutions

### Diminishing Returns

Quality improvements plateau beyond num_reads=100:
- 30→50: noticeable improvement
- 50→100: moderate improvement
- 100→200: minimal improvement (~1-2%)

This is why we set 100 as the default "quality" mode.

---

## Example Workflow

### 1. Development/Prototyping
```bash
# Use fast mode for quick iteration
--speed_mode fast
```

### 2. Validation
```bash
# Use balanced mode to verify quality
--speed_mode balanced
```

### 3. Final Evaluation
```bash
# Use quality mode for paper/benchmark results
--speed_mode quality
```

---

## Frequently Asked Questions

**Q: What should I use for paper results?**  
A: Always use `--speed_mode quality` or `--num_reads 100` (default) to report the best possible F1 score.

**Q: What's the best for production deployment?**  
A: `--speed_mode balanced` (num_reads=50) offers the best quality-speed tradeoff for most applications.

**Q: Can I use values other than 30/50/100?**  
A: Yes! Use `--num_reads <value>` directly. Any value from 10-200 is reasonable.

**Q: Does this apply to other policies (H2O, SnapKV)?**  
A: No, `num_reads` only affects QORE. Other policies have fixed algorithms.

**Q: Will quality mode always be slower?**  
A: Yes, but the slowdown is in the prefill phase (happens once). Generation speed is unaffected.

---

## Related Parameters

Other parameters that affect QORE performance:

- `--solver_method`: Choice of QUBO solver (default: `anneal`)
  - `anneal`: Simulated annealing (default, configurable via num_reads)
  - `greedy`: Fast heuristic (ignore num_reads)
  
- `--quality`: Quality signal for QORE (default: `attention`)
  - `attention`: Real cumulative attention (best quality)
  - `keynorm`: Key norm proxy (faster)

For most users, the default settings with `--speed_mode` adjustment are sufficient.

---

## Recommendations Summary

| Scenario | Recommendation |
|----------|----------------|
| **Default** | No flag needed (quality mode, num_reads=100) |
| **Production** | `--speed_mode balanced` |
| **Prototyping** | `--speed_mode fast` |
| **Latency-critical** | `--speed_mode fast` or `--num_reads 20-30` |
| **Ablation study** | Test all three modes, report tradeoff curve |

---

**Last updated**: 2026-07-16  
**QORE Version**: 1.0
