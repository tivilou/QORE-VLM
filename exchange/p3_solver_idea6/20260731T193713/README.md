# Idea 6 Phase 3 Results

**Timestamp**: 20260731T193713

---

## Quick Summary

### baseline

| Seed | Recall@5 | F1 | EM | Redundancy |
|------|----------|----|----|------------|
| 42 | 0.3407 | N/A | N/A | 0.8110 |
| 43 | 0.3407 | N/A | N/A | 0.8110 |
| 44 | 0.3407 | N/A | N/A | 0.8110 |

### idea6_recommended

| Seed | Recall@5 | F1 | EM | Redundancy |
|------|----------|----|----|------------|
| 42 | 0.4802 | N/A | N/A | 0.7896 |
| 43 | 0.4802 | N/A | N/A | 0.7896 |
| 44 | 0.4802 | N/A | N/A | 0.7896 |

### idea6_best

| Seed | Recall@5 | F1 | EM | Redundancy |
|------|----------|----|----|------------|
| 42 | 0.4913 | N/A | N/A | 0.7986 |
| 43 | 0.4913 | N/A | N/A | 0.7986 |
| 44 | 0.4913 | N/A | N/A | 0.7986 |

---

## Files

- `results.zip`: Compressed evaluator result JSON files
- `seed_XX/CONFIG/log.txt`: Execution logs (uncompressed)
- `git_*.txt`: Git status information

For detailed analysis, run:
```bash
python scripts/collab/idea6_phase3/analyze_p3_results.py exchange/p3_solver_idea6/20260731T193713
```
