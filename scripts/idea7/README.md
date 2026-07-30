# Idea 7: End-to-End QUBO Optimization

End-to-end training of QUBO objective using task loss (Recall/F1) instead of heuristic quality/redundancy signals.

## Motivation

Current QUBO uses **heuristic objective**:
- Quality: DPR similarity score or answer scorer
- Redundancy: embedding cosine similarity
- Fixed weights: `γ` balances quality vs redundancy

**Problem**: This objective may not align with the true task goal (Recall/F1).

**Idea 7 hypothesis**: Learn QUBO weights end-to-end from task loss to close the optimization gap.

## Technical Approach

### 1. Soft QUBO (Differentiable Relaxation)

Replace discrete binary selection `x ∈ {0,1}^N` with continuous probabilities `p ∈ [0,1]^N`:

```python
# Hard QUBO (non-differentiable)
x = solve_qubo(a, b, K)  # returns binary {0,1}^N

# Soft QUBO (differentiable)
p = soft_qubo(a, b, K)   # returns probabilities [0,1]^N
```

**Implementation**: Gumbel-Softmax with temperature annealing
- Start with high temperature (1.0) → soft probabilities
- Anneal to low temperature (0.3) → approaches discrete selection
- Gradients flow through `p` back to `a`, `b`, or upstream encoders

### 2. Learnable Weights

Instead of fixed `γ`, learn optimal weights:

```python
class LearnableQUBO(nn.Module):
    def __init__(self):
        self.w_a = nn.Parameter(torch.tensor(1.0))  # quality weight
        self.w_b = nn.Parameter(torch.tensor(1.0))  # redundancy weight
    
    def forward(self, a, b, K):
        a_weighted = self.w_a * a
        b_weighted = self.w_b * b
        p = soft_qubo(a_weighted, b_weighted, K)
        return p
```

### 3. Task Loss

Train with Recall as the loss function:

```
Recall(p, gold) = Σ_i [i ∈ gold] * p_i / |gold|
Loss = 1 - Recall
```

Differentiable w.r.t. `p` → backprop to QUBO weights.

## Usage

### Step 1: Quick Validation (10 samples, 5 min)

```bash
python -m scripts.idea7.train_soft_qubo \
    --max_samples 10 \
    --epochs 50 \
    --output_dir exchange/idea7_mvp
```

**Success criteria**:
- Training runs without errors ✅
- Recall improves over epochs (e.g., 0.3 → 0.5) ✅
- Learned weights are sensible (w_a > 0, w_b > 0) ✅

### Step 2: Full Training (200 samples, 2 hours)

```bash
python -m scripts.idea7.train_soft_qubo \
    --max_samples 200 \
    --epochs 100 \
    --model_type learnable \
    --lr 0.01 \
    --temperature_init 1.0 \
    --temperature_final 0.3 \
    --output_dir exchange/idea7_full
```

### Step 3: Evaluation

Compare trained model vs baseline on Phase 2 test set:

```bash
# TODO: Create eval_with_soft_qubo.py
python -m scripts.idea7.eval_with_soft_qubo \
    --checkpoint exchange/idea7_full/best_model.pt \
    --max_samples 200 \
    --output_file exchange/idea7_results.json
```

## Expected Results

### Hypothesis

Idea 7 should **close the QUBO gap** measured in Phase 1/Phase 2:
- Phase 1/Phase 2 gap: **0.3004** (30% of optimal Recall lost)
- Idea 7 target: **gap < 0.15** (close gap by half)

### Metrics to Track

| Metric | Baseline (Phase 2) | Idea 7 Target | Notes |
|--------|-------------------|---------------|-------|
| Recall@5 | 0.4454 | **0.55-0.60** | +0.10-0.15 gain |
| F1 | 0.5092 | **0.55-0.58** | Via Recall→F1 conversion (slope 0.271) |
| QUBO gap | 0.3004 | **< 0.15** | Close gap by half |
| Hit rate | 34.7% | **> 60%** | QUBO finds near-optimal subset |

### Decision Criteria

- **✅ Proceed with Idea 7** if:
  - Recall improves by **≥ 5%** (0.4454 → 0.467+)
  - QUBO gap decreases by **≥ 0.10** (0.3004 → 0.20 or better)
  - Training converges within 100 epochs

- **❌ Abandon Idea 7** if:
  - No improvement after 100 epochs
  - Learned weights collapse (w_a → 0 or w_b → 0)
  - Overfitting (train Recall ↑, val Recall ↓)

## Implementation Status

- [x] `qore/soft_qubo.py` - Differentiable QUBO (SoftQUBO, LearnableQUBO)
- [x] `scripts/idea7/train_soft_qubo.py` - Training script
- [ ] `scripts/idea7/eval_with_soft_qubo.py` - Evaluation script
- [ ] `scripts/idea7/compare_baselines.py` - Compare vs Phase 2 baselines

## Timeline

- **Day 1-2**: Validate MVP (10 samples) ← **WE ARE HERE**
- **Day 3-5**: Full training (200 samples) + tuning
- **Day 6-7**: Evaluation + comparison with Phase 2
- **Day 8-10**: Analyze results, decide go/no-go
- **Day 11-14**: Paper writeup if successful

## References

- Phase 1 diagnosis: `exchange/p1_diagnosis/qubo_diagnosis.md` (gap = 0.3004)
- Phase 2 Idea 6 results: `exchange/p2_idea6_gamma_delta_grid/` (Recall = 0.4454)
- Phase 2 Idea 7 re-eval: `exchange/p2_idea7_diagnosis/20260730T111502/` (gap unchanged)
- Next steps plan: `docs/idea7_next_steps.md`
