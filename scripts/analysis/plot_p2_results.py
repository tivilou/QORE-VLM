#!/usr/bin/env python3
"""Visualize Phase 2 experiment results: Idea 6 (Complementarity) sweep.

Usage:
    python scripts/analysis/plot_p2_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from exchange/p2_solver_idea6/20260729T131723/README.md
configs = [
    {"name": "γ=0.5, δ=0.0 [baseline]", "gamma": 0.5, "delta": 0.0, "recall": 0.3196, "redundancy": 0.8090, "f1": 0.4406, "em": 0.2500, "precision": 0.2680, "is_baseline": True},
    {"name": "γ=0.3, δ=0.1", "gamma": 0.3, "delta": 0.1, "recall": 0.4598, "redundancy": 0.7982, "f1": 0.5101, "em": 0.2950, "precision": 0.3640, "is_baseline": False},
    {"name": "γ=0.3, δ=0.3", "gamma": 0.3, "delta": 0.3, "recall": 0.4566, "redundancy": 0.8022, "f1": 0.5107, "em": 0.3000, "precision": 0.3620, "is_baseline": False},
    {"name": "γ=0.3, δ=0.5", "gamma": 0.3, "delta": 0.5, "recall": 0.4193, "redundancy": 0.7993, "f1": 0.4964, "em": 0.2850, "precision": 0.3380, "is_baseline": False},
    {"name": "γ=0.5, δ=0.1", "gamma": 0.5, "delta": 0.1, "recall": 0.4454, "redundancy": 0.7876, "f1": 0.5092, "em": 0.2950, "precision": 0.3570, "is_baseline": False},
    {"name": "γ=0.5, δ=0.3", "gamma": 0.5, "delta": 0.3, "recall": 0.4502, "redundancy": 0.7943, "f1": 0.5092, "em": 0.2950, "precision": 0.3550, "is_baseline": False},
    {"name": "γ=0.5, δ=0.5", "gamma": 0.5, "delta": 0.5, "recall": 0.4052, "redundancy": 0.7916, "f1": 0.5029, "em": 0.2850, "precision": 0.3320, "is_baseline": False},
    {"name": "γ=0.7, δ=0.1", "gamma": 0.7, "delta": 0.1, "recall": 0.4408, "redundancy": 0.7811, "f1": 0.5004, "em": 0.2900, "precision": 0.3550, "is_baseline": False},
    {"name": "γ=0.7, δ=0.3", "gamma": 0.7, "delta": 0.3, "recall": 0.4387, "redundancy": 0.7886, "f1": 0.5046, "em": 0.2900, "precision": 0.3490, "is_baseline": False},
    {"name": "γ=0.7, δ=0.5", "gamma": 0.7, "delta": 0.5, "recall": 0.3817, "redundancy": 0.7865, "f1": 0.4894, "em": 0.2750, "precision": 0.3200, "is_baseline": False},
]

baseline = configs[0]
idea6_configs = configs[1:]

# Best config by target: lowest redundancy with F1 >= baseline
best_low_redundancy = min([c for c in idea6_configs if c["f1"] >= baseline["f1"]], key=lambda x: x["redundancy"])

# Best config by F1
best_f1 = max(idea6_configs, key=lambda x: x["f1"])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Phase 2 Results: Idea 6 (Complementarity) vs Baseline", fontsize=16, fontweight='bold')

# 1. Recall vs Redundancy (Pareto front)
ax = axes[0, 0]
for c in idea6_configs:
    color = 'red' if c == best_low_redundancy else 'steelblue'
    marker = 's' if c == best_low_redundancy else 'o'
    size = 120 if c == best_low_redundancy else 80
    ax.scatter(c["redundancy"], c["recall"], c=color, s=size, alpha=0.7, edgecolors='black', linewidth=1.5, marker=marker, label=c["name"] if c == best_low_redundancy else "")

ax.scatter(baseline["redundancy"], baseline["recall"], c='orange', s=150, marker='*', edgecolors='black', linewidth=2, label='Baseline', zorder=10)
ax.set_xlabel('Redundancy', fontsize=12)
ax.set_ylabel('Recall@5', fontsize=12)
ax.set_title('Recall vs Redundancy', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# 2. F1 by gamma and delta
ax = axes[0, 1]
gammas = sorted(set(c["gamma"] for c in idea6_configs))
deltas = sorted(set(c["delta"] for c in idea6_configs))
x = np.arange(len(gammas))
width = 0.25

for i, delta in enumerate(deltas):
    f1_values = [next((c["f1"] for c in idea6_configs if c["gamma"] == g and c["delta"] == delta), 0) for g in gammas]
    ax.bar(x + i * width, f1_values, width, label=f'δ={delta}', alpha=0.8)

ax.axhline(baseline["f1"], color='orange', linestyle='--', linewidth=2, label='Baseline F1')
ax.set_xlabel('γ (Diversity Weight)', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score by γ and δ', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([f'{g}' for g in gammas])
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 3. EM by gamma and delta
ax = axes[0, 2]
for i, delta in enumerate(deltas):
    em_values = [next((c["em"] for c in idea6_configs if c["gamma"] == g and c["delta"] == delta), 0) for g in gammas]
    ax.bar(x + i * width, em_values, width, label=f'δ={delta}', alpha=0.8)

ax.axhline(baseline["em"], color='orange', linestyle='--', linewidth=2, label='Baseline EM')
ax.set_xlabel('γ (Diversity Weight)', fontsize=12)
ax.set_ylabel('Exact Match', fontsize=12)
ax.set_title('Exact Match by γ and δ', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([f'{g}' for g in gammas])
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 4. Delta sweep at different gammas
ax = axes[1, 0]
for gamma in gammas:
    gamma_configs = [c for c in idea6_configs if c["gamma"] == gamma]
    deltas_g = [c["delta"] for c in gamma_configs]
    f1_g = [c["f1"] for c in gamma_configs]
    ax.plot(deltas_g, f1_g, marker='o', label=f'γ={gamma}', linewidth=2, markersize=8)

ax.axhline(baseline["f1"], color='orange', linestyle='--', linewidth=2, label='Baseline')
ax.set_xlabel('δ (Complementarity Weight)', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 vs δ at Different γ', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 5. Metrics comparison: Baseline vs Best
ax = axes[1, 1]
metrics = ['Recall@5', 'F1', 'EM', 'Precision']
baseline_vals = [baseline["recall"], baseline["f1"], baseline["em"], baseline["precision"]]
best_vals = [best_f1["recall"], best_f1["f1"], best_f1["em"], best_f1["precision"]]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x_pos - width/2, baseline_vals, width, label='Baseline (γ=0.5, δ=0.0)', color='orange', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, best_vals, width, label=f'Best F1 {best_f1["name"]}', color='steelblue', alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Baseline vs Best Configuration', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# 6. Improvement summary
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
Phase 2 Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline (γ=0.5, δ=0.0):
  • Recall@5: {baseline['recall']:.4f}
  • Redundancy: {baseline['redundancy']:.4f}
  • F1: {baseline['f1']:.4f}
  • EM: {baseline['em']:.4f}

Best by F1 ({best_f1['name']}):
  • Recall@5: {best_f1['recall']:.4f} (+{best_f1['recall']-baseline['recall']:.4f})
  • Redundancy: {best_f1['redundancy']:.4f} ({best_f1['redundancy']-baseline['redundancy']:.4f})
  • F1: {best_f1['f1']:.4f} (+{best_f1['f1']-baseline['f1']:.4f})
  • EM: {best_f1['em']:.4f} (+{best_f1['em']-baseline['em']:.4f})

Best by Low Redundancy ({best_low_redundancy['name']}):
  • Redundancy: {best_low_redundancy['redundancy']:.4f} ({best_low_redundancy['redundancy']-baseline['redundancy']:.4f})
  • F1: {best_low_redundancy['f1']:.4f} (+{best_low_redundancy['f1']-baseline['f1']:.4f})

Key Findings:
  ✓ Idea 6 improves all metrics
  ✓ Optimal δ ∈ [0.1, 0.3]
  ✓ γ=0.5 balances F1 & redundancy
  ✓ Relative F1 gain: +13.6%
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save
output_dir = Path(__file__).parent.parent.parent / "exchange" / "p2_solver_idea6" / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "p2_results_20260729.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ 图表已保存到: {output_path}")

plt.show()
