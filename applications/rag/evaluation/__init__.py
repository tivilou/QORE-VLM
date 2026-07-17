"""RAG evaluation: metrics for selection quality and QA performance."""

from .metrics import (
    Evaluator,
    compute_exact,
    compute_f1,
    diversity_ratio,
    evaluate_answer,
    normalize_answer,
    precision_at_k,
    recall_at_k,
    redundancy_ratio,
)

__all__ = [
    "Evaluator",
    "recall_at_k",
    "precision_at_k",
    "redundancy_ratio",
    "diversity_ratio",
    "compute_exact",
    "compute_f1",
    "evaluate_answer",
    "normalize_answer",
]
