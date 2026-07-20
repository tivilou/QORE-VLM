"""Evaluation metrics for RAG passage selection and end-to-end QA.

Computes:
- Selection quality: Recall@K, Precision@K (needs gold passage labels),
  Redundancy, Diversity (from embeddings)
- QA quality: EM, F1 (from prediction vs gold answers)

All metrics are collected into a dict per sample, then aggregated across the
full dataset. Use the Evaluator class for a stateful runner, or the standalone
functions for one-off scoring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Selection metrics
# ────────────────────────────────────────────────────────────────────────────

def recall_at_k(selected_indices: set, gold_indices: set) -> Optional[float]:
    """Fraction of gold passages captured in the selected set.

    Returns None if gold_indices is empty (no gold to recall).
    """
    if not gold_indices:
        return None
    return len(selected_indices & gold_indices) / len(gold_indices)


def precision_at_k(selected_indices: set, gold_indices: set) -> Optional[float]:
    """Fraction of selected passages that are gold.

    Returns None if selected_indices is empty.
    """
    if not selected_indices:
        return None
    return len(selected_indices & gold_indices) / len(selected_indices)


def redundancy_ratio(selected_embeddings: np.ndarray, threshold: float = 0.85) -> float:
    """Mean pairwise cosine similarity among selected passages.

    High redundancy → passages are similar. threshold is unused in the mean but
    kept for API compat with a potential future "fraction above threshold" metric.
    """
    emb = np.asarray(selected_embeddings, dtype=np.float64)
    if len(emb) < 2:
        return 0.0
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = emb / norms
    sim = normed @ normed.T
    # Upper triangle excluding diagonal
    k = len(sim)
    if k < 2:
        return 0.0
    pairs = k * (k - 1) // 2
    total = (sim.sum() - np.trace(sim)) / 2.0
    return float(total / pairs) if pairs else 0.0


def diversity_ratio(selected_embeddings: np.ndarray, threshold: float = 0.85) -> float:
    """1 - redundancy_ratio. Higher is more diverse."""
    return 1.0 - redundancy_ratio(selected_embeddings, threshold)


# ────────────────────────────────────────────────────────────────────────────
# QA metrics (token-level)
# ────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower-case, strip articles/punctuation, collapse whitespace."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization (1.0 or 0.0)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1: 2*P*R/(P+R) over the word bags."""
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(ground_truth).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = sum((min(pred_toks.count(w), gold_toks.count(w)) for w in set(pred_toks)))
    if common == 0:
        return 0.0
    prec = common / len(pred_toks)
    rec = common / len(gold_toks)
    return 2.0 * prec * rec / (prec + rec)


def evaluate_answer(prediction: str, gold_answers: list[str]) -> dict:
    """Max EM and max F1 over all gold answer strings."""
    if not gold_answers:
        return {"em": 0.0, "f1": 0.0}
    em = max(compute_exact(prediction, g) for g in gold_answers)
    f1 = max(compute_f1(prediction, g) for g in gold_answers)
    return {"em": float(em), "f1": float(f1)}


# ────────────────────────────────────────────────────────────────────────────
# Evaluator (stateful)
# ────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Stateful evaluator collecting metrics across samples."""

    def __init__(self):
        self.samples = []

    def evaluate_sample(
        self,
        question_id,
        selected_indices: set,
        selected_embeddings: np.ndarray,
        gold_indices: set,
        prediction: Optional[str] = None,
        gold_answers: Optional[list[str]] = None,
        selection_time_ms: float = 0.0,
        generation_time_ms: float = 0.0,
        answer_hit_at_retrieved: Optional[bool] = None,
    ) -> dict:
        """Score one sample and append to history.

        Args:
            answer_hit_at_retrieved: Whether ANY of the retrieved candidates
                contains a gold answer string. True = retrieval succeeded;
                False = retrieval failure (gold not in Top-K candidates).
                None = unknown / not applicable (e.g. aligned mode).

        Returns the per-sample metric dict for immediate inspection.
        """
        metrics = {
            "question_id": question_id,
            "recall": recall_at_k(selected_indices, gold_indices),
            "precision": precision_at_k(selected_indices, gold_indices),
            "redundancy": redundancy_ratio(selected_embeddings),
            "diversity": diversity_ratio(selected_embeddings),
            "selection_time_ms": selection_time_ms,
            "generation_time_ms": generation_time_ms,
        }
        if answer_hit_at_retrieved is not None:
            metrics["answer_hit_at_retrieved"] = answer_hit_at_retrieved

        if prediction is not None and gold_answers is not None:
            qa = evaluate_answer(prediction, gold_answers)
            metrics.update(qa)
        self.samples.append(metrics)
        return metrics

    def aggregate(self) -> dict:
        """Compute mean/std over all evaluated samples.

        Key distinctions in output:
        - mean_recall: conditional on having gold in retrieved set (n_with_gold)
        - n_with_gold: questions where answer was found in retrieved candidates
        - n_retrieval_failure: questions where answer was NOT in retrieved candidates
          (only counted when answer_hit_at_retrieved is explicitly recorded)
        """
        if not self.samples:
            return {}

        keys = [
            "recall", "precision", "redundancy", "diversity",
            "em", "f1", "selection_time_ms", "generation_time_ms",
        ]
        agg = {}
        for k in keys:
            vals = [s[k] for s in self.samples if k in s and s[k] is not None]
            if vals:
                agg[f"mean_{k}"] = float(np.mean(vals))
                agg[f"std_{k}"] = float(np.std(vals))

        agg["n_samples"] = len(self.samples)

        # n_with_gold: questions where Top-K retrieval found at least one answer
        hit_recorded = [s for s in self.samples if "answer_hit_at_retrieved" in s]
        if hit_recorded:
            agg["n_with_gold"] = sum(1 for s in hit_recorded if s["answer_hit_at_retrieved"])
            agg["n_retrieval_failure"] = sum(1 for s in hit_recorded if not s["answer_hit_at_retrieved"])
        else:
            # Fallback: count samples where recall is not None (has gold in aligned/precomputed mode)
            agg["n_with_gold"] = sum(1 for s in self.samples if s.get("recall") is not None)

        return agg
