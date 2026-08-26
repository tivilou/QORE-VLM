"""Gold-free, observation-only context-budget routing for Phase 10A."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from applications.rag.answer_evidence import passage_duplication_matrix


class AdaptiveContextError(ValueError):
    """Raised when a routing contract is malformed."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def risk_features(answer_scores: Sequence[float], selected_local: Sequence[int], selected_texts: Sequence[str]) -> dict[str, float]:
    """Compute pre-generation risk signals only from baseline-available values."""
    scores = np.asarray(answer_scores, dtype=np.float64)
    selected = np.asarray([int(value) for value in selected_local], dtype=np.int64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise AdaptiveContextError("answer_scores must be a finite vector")
    if selected.ndim != 1 or len(selected) == 0 or np.any(selected < 0) or np.any(selected >= len(scores)):
        raise AdaptiveContextError("selected_local is out of range")
    if len(selected_texts) != len(selected):
        raise AdaptiveContextError("selected_texts and selected_local must have equal length")
    confidence = float(np.clip(np.mean(scores[selected]), 0.0, 1.0))
    ordered = np.sort(scores)[::-1]
    margin = float(np.clip(ordered[0] - ordered[1], 0.0, 1.0)) if len(ordered) > 1 else 1.0
    duplication = passage_duplication_matrix([str(value) for value in selected_texts])
    pairs = duplication[np.triu_indices(len(selected_texts), k=1)]
    duplication_mean = float(np.mean(pairs)) if len(pairs) else 0.0
    confidence_risk = 1.0 - confidence
    margin_risk = 1.0 - margin
    risk = 0.50 * confidence_risk + 0.25 * margin_risk + 0.25 * duplication_mean
    values = {
        "selected_confidence": confidence,
        "score_margin": margin,
        "selected_duplication": duplication_mean,
        "confidence_risk": confidence_risk,
        "margin_risk": margin_risk,
        "risk_score": float(np.clip(risk, 0.0, 1.0)),
    }
    if not all(_finite(value) for value in values.values()):
        raise AdaptiveContextError("risk features are not finite")
    return values


def route_extra_context(risk_score: float, threshold: float) -> bool:
    """Return whether the fixed extra-context arm should be applied."""
    if not _finite(risk_score) or not _finite(threshold) or not 0.0 <= float(risk_score) <= 1.0:
        raise AdaptiveContextError("risk_score must be in [0,1] and finite")
    if not 0.0 <= float(threshold) <= 1.0:
        raise AdaptiveContextError("threshold must be in [0,1]")
    return float(risk_score) >= float(threshold)


def build_wide_context(
    retrieved_texts: Sequence[str], selected_local: Sequence[int], *, extra_count: int
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Append first unselected retrieved passages; preserve the selected prefix."""
    texts = tuple(str(value) for value in retrieved_texts)
    selected = tuple(int(value) for value in selected_local)
    if len(set(selected)) != len(selected) or any(value < 0 or value >= len(texts) for value in selected):
        raise AdaptiveContextError("selected_local must contain unique in-range indices")
    if not isinstance(extra_count, int) or extra_count < 0:
        raise AdaptiveContextError("extra_count must be a non-negative integer")
    extras = tuple(index for index in range(len(texts)) if index not in set(selected))[:extra_count]
    wide = tuple(texts[index] for index in selected) + tuple(texts[index] for index in extras)
    return wide, extras


def prefix_digest(retrieved_indices: Sequence[int], selected_local: Sequence[int]) -> str:
    """Hash selected global IDs for compact prefix-parity auditing."""
    selected = [int(retrieved_indices[int(index)]) for index in selected_local]
    payload = ",".join(map(str, selected)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AdaptiveContextError",
    "build_wide_context",
    "prefix_digest",
    "risk_features",
    "route_extra_context",
]
