"""Pure utilities for Phase 7A answer-identity diagnostics.

The functions in this module do not alter retrieval, selection, generation, or
evaluation.  They convert already-computed reader logits into compact evidence
diagnostics that can be inspected before an answer-conditioned selector exists.
"""

from __future__ import annotations

import hashlib
import math
import re
import string
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer_text(text: str) -> str:
    """Return an NQ-style canonical form for answer and alias comparison."""
    lowered = str(text).casefold()
    without_punctuation = "".join(
        " " if character in string.punctuation else character
        for character in lowered
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return _WHITESPACE.sub(" ", without_articles).strip()


def build_alias_map(alias_groups: Iterable[Iterable[str]] | None = None) -> dict[str, str]:
    """Map normalized aliases to the first normalized non-empty group member."""
    aliases: dict[str, str] = {}
    for group in alias_groups or ():
        normalized = [normalize_answer_text(value) for value in group]
        canonical = next((value for value in normalized if value), "")
        if not canonical:
            continue
        for value in normalized:
            if value:
                aliases[value] = canonical
    return aliases


def _softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    shifted = array - np.max(array)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _passage_positions(
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
    tokenizer: Any,
) -> list[int]:
    valid = np.flatnonzero(attention_mask > 0).tolist()
    if not valid:
        return []
    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    sep_id = getattr(tokenizer, "sep_token_id", None)
    first_passage_position = valid[0]
    if sep_id is not None:
        separators = [index for index in valid if int(token_ids[index]) == int(sep_id)]
        if separators:
            first_passage_position = separators[0] + 1
    return [
        index
        for index in valid
        if index >= first_passage_position and int(token_ids[index]) not in special_ids
    ]


def extract_top_answer_spans(
    input_ids: Any,
    start_logits: Any,
    end_logits: Any,
    tokenizer: Any,
    *,
    attention_mask: Any | None = None,
    top_m: int = 3,
    max_answer_tokens: int = 10,
) -> list[list[dict[str, Any]]]:
    """Extract deterministic top answer spans from batched DPR reader logits.

    Candidate starts and ends are bounded before pairing so the diagnostic is
    inexpensive relative to the reader forward pass. Duplicate normalized spans
    are merged by retaining their highest logit score.
    """
    if top_m < 1:
        raise ValueError("top_m must be at least 1")
    if max_answer_tokens < 1:
        raise ValueError("max_answer_tokens must be at least 1")

    ids = np.asarray(input_ids)
    starts = np.asarray(start_logits, dtype=np.float64)
    ends = np.asarray(end_logits, dtype=np.float64)
    if ids.ndim != 2 or starts.shape != ids.shape or ends.shape != ids.shape:
        raise ValueError("input_ids, start_logits, and end_logits must share a 2D shape")
    masks = np.ones_like(ids) if attention_mask is None else np.asarray(attention_mask)
    if masks.shape != ids.shape:
        raise ValueError("attention_mask must match input_ids")

    results: list[list[dict[str, Any]]] = []
    candidate_limit = max(16, top_m * 8)
    for row in range(ids.shape[0]):
        positions = _passage_positions(ids[row], masks[row], tokenizer)
        ranked_starts = sorted(positions, key=lambda index: (-starts[row, index], index))[
            :candidate_limit
        ]
        ranked_ends = sorted(positions, key=lambda index: (-ends[row, index], index))[
            :candidate_limit
        ]
        unique: dict[str, dict[str, Any]] = {}
        for start in ranked_starts:
            for end in ranked_ends:
                if end < start or end - start + 1 > max_answer_tokens:
                    continue
                text = tokenizer.decode(
                    ids[row, start : end + 1].tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                normalized = normalize_answer_text(text)
                if not normalized:
                    continue
                score = float(starts[row, start] + ends[row, end])
                candidate = {
                    "text": text,
                    "normalized": normalized,
                    "score": score,
                    "start": int(start),
                    "end": int(end),
                }
                previous = unique.get(normalized)
                if previous is None or (
                    score,
                    -int(start),
                    -int(end),
                ) > (
                    float(previous["score"]),
                    -int(previous["start"]),
                    -int(previous["end"]),
                ):
                    unique[normalized] = candidate

        ranked = sorted(
            unique.values(),
            key=lambda item: (-float(item["score"]), int(item["start"]), int(item["end"])),
        )[:top_m]
        probabilities = _softmax([float(item["score"]) for item in ranked])
        for item, probability in zip(ranked, probabilities):
            item["probability"] = float(probability)
        results.append(ranked)
    return results


def hypothesis_distribution(
    hypotheses: Sequence[Mapping[str, Any]],
    *,
    alias_map: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Convert span hypotheses into a normalized canonical-answer posterior."""
    distribution: dict[str, float] = {}
    aliases = alias_map or {}
    for item in hypotheses:
        normalized = normalize_answer_text(str(item.get("normalized") or item.get("text") or ""))
        canonical = aliases.get(normalized, normalized)
        probability = float(item.get("probability", 0.0))
        if canonical and math.isfinite(probability) and probability > 0.0:
            distribution[canonical] = distribution.get(canonical, 0.0) + probability
    total = sum(distribution.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in sorted(distribution.items())}


def answer_agreement_matrix(
    hypotheses: Sequence[Sequence[Mapping[str, Any]]],
    *,
    alias_groups: Iterable[Iterable[str]] | None = None,
) -> np.ndarray:
    """Compute Bhattacharyya agreement between passage answer posteriors."""
    aliases = build_alias_map(alias_groups)
    distributions = [hypothesis_distribution(items, alias_map=aliases) for items in hypotheses]
    size = len(distributions)
    agreement = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            shared = set(distributions[left]) & set(distributions[right])
            value = sum(
                math.sqrt(distributions[left][key] * distributions[right][key])
                for key in shared
            )
            agreement[left, right] = agreement[right, left] = float(np.clip(value, 0.0, 1.0))
    return agreement


def answer_conflict_matrix(
    hypotheses: Sequence[Sequence[Mapping[str, Any]]],
    agreement: np.ndarray,
    *,
    passage_confidence: Sequence[float] | None = None,
) -> np.ndarray:
    """Score confident, different-top-answer pairs as potential conflicts."""
    size = len(hypotheses)
    if agreement.shape != (size, size):
        raise ValueError("agreement matrix shape does not match hypotheses")
    confidence = np.ones(size, dtype=np.float64)
    if passage_confidence is not None:
        confidence = np.clip(np.asarray(passage_confidence, dtype=np.float64), 0.0, 1.0)
        if confidence.shape != (size,):
            raise ValueError("passage_confidence must match hypotheses")

    top_answers: list[str] = []
    top_probability = np.zeros(size, dtype=np.float64)
    for index, items in enumerate(hypotheses):
        distribution = hypothesis_distribution(items)
        if distribution:
            answer, probability = max(distribution.items(), key=lambda item: (item[1], item[0]))
            top_answers.append(answer)
            top_probability[index] = probability
        else:
            top_answers.append("")

    conflict = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            if not top_answers[left] or not top_answers[right] or top_answers[left] == top_answers[right]:
                continue
            value = (
                confidence[left]
                * confidence[right]
                * top_probability[left]
                * top_probability[right]
                * (1.0 - agreement[left, right])
            )
            conflict[left, right] = conflict[right, left] = float(np.clip(value, 0.0, 1.0))
    return conflict


def decisive_conflict_matrix(
    hypotheses: Sequence[Sequence[Mapping[str, Any]]],
    agreement: np.ndarray,
    *,
    passage_confidence: Sequence[float] | None = None,
    alias_groups: Iterable[Iterable[str]] | None = None,
    confidence_threshold: float = 0.5,
    margin_threshold: float = 0.2,
) -> np.ndarray:
    """Score only concentrated, high-confidence answer divergences.

    The Phase 7B ``conflict`` feature treats any different top spans as a
    potential contradiction.  This stricter diagnostic requires both passages
    to have confidence and posterior margin above configurable thresholds.  It
    remains a diagnostic feature: different answers are not assumed to be
    logically contradictory without an entailment or exclusivity model.
    """
    size = len(hypotheses)
    if agreement.shape != (size, size):
        raise ValueError("agreement matrix shape does not match hypotheses")
    if not 0.0 <= confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be in [0, 1)")
    if not 0.0 <= margin_threshold < 1.0:
        raise ValueError("margin_threshold must be in [0, 1)")

    aliases = build_alias_map(alias_groups)
    confidence = np.ones(size, dtype=np.float64)
    if passage_confidence is not None:
        confidence = np.clip(np.asarray(passage_confidence, dtype=np.float64), 0.0, 1.0)
        if confidence.shape != (size,):
            raise ValueError("passage_confidence must match hypotheses")

    top_answers: list[str] = []
    concentration = np.zeros(size, dtype=np.float64)
    for index, items in enumerate(hypotheses):
        distribution = hypothesis_distribution(items, alias_map=aliases)
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            top_answers.append("")
            continue
        top_answers.append(ranked[0][0])
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        concentration[index] = float(
            np.clip(ranked[0][1] - second_probability, 0.0, 1.0)
        )

    def gate(value: float, threshold: float) -> float:
        return float(np.clip((value - threshold) / (1.0 - threshold), 0.0, 1.0))

    conflict = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            if (
                not top_answers[left]
                or not top_answers[right]
                or top_answers[left] == top_answers[right]
            ):
                continue
            confidence_gate = gate(float(confidence[left]), confidence_threshold) * gate(
                float(confidence[right]), confidence_threshold
            )
            margin_gate = gate(float(concentration[left]), margin_threshold) * gate(
                float(concentration[right]), margin_threshold
            )
            value = confidence_gate * margin_gate * (1.0 - agreement[left, right])
            conflict[left, right] = conflict[right, left] = float(np.clip(value, 0.0, 1.0))
    return conflict


def passage_duplication_matrix(passages: Sequence[str]) -> np.ndarray:
    """Compute deterministic token-Jaccard near-duplication scores."""
    token_sets = [set(normalize_answer_text(passage).split()) for passage in passages]
    size = len(token_sets)
    duplication = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            union = token_sets[left] | token_sets[right]
            value = len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0
            duplication[left, right] = duplication[right, left] = value
    return duplication


def build_answer_evidence_matrices(
    passages: Sequence[str],
    hypotheses: Sequence[Sequence[Mapping[str, Any]]],
    *,
    passage_confidence: Sequence[float] | None = None,
    alias_groups: Iterable[Iterable[str]] | None = None,
    decisive_confidence_threshold: float = 0.5,
    decisive_margin_threshold: float = 0.2,
) -> dict[str, np.ndarray]:
    """Build the Phase 7A agreement, conflict, duplicate, and support matrices."""
    if len(passages) != len(hypotheses):
        raise ValueError("passages and hypotheses must have the same length")
    agreement = answer_agreement_matrix(hypotheses, alias_groups=alias_groups)
    conflict = answer_conflict_matrix(
        hypotheses,
        agreement,
        passage_confidence=passage_confidence,
    )
    decisive_conflict = decisive_conflict_matrix(
        hypotheses,
        agreement,
        passage_confidence=passage_confidence,
        alias_groups=alias_groups,
        confidence_threshold=decisive_confidence_threshold,
        margin_threshold=decisive_margin_threshold,
    )
    duplication = passage_duplication_matrix(passages)
    corroboration = agreement * (1.0 - duplication)
    np.fill_diagonal(corroboration, 0.0)
    return {
        "agreement": agreement,
        "conflict": conflict,
        "decisive_conflict": decisive_conflict,
        "duplication": duplication,
        "corroboration": corroboration,
    }


def select_counterfactual_swap(
    conflict: np.ndarray,
    selected_indices: Sequence[int],
    *,
    minimum_conflict: float = 0.0,
) -> tuple[int, int, float] | None:
    """Choose the strongest selected-to-unselected answer-conflict swap."""
    matrix = np.asarray(conflict, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("conflict must be a square matrix")
    selected = sorted({int(index) for index in selected_indices})
    if any(index < 0 or index >= matrix.shape[0] for index in selected):
        raise ValueError("selected index is out of range")
    unselected = [index for index in range(matrix.shape[0]) if index not in selected]
    candidates = [
        (float(matrix[source, replacement]), source, replacement)
        for source in selected
        for replacement in unselected
        if float(matrix[source, replacement]) >= minimum_conflict
    ]
    if not candidates:
        return None
    value, source, replacement = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
    if value <= 0.0:
        return None
    return source, replacement, value


def _pair_values(matrix: np.ndarray, indices: Sequence[int]) -> list[float]:
    ordered = sorted({int(index) for index in indices})
    return [float(matrix[left, right]) for offset, left in enumerate(ordered) for right in ordered[offset + 1 :]]


def summarize_answer_evidence(
    matrices: Mapping[str, np.ndarray],
    hypotheses: Sequence[Sequence[Mapping[str, Any]]],
    selected_indices: Sequence[int],
    *,
    high_conflict_threshold: float = 0.25,
) -> dict[str, Any]:
    """Return a passage-free compact summary suitable for Exchange commits."""
    size = len(hypotheses)
    selected = sorted({int(index) for index in selected_indices})
    unselected = [index for index in range(size) if index not in selected]
    output: dict[str, Any] = {
        "candidate_count": size,
        "selected_count": len(selected),
        "nonempty_hypothesis_count": sum(bool(items) for items in hypotheses),
    }
    matrix_names = ["agreement", "conflict", "duplication", "corroboration"]
    if "decisive_conflict" in matrices:
        matrix_names.insert(2, "decisive_conflict")
    for name in matrix_names:
        matrix = np.asarray(matrices[name], dtype=np.float64)
        if matrix.shape != (size, size):
            raise ValueError(f"{name} matrix shape does not match hypotheses")
        if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0.0):
            raise ValueError(f"{name} matrix must be symmetric with zero diagonal")
        all_values = _pair_values(matrix, range(size))
        selected_values = _pair_values(matrix, selected)
        unselected_values = _pair_values(matrix, unselected)
        cross_values = [
            float(matrix[left, right])
            for left in selected
            for right in unselected
        ]
        output[name] = {
            "all_pair_mean": float(np.mean(all_values)) if all_values else 0.0,
            "all_pair_max": max(all_values, default=0.0),
            "selected_pair_mean": float(np.mean(selected_values)) if selected_values else 0.0,
            "selected_pair_max": max(selected_values, default=0.0),
            "unselected_pair_mean": float(np.mean(unselected_values)) if unselected_values else 0.0,
            "selected_to_unselected_mean": float(np.mean(cross_values)) if cross_values else 0.0,
        }
    conflict = np.asarray(matrices["conflict"], dtype=np.float64)
    cross_values = [float(conflict[left, right]) for left in selected for right in unselected]
    output["conflict"]["selected_to_unselected_max"] = max(cross_values, default=0.0)
    output["conflict"]["high_conflict_pair_count"] = sum(
        value >= high_conflict_threshold for value in _pair_values(conflict, range(size))
    )
    return output


def stable_identifier(value: str, *, namespace: str = "phase7a") -> str:
    """Hash an identifier so compact reports need not contain answer text."""
    payload = f"{namespace}:{normalize_answer_text(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
