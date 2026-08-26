"""Consensus-gated answer-hypothesis bridge retrieval diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .answer_evidence import normalize_answer_text


BRIDGE_PLUGIN_ID = "answer_hypothesis_evidence_bridge"


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("hypothesis scores must be finite")
    return result


def consensus_hypothesis(
    hypotheses: Sequence[Sequence[Mapping[str, Any]]],
    *,
    min_support: int = 2,
    min_probability: float = 0.2,
) -> dict[str, Any] | None:
    """Return the strongest answer supported by distinct passages."""
    if int(min_support) != min_support or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    if not 0.0 <= float(min_probability) <= 1.0:
        raise ValueError("min_probability must be in [0, 1]")

    candidates: dict[str, dict[str, Any]] = {}
    for passage_index, row in enumerate(hypotheses):
        per_passage: dict[str, dict[str, Any]] = {}
        for item in row:
            text = str(item.get("text") or item.get("normalized") or "").strip()
            normalized = normalize_answer_text(str(item.get("normalized") or text))
            if not normalized:
                continue
            probability = _finite(item.get("probability", 0.0))
            score = _finite(item.get("score", 0.0))
            if probability < 0.0:
                raise ValueError("hypothesis probability must be non-negative")
            candidate = {"text": text, "probability": probability, "score": score}
            previous = per_passage.get(normalized)
            if previous is None or (probability, score, text) > (
                float(previous["probability"]), float(previous["score"]), str(previous["text"])
            ):
                per_passage[normalized] = candidate
        for normalized, candidate in per_passage.items():
            aggregate = candidates.setdefault(
                normalized,
                {"normalized": normalized, "text": candidate["text"], "passage_indices": [], "probabilities": [], "scores": []},
            )
            aggregate["passage_indices"].append(int(passage_index))
            aggregate["probabilities"].append(float(candidate["probability"]))
            aggregate["scores"].append(float(candidate["score"]))
            aggregate["text"] = min(str(aggregate["text"]), str(candidate["text"]))

    eligible = []
    for aggregate in candidates.values():
        support = len(aggregate["passage_indices"])
        mean_probability = sum(aggregate["probabilities"]) / max(support, 1)
        if support >= min_support and mean_probability >= float(min_probability):
            eligible.append({
                "text": str(aggregate["text"]),
                "normalized": str(aggregate["normalized"]),
                "support": int(support),
                "mean_probability": float(mean_probability),
                "max_score": float(max(aggregate["scores"])),
                "passage_indices": sorted(aggregate["passage_indices"]),
            })
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item["support"], -item["mean_probability"], -item["max_score"], item["normalized"]))
    return eligible[0]


def build_bridge_query(question: str, hypothesis: Mapping[str, Any]) -> str:
    question_text = str(question).strip()
    answer_text = str(hypothesis.get("text") or hypothesis.get("normalized") or "").strip()
    if not question_text or not answer_text:
        raise ValueError("question and hypothesis text must be non-empty")
    return f"{question_text} Answer hypothesis: {answer_text}"


def merge_candidate_indices(initial_indices: Sequence[int], bridge_indices: Sequence[int]) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()
    for value in [*initial_indices, *bridge_indices]:
        index = int(value)
        if index < 0:
            raise ValueError("candidate indices must be non-negative")
        if index not in seen:
            seen.add(index)
            merged.append(index)
    return merged


def bridge_spec_hash(*, min_support: int, min_probability: float, query_template: str = "question + answer_hypothesis") -> str:
    payload = {"plugin_id": BRIDGE_PLUGIN_ID, "min_support": int(min_support), "min_probability": float(min_probability), "query_template": str(query_template)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
