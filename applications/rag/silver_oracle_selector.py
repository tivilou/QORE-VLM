"""Gold-conditioned silver-oracle selection for offline diagnostics only.

This module must never be registered as a production selector.  It consumes
three-model evidence annotations that were created with access to gold answer
targets, so its output is an L0 diagnostic ceiling rather than a deployable
ranking method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ALLOWED_LABELS = frozenset(
    {"direct", "partial", "contradictory", "irrelevant", "uncertain"}
)
POSITIVE_LABELS = frozenset({"direct", "partial"})
EXPECTED_JUDGES = 3
CONSENSUS_VOTES = 2


class SilverOracleError(ValueError):
    """Raised when an oracle input violates the frozen diagnostic contract."""


@dataclass(frozen=True)
class SilverOracleSelection:
    """Deterministic oracle membership and an auditable priority trace."""

    selected_indices: tuple[int, ...]
    oracle_priority_indices: tuple[int, ...]
    trace: tuple[dict[str, Any], ...]


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SilverOracleError(f"{field} must be numeric") from exc
    if result != result or result in {float("inf"), float("-inf")}:
        raise SilverOracleError(f"{field} must be finite")
    return result


def _evidence_stats(evidence: Mapping[str, Any]) -> tuple[list[str], float]:
    models = evidence.get("models")
    labels: list[str] = []
    confidences: list[float] = []
    if isinstance(models, Mapping):
        for model in sorted(models):
            row = models[model]
            if not isinstance(row, Mapping):
                raise SilverOracleError("evidence.models entries must be mappings")
            labels.append(str(row.get("label", "")))
            confidences.append(
                _finite_float(row.get("confidence"), field="evidence confidence")
            )
    else:
        raw_labels = evidence.get("model_labels")
        if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
            raise SilverOracleError("evidence must contain models or model_labels")
        labels = [str(label) for label in raw_labels]
        if evidence.get("mean_confidence") is None:
            raise SilverOracleError("aggregate evidence requires mean_confidence")
        confidences = [
            _finite_float(evidence.get("mean_confidence"), field="mean_confidence")
        ]
    if len(labels) != EXPECTED_JUDGES:
        raise SilverOracleError(
            f"expected {EXPECTED_JUDGES} evidence labels, found {len(labels)}"
        )
    invalid = sorted(set(labels) - ALLOWED_LABELS)
    if invalid:
        raise SilverOracleError(f"invalid evidence labels: {invalid}")
    if any(confidence < 0.0 or confidence > 1.0 for confidence in confidences):
        raise SilverOracleError("evidence confidence must be in [0, 1]")
    return labels, sum(confidences) / len(confidences)


def evidence_counts(passage: Mapping[str, Any]) -> dict[str, Any]:
    """Return frozen three-model vote counts for one candidate passage."""

    evidence = passage.get("evidence")
    if not isinstance(evidence, Mapping):
        raise SilverOracleError("every candidate must contain evidence")
    labels, mean_confidence = _evidence_stats(evidence)
    direct_votes = sum(label == "direct" for label in labels)
    positive_votes = sum(label in POSITIVE_LABELS for label in labels)
    return {
        "model_labels": list(labels),
        "mean_confidence": mean_confidence,
        "direct_votes": direct_votes,
        "positive_votes": positive_votes,
        "direct_consensus": direct_votes >= CONSENSUS_VOTES,
        "positive_consensus": positive_votes >= CONSENSUS_VOTES,
    }


def select(passages: Sequence[Mapping[str, Any]], *, k: int = 5) -> SilverOracleSelection:
    """Select an evidence-maximizing diagnostic Top-K deterministically.

    Priority is direct consensus, then broad-positive consensus.  Candidates
    within those tiers are ordered by vote strength, annotation confidence,
    Answer Scorer score, and retrieval rank.  If fewer than K consensus-positive
    candidates exist, the remaining slots are filled by Answer Scorer score.
    The returned membership is intentionally target-conditioned and is not a
    valid online selector.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise SilverOracleError("k must be a positive integer")
    if len(passages) < k:
        raise SilverOracleError(f"cannot select {k} passages from {len(passages)}")
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, passage in enumerate(passages):
        candidate_id = str(passage.get("id", ""))
        if not candidate_id or candidate_id in seen_ids:
            raise SilverOracleError("candidate ids must be non-empty and unique")
        seen_ids.add(candidate_id)
        try:
            retrieved_rank = int(passage.get("retrieved_rank"))
        except (TypeError, ValueError) as exc:
            raise SilverOracleError("retrieved_rank must be an integer") from exc
        if retrieved_rank <= 0:
            raise SilverOracleError("retrieved_rank must be positive")
        answer_score = _finite_float(
            passage.get("answer_scorer_score"), field="answer_scorer_score"
        )
        evidence = evidence_counts(passage)
        tier = 0 if evidence["direct_consensus"] else 1 if evidence["positive_consensus"] else 2
        rows.append(
            {
                "index": index,
                "candidate_id": candidate_id,
                "retrieved_rank": retrieved_rank,
                "answer_scorer_score": answer_score,
                "tier": tier,
                **evidence,
            }
        )

    consensus_rows = sorted(
        (row for row in rows if row["tier"] < 2),
        key=lambda row: (
            row["tier"],
            -row["direct_votes"],
            -row["positive_votes"],
            -row["mean_confidence"],
            -row["answer_scorer_score"],
            row["retrieved_rank"],
            row["candidate_id"],
        ),
    )
    fallback_rows = sorted(
        (row for row in rows if row["tier"] == 2),
        key=lambda row: (
            -row["answer_scorer_score"],
            row["retrieved_rank"],
            row["candidate_id"],
        ),
    )
    priority = consensus_rows + fallback_rows
    chosen = priority[:k]
    trace = tuple(
        {
            "oracle_priority": priority_rank,
            "candidate_id": row["candidate_id"],
            "tier": (
                "direct_consensus"
                if row["tier"] == 0
                else "positive_consensus"
                if row["tier"] == 1
                else "answer_scorer_fill"
            ),
            "direct_votes": row["direct_votes"],
            "positive_votes": row["positive_votes"],
            "mean_confidence": row["mean_confidence"],
            "answer_scorer_score": row["answer_scorer_score"],
            "retrieved_rank": row["retrieved_rank"],
        }
        for priority_rank, row in enumerate(chosen, start=1)
    )
    return SilverOracleSelection(
        selected_indices=tuple(int(row["index"]) for row in chosen),
        oracle_priority_indices=tuple(int(row["index"]) for row in priority),
        trace=trace,
    )


def count_consensus(passages: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count direct and broad-positive consensus candidates in a passage set."""

    stats = [evidence_counts(passage) for passage in passages]
    return {
        "direct": sum(bool(item["direct_consensus"]) for item in stats),
        "broad": sum(bool(item["positive_consensus"]) for item in stats),
    }
