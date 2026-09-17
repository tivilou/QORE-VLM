"""Evidence Conflict-Aware Diversification (ECAD) selector.

ECAD consumes a provenance-locked pairwise NLI artifact for the frozen
Top-50.  It greedily maximizes normalized Answer Scorer quality while
penalizing contradiction with passages already selected.  NLI scores are
computed before selection and are independent of Silver/gold labels and
Generator output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import subprocess
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rag.ecad_pairwise_nli.v1"
ARTIFACT_TYPE = "ecad_pairwise_nli_artifact"
SELECTOR_ID = "ecad_conflict_aware"
SELECTOR_VERSION = "1.0"
NLI_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
DEFAULT_K = 5
DEFAULT_CONFLICT_PENALTY = 0.20
DEFAULT_CONFLICT_THRESHOLD = 0.50
EXPECTED_TOP50 = 50
EXPECTED_PAIR_COUNT = EXPECTED_TOP50 * (EXPECTED_TOP50 - 1) // 2

_FORBIDDEN_FIELDS = frozenset(
    {
        "evidence",
        "gold",
        "gold_answers",
        "answer",
        "prediction",
        "metrics",
        "generation_panel",
        "generation_consensus",
        "evaluator",
        "evaluation",
        "label",
        "labels",
        "selectors",
    }
)
_ALLOWED_FIELDS = frozenset(
    {
        "id",
        "passage_id",
        "title",
        "text",
        "passage",
        "retrieved_rank",
        "rank",
        "retrieval_score",
        "answer_scorer_score",
        "answer_scorer",
    }
)


class ECADError(ValueError):
    """Raised when the ECAD selector or pairwise artifact is invalid."""


@dataclass(frozen=True)
class OnlineCandidate:
    passage_id: str
    text: str
    retrieved_rank: int
    retrieval_score: float
    answer_scorer_score: float


@dataclass(frozen=True)
class PairScore:
    left_id: str
    right_id: str
    contradiction: float
    entailment: float
    neutral: float


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    trace: tuple[dict[str, Any], ...]


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ECADError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ECADError(f"{field} must be a finite number")
    return result


def _candidate_text(value: Mapping[str, Any]) -> str:
    title = str(value.get("title") or "").strip()
    text = value.get("text", value.get("passage"))
    if not isinstance(text, str) or not text.strip():
        raise ECADError("candidate text must be non-empty")
    body = text.strip()
    return f"{title}. {body}" if title else body


def coerce_candidate(value: OnlineCandidate | Mapping[str, Any]) -> OnlineCandidate:
    if isinstance(value, OnlineCandidate):
        return value
    if not isinstance(value, Mapping):
        raise ECADError("candidate must be an object")
    keys = {str(key) for key in value}
    forbidden = sorted(keys & _FORBIDDEN_FIELDS)
    if forbidden:
        raise ECADError(f"forbidden selection fields: {', '.join(forbidden)}")
    unknown = sorted(keys - _ALLOWED_FIELDS)
    if unknown:
        raise ECADError(f"unknown selection fields: {', '.join(unknown)}")
    passage_id = value.get("id", value.get("passage_id"))
    if not isinstance(passage_id, (str, int)) or not str(passage_id).strip():
        raise ECADError("candidate id must be non-empty")
    rank = value.get("retrieved_rank", value.get("rank"))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ECADError("retrieved_rank must be a positive integer")
    score = value.get("answer_scorer_score", value.get("answer_scorer"))
    if score is None:
        raise ECADError("answer_scorer_score is required")
    return OnlineCandidate(
        passage_id=str(passage_id),
        text=_candidate_text(value),
        retrieved_rank=rank,
        retrieval_score=_finite(value.get("retrieval_score", 0.0), "retrieval_score"),
        answer_scorer_score=_finite(score, "answer_scorer_score"),
    )


def coerce_candidates(values: Sequence[OnlineCandidate | Mapping[str, Any]]) -> tuple[OnlineCandidate, ...]:
    candidates = tuple(coerce_candidate(value) for value in values)
    if not candidates:
        raise ECADError("at least one candidate is required")
    ids = [candidate.passage_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ECADError("candidate passage IDs must be unique")
    return candidates


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return str(left_id), str(right_id)


def _normalise_quality(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    return [(value - low) / span if span > 1e-12 else 0.0 for value in values]


def _pair_score(
    pair_scores: Mapping[tuple[str, str], PairScore],
    left_id: str,
    right_id: str,
) -> PairScore:
    pair = pair_scores.get(_pair_key(left_id, right_id))
    if pair is not None:
        return pair
    pair = pair_scores.get(_pair_key(right_id, left_id))
    if pair is not None:
        return pair
    raise ECADError(f"missing pairwise NLI score for ({left_id}, {right_id})")


def select(
    candidates: Sequence[OnlineCandidate | Mapping[str, Any]],
    pair_scores: Mapping[tuple[str, str], PairScore],
    *,
    k: int = DEFAULT_K,
    conflict_penalty: float = DEFAULT_CONFLICT_PENALTY,
) -> SelectionResult:
    """Select a fixed-size set with greedy contradiction-aware diversification."""

    rows = coerce_candidates(candidates)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ECADError("k must be a positive integer")
    if len(rows) < k:
        raise ECADError(f"need at least {k} candidates")
    conflict_penalty = _finite(conflict_penalty, "conflict_penalty")
    if conflict_penalty < 0.0:
        raise ECADError("conflict_penalty must be non-negative")
    ids = [row.passage_id for row in rows]
    normalized_quality = _normalise_quality([row.answer_scorer_score for row in rows])
    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    while len(selected) < k:
        available = [index for index in range(len(rows)) if index not in selected]
        scored: list[tuple[float, float, int, str, int, float]] = []
        for index in available:
            contradictions = [
                _pair_score(pair_scores, ids[index], ids[chosen]).contradiction
                for chosen in selected
            ]
            burden = sum(contradictions) / len(contradictions) if contradictions else 0.0
            objective = normalized_quality[index] - conflict_penalty * burden
            scored.append(
                (
                    objective,
                    normalized_quality[index],
                    -rows[index].retrieved_rank,
                    rows[index].passage_id,
                    index,
                    burden,
                )
            )
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        best = scored[0]
        selected.append(best[4])
        trace.append(
            {
                "step": len(selected),
                "candidate_rank": rows[best[4]].retrieved_rank,
                "objective": round(best[0], 8),
                "normalized_answer_scorer": round(best[1], 8),
                "contradiction_burden": round(best[5], 8),
                "conflict_penalty": conflict_penalty,
            }
        )
    # Use a neutral order for downstream context construction.
    selected.sort(key=lambda index: (rows[index].retrieved_rank, rows[index].passage_id))
    return SelectionResult(tuple(selected), tuple(trace))


def validate_pairwise_artifact(
    payload: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_model_id: str = NLI_MODEL_ID,
    expected_revision: str = NLI_REVISION,
) -> dict[int, dict[tuple[str, str], PairScore]]:
    """Validate complete rank-ordered NLI pairs against the exact input."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ECADError(f"expected schema {SCHEMA_VERSION}")
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ECADError("unexpected ECAD artifact type")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ECADError("ECAD provenance is required")
    if provenance.get("model_id") != expected_model_id:
        raise ECADError("ECAD NLI model id is not frozen")
    if provenance.get("revision") != expected_revision:
        raise ECADError("ECAD NLI revision is not frozen")
    config_hash = str(provenance.get("config_sha256", ""))
    if len(config_hash) != 64 or any(char not in "0123456789abcdef" for char in config_hash.lower()):
        raise ECADError("ECAD NLI config_sha256 is invalid")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ECADError("ECAD protocol is required")
    if int(protocol.get("case_count", -1)) != len(cases) or int(protocol.get("top50_count", -1)) != EXPECTED_TOP50:
        raise ECADError("ECAD case/Top-50 count is not frozen")
    if int(protocol.get("pair_count_per_case", -1)) != EXPECTED_PAIR_COUNT:
        raise ECADError("ECAD pair count is not frozen")
    if protocol.get("pair_order") != "top50_position_i_lt_j":
        raise ECADError("ECAD pair order is not frozen")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(cases):
        raise ECADError("ECAD artifact case rows are incomplete")
    by_number: dict[int, Mapping[str, Any]] = {}
    for item in raw_cases:
        if not isinstance(item, Mapping):
            raise ECADError("ECAD case row must be an object")
        number = int(item.get("case_number", -1))
        if number in by_number:
            raise ECADError(f"duplicate ECAD case number {number}")
        by_number[number] = item
    result: dict[int, dict[tuple[str, str], PairScore]] = {}
    for case_number, case in enumerate(cases, start=1):
        item = by_number.get(case_number)
        if item is None:
            raise ECADError(f"ECAD case {case_number} is missing")
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise ECADError(f"input case {case_number} does not contain Top-50")
        candidates = coerce_candidates(top50)
        ids = [candidate.passage_id for candidate in candidates]
        if item.get("candidate_ids") != ids:
            raise ECADError(f"ECAD candidate identity/order mismatch in case {case_number}")
        pairs = item.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIR_COUNT:
            raise ECADError(f"ECAD case {case_number} must contain {EXPECTED_PAIR_COUNT} pairs")
        pair_map: dict[tuple[str, str], PairScore] = {}
        pair_index = 0
        for left_index in range(EXPECTED_TOP50):
            for right_index in range(left_index + 1, EXPECTED_TOP50):
                row = pairs[pair_index]
                pair_index += 1
                if not isinstance(row, Mapping):
                    raise ECADError("ECAD pair row must be an object")
                left_id, right_id = str(row.get("left_id", "")), str(row.get("right_id", ""))
                if (left_id, right_id) != (ids[left_index], ids[right_index]):
                    raise ECADError(f"ECAD pair order mismatch in case {case_number} at pair {pair_index}")
                contradiction = _finite(row.get("contradiction"), "contradiction")
                entailment = _finite(row.get("entailment"), "entailment")
                neutral = _finite(row.get("neutral"), "neutral")
                if any(value < 0.0 or value > 1.0 for value in (contradiction, entailment, neutral)):
                    raise ECADError("ECAD NLI scores must be in [0, 1]")
                if not math.isclose(contradiction + entailment + neutral, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                    raise ECADError("ECAD NLI probabilities must sum to one")
                pair_map[(left_id, right_id)] = PairScore(left_id, right_id, contradiction, entailment, neutral)
        result[case_number] = pair_map
    return result


class FixedNLIModel:
    """Fixed three-way NLI model used to build ECAD pair artifacts."""

    model_id = NLI_MODEL_ID
    revision = NLI_REVISION

    def __init__(
        self,
        *,
        model_id: str = NLI_MODEL_ID,
        revision: str = NLI_REVISION,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        if model_id != NLI_MODEL_ID or revision != NLI_REVISION:
            raise ECADError("ECAD NLI identity is fixed")
        if batch_size < 1 or max_length < 32:
            raise ECADError("invalid NLI batching or max_length")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ECADError("ECAD NLI requires torch and transformers") from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).to(self.device)
        self.model.eval()
        config_json = self.model.config.to_json_string().encode("utf-8")
        tokenizer_json = repr(sorted(self.tokenizer.init_kwargs.items())).encode("utf-8")
        self.provenance = {
            "model_id": model_id,
            "revision": revision,
            "config_sha256": hashlib.sha256(config_json).hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(tokenizer_json).hexdigest(),
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "score": "softmax(sequence_classification_logits)",
        }
        raw_labels = getattr(self.model.config, "id2label", {}) or {}
        labels = {int(key): str(value).lower() for key, value in raw_labels.items()}
        self._label_indices = {}
        for index, label in labels.items():
            if "contrad" in label:
                self._label_indices["contradiction"] = index
            elif "entail" in label:
                self._label_indices["entailment"] = index
            elif "neutral" in label:
                self._label_indices["neutral"] = index
        if set(self._label_indices) != {"contradiction", "entailment", "neutral"}:
            raise ECADError(f"NLI model labels are not three-way: {labels}")

    def score_pairs(self, left_texts: Sequence[str], right_texts: Sequence[str]) -> list[PairScore]:
        if len(left_texts) != len(right_texts):
            raise ECADError("NLI pair arrays must have equal length")
        output: list[PairScore] = []
        for start in range(0, len(left_texts), self.batch_size):
            left = list(left_texts[start : start + self.batch_size])
            right = list(right_texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                left,
                right,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                logits = self.model(**encoded).logits
            probabilities = self._torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
            for row in probabilities:
                output.append(
                    PairScore(
                        "",
                        "",
                        float(row[self._label_indices["contradiction"]]),
                        float(row[self._label_indices["entailment"]]),
                        float(row[self._label_indices["neutral"]]),
                    )
                )
        return output


def build_pairwise_artifact(
    cases: Sequence[Mapping[str, Any]],
    model: FixedNLIModel,
) -> dict[str, Any]:
    """Compute all 1,225 rank-ordered pairs for every fixed case."""

    if len(cases) != 100:
        raise ECADError("ECAD replay is frozen to 100 cases")
    rows: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases, start=1):
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise ECADError(f"case {case_number} must contain Top-50")
        candidates = coerce_candidates(top50)
        pair_ids: list[tuple[str, str]] = []
        left_texts: list[str] = []
        right_texts: list[str] = []
        for left_index in range(EXPECTED_TOP50):
            for right_index in range(left_index + 1, EXPECTED_TOP50):
                pair_ids.append((candidates[left_index].passage_id, candidates[right_index].passage_id))
                left_texts.append(candidates[left_index].text)
                right_texts.append(candidates[right_index].text)
        scores = model.score_pairs(left_texts, right_texts)
        if len(scores) != EXPECTED_PAIR_COUNT:
            raise ECADError(f"NLI model returned invalid pair count for case {case_number}")
        pairs = []
        for (left_id, right_id), score in zip(pair_ids, scores):
            pairs.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "contradiction": round(_finite(score.contradiction, "contradiction"), 8),
                    "entailment": round(_finite(score.entailment, "entailment"), 8),
                    "neutral": round(_finite(score.neutral, "neutral"), 8),
                }
            )
        rows.append(
            {
                "case_number": case_number,
                "question_id": str(case.get("question_id", "")),
                "candidate_ids": [candidate.passage_id for candidate in candidates],
                "pairs": pairs,
            }
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "diagnostic_only": True,
        "protocol": {
            "case_count": len(cases),
            "top50_count": EXPECTED_TOP50,
            "pair_count_per_case": EXPECTED_PAIR_COUNT,
            "pair_order": "top50_position_i_lt_j",
            "candidate_text_order": "title_then_text",
            "gold_used": False,
            "silver_used": False,
            "generator_used": False,
        },
        "provenance": {
            **dict(model.provenance),
            "model_id": str(model.model_id),
            "revision": str(model.revision),
            "code_revision": _git_revision(),
        },
        "cases": rows,
    }
    validate_pairwise_artifact(artifact, cases)
    return artifact


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_CONFLICT_PENALTY",
    "DEFAULT_CONFLICT_THRESHOLD",
    "DEFAULT_K",
    "EXPECTED_PAIR_COUNT",
    "EXPECTED_TOP50",
    "ECADError",
    "FixedNLIModel",
    "NLI_MODEL_ID",
    "NLI_REVISION",
    "OnlineCandidate",
    "PairScore",
    "SCHEMA_VERSION",
    "SELECTOR_ID",
    "SELECTOR_VERSION",
    "SelectionResult",
    "build_pairwise_artifact",
    "coerce_candidates",
    "select",
    "validate_pairwise_artifact",
]
