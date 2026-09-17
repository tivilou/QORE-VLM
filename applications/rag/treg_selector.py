"""Top-k Residual Evidence Gain (TREG) selector and gain artifact builder.

TREG is intentionally isolated from the production selector.  A fixed DPR
Reader scores a deterministic four-passage base context (B4) and the same
context with one residual candidate appended.  The selector then keeps B4 and
uses the measured conditional gain to fill the remaining slot(s).

The gain artifact contains only candidate ids and numeric scores.  It must be
validated against the exact frozen Top-50 before it is consumed by a selector;
gold labels, generated answers, and evaluator output are never accepted at the
online boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import subprocess
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = "rag.treg_reader_gain.v1"
ARTIFACT_TYPE = "treg_reader_gain_artifact"
SELECTOR_ID = "treg_reader_gain"
SELECTOR_VERSION = "1.0"
READER_MODEL_ID = "facebook/dpr-reader-single-nq-base"
READER_REVISION = "38f47a4986084c53447ba92ab0a83076b58d86a8"
DEFAULT_K = 5
DEFAULT_B4_SIZE = 4
DEFAULT_PASSAGE_TOKEN_BUDGET = 64
DEFAULT_READER_MAX_LENGTH = 384

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


class TREGError(ValueError):
    """Raised when the TREG contract or artifact is invalid."""


@dataclass(frozen=True)
class OnlineCandidate:
    passage_id: str
    text: str
    retrieved_rank: int
    retrieval_score: float
    answer_scorer_score: float


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    base_indices: tuple[int, ...]
    residual_indices: tuple[int, ...]
    gains: tuple[float, ...]
    trace: tuple[dict[str, Any], ...]


class ContextReader(Protocol):
    """Minimal fixed Reader contract used by ``build_gain_artifact``."""

    model_id: str
    revision: str
    provenance: Mapping[str, Any]

    def pack_context(self, passages: Sequence[str]) -> str:
        ...

    def score_contexts(self, question: str, contexts: Sequence[str]) -> Sequence[float]:
        ...


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TREGError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TREGError(f"{field} must be a finite number")
    return result


def _candidate_text(value: Mapping[str, Any]) -> str:
    title = str(value.get("title") or "").strip()
    text = value.get("text", value.get("passage"))
    if not isinstance(text, str) or not text.strip():
        raise TREGError("candidate text must be non-empty")
    body = text.strip()
    return f"{title}. {body}" if title else body


def coerce_candidate(value: OnlineCandidate | Mapping[str, Any]) -> OnlineCandidate:
    if isinstance(value, OnlineCandidate):
        return value
    if not isinstance(value, Mapping):
        raise TREGError("candidate must be an object")
    keys = {str(key) for key in value}
    forbidden = sorted(keys & _FORBIDDEN_FIELDS)
    if forbidden:
        raise TREGError(f"forbidden selection fields: {', '.join(forbidden)}")
    unknown = sorted(keys - _ALLOWED_FIELDS)
    if unknown:
        raise TREGError(f"unknown selection fields: {', '.join(unknown)}")
    passage_id = value.get("id", value.get("passage_id"))
    if not isinstance(passage_id, (str, int)) or not str(passage_id).strip():
        raise TREGError("candidate id must be non-empty")
    rank = value.get("retrieved_rank", value.get("rank"))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise TREGError("retrieved_rank must be a positive integer")
    score = value.get("answer_scorer_score", value.get("answer_scorer"))
    if score is None:
        raise TREGError("answer_scorer_score is required")
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
        raise TREGError("at least one candidate is required")
    ids = [candidate.passage_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise TREGError("candidate passage IDs must be unique")
    return candidates


def base_indices(
    candidates: Sequence[OnlineCandidate | Mapping[str, Any]],
    *,
    b4_size: int = DEFAULT_B4_SIZE,
) -> tuple[int, ...]:
    """Return the deterministic B4 anchor indices."""

    rows = coerce_candidates(candidates)
    if isinstance(b4_size, bool) or not isinstance(b4_size, int) or b4_size < 1:
        raise TREGError("b4_size must be a positive integer")
    if len(rows) < b4_size:
        raise TREGError("candidate pool is smaller than B4")
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            -rows[index].answer_scorer_score,
            rows[index].retrieved_rank,
            rows[index].passage_id,
        ),
    )
    return tuple(order[:b4_size])


def select(
    candidates: Sequence[OnlineCandidate | Mapping[str, Any]],
    gains: Mapping[str, Any],
    *,
    k: int = DEFAULT_K,
    b4_size: int = DEFAULT_B4_SIZE,
) -> SelectionResult:
    """Select ``k`` passages using a validated residual gain map.

    B4 is selected by the frozen Answer Scorer tie-break order.  Residual
    candidates are ordered by Reader gain, then Answer Scorer, original rank,
    and id.  The returned order is the neutral original retrieval-rank order so
    downstream context ordering cannot create a method-specific advantage.
    """

    rows = coerce_candidates(candidates)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise TREGError("k must be a positive integer")
    if len(rows) < k:
        raise TREGError(f"need at least {k} candidates")
    if isinstance(b4_size, bool) or not isinstance(b4_size, int) or b4_size < 1:
        raise TREGError("b4_size must be a positive integer")
    b4_size = min(b4_size, k, len(rows))
    if not isinstance(gains, Mapping):
        raise TREGError("gains must be a mapping from passage id to gain")
    base = base_indices(rows, b4_size=b4_size)
    base_set = set(base)
    residual = [index for index in range(len(rows)) if index not in base_set]
    gain_by_index: dict[int, float] = {}
    for index in residual:
        identifier = rows[index].passage_id
        if identifier not in gains:
            raise TREGError(f"missing gain for residual candidate {identifier}")
        gain_by_index[index] = _finite(gains[identifier], f"gain[{identifier}]")
    residual.sort(
        key=lambda index: (
            -gain_by_index[index],
            -rows[index].answer_scorer_score,
            rows[index].retrieved_rank,
            rows[index].passage_id,
        )
    )
    chosen = list(base) + residual[: max(0, k - len(base))]
    chosen.sort(key=lambda index: (rows[index].retrieved_rank, rows[index].passage_id))
    selected_residual = [index for index in chosen if index not in base_set]
    trace = tuple(
        {
            "candidate_rank": rows[index].retrieved_rank,
            "gain": round(gain_by_index[index], 8),
            "answer_scorer_score": round(rows[index].answer_scorer_score, 8),
        }
        for index in selected_residual
    )
    return SelectionResult(
        selected_indices=tuple(chosen),
        base_indices=tuple(sorted(base, key=lambda index: (rows[index].retrieved_rank, rows[index].passage_id))),
        residual_indices=tuple(selected_residual),
        gains=tuple(gain_by_index[index] for index in selected_residual),
        trace=trace,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def validate_gain_artifact(
    payload: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_reader_model_id: str = READER_MODEL_ID,
    expected_reader_revision: str = READER_REVISION,
    k: int = DEFAULT_K,
    b4_size: int = DEFAULT_B4_SIZE,
) -> dict[int, dict[str, float]]:
    """Validate a gain artifact and return case-number keyed residual gains."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise TREGError(f"expected schema {SCHEMA_VERSION}")
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise TREGError("unexpected TREG artifact type")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TREGError("TREG provenance is required")
    if provenance.get("model_id") != expected_reader_model_id:
        raise TREGError("TREG Reader model id is not frozen")
    if provenance.get("revision") != expected_reader_revision:
        raise TREGError("TREG Reader revision is not frozen")
    config_hash = str(provenance.get("config_sha256", ""))
    if len(config_hash) != 64 or any(char not in "0123456789abcdef" for char in config_hash.lower()):
        raise TREGError("TREG Reader config_sha256 is invalid")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TREGError("TREG protocol is required")
    if int(protocol.get("case_count", -1)) != len(cases):
        raise TREGError("TREG case count does not match input")
    if int(protocol.get("top50_count", -1)) != 50:
        raise TREGError("TREG Top-50 count is not frozen")
    if int(protocol.get("k", -1)) != k or int(protocol.get("b4_size", -1)) != b4_size:
        raise TREGError("TREG K/B4 protocol is not frozen")
    if protocol.get("gain_definition") != "Reader(q,B4+p)-Reader(q,B4)":
        raise TREGError("TREG gain definition is not frozen")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(cases):
        raise TREGError("TREG artifact case rows are incomplete")
    by_number: dict[int, Mapping[str, Any]] = {}
    for item in raw_cases:
        if not isinstance(item, Mapping):
            raise TREGError("TREG case row must be an object")
        number = int(item.get("case_number", -1))
        if number in by_number:
            raise TREGError(f"duplicate TREG case number {number}")
        by_number[number] = item
    result: dict[int, dict[str, float]] = {}
    for expected_number, case in enumerate(cases, start=1):
        if expected_number not in by_number:
            raise TREGError(f"TREG case {expected_number} is missing")
        item = by_number[expected_number]
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != 50:
            raise TREGError(f"input case {expected_number} does not contain Top-50")
        candidates = coerce_candidates(top50)
        expected_ids = [candidate.passage_id for candidate in candidates]
        if item.get("candidate_ids") != expected_ids:
            raise TREGError(f"TREG candidate identity/order mismatch in case {expected_number}")
        expected_base = [candidates[index].passage_id for index in base_indices(candidates, b4_size=b4_size)]
        if item.get("b4_ids") != expected_base:
            raise TREGError(f"TREG B4 mismatch in case {expected_number}")
        raw_gains = item.get("gains")
        if not isinstance(raw_gains, list):
            raise TREGError(f"TREG gains are missing in case {expected_number}")
        expected_residual = [identifier for identifier in expected_ids if identifier not in set(expected_base)]
        if [row.get("id") for row in raw_gains if isinstance(row, Mapping)] != expected_residual:
            raise TREGError(f"TREG residual gain order mismatch in case {expected_number}")
        case_gains: dict[str, float] = {}
        for row in raw_gains:
            if not isinstance(row, Mapping):
                raise TREGError("TREG gain row must be an object")
            identifier = str(row.get("id", ""))
            base_score = _finite(row.get("base_score"), f"TREG base_score[{identifier}]")
            with_score = _finite(row.get("with_candidate_score"), f"TREG with_score[{identifier}]")
            gain = _finite(row.get("gain"), f"TREG gain[{identifier}]")
            if not math.isclose(gain, with_score - base_score, rel_tol=1e-6, abs_tol=1e-6):
                raise TREGError(f"TREG gain arithmetic mismatch for {identifier}")
            case_gains[identifier] = gain
        if len(case_gains) != len(expected_residual):
            raise TREGError(f"TREG gain coverage mismatch in case {expected_number}")
        result[expected_number] = case_gains
    return result


class FixedDPRReader:
    """Fixed provenance DPR Reader used to build TREG gains."""

    model_id = READER_MODEL_ID
    revision = READER_REVISION

    def __init__(
        self,
        *,
        model_id: str = READER_MODEL_ID,
        revision: str = READER_REVISION,
        device: str | None = None,
        batch_size: int = 8,
        passage_token_budget: int = DEFAULT_PASSAGE_TOKEN_BUDGET,
        max_length: int = DEFAULT_READER_MAX_LENGTH,
    ) -> None:
        if model_id != READER_MODEL_ID or revision != READER_REVISION:
            raise TREGError("TREG Reader identity is fixed")
        if batch_size < 1 or passage_token_budget < 1 or max_length < 32:
            raise TREGError("invalid Reader batching or token budget")
        try:
            import torch
            from transformers import DPRReader, DPRReaderTokenizer
        except ImportError as exc:
            raise TREGError("TREG Reader requires torch and transformers") from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.passage_token_budget = int(passage_token_budget)
        self.max_length = int(max_length)
        self.tokenizer = DPRReaderTokenizer.from_pretrained(model_id, revision=revision)
        self.model = DPRReader.from_pretrained(model_id, revision=revision).to(self.device)
        self.model.eval()
        config_json = self.model.config.to_json_string().encode("utf-8")
        tokenizer_json = repr(sorted(self.tokenizer.init_kwargs.items())).encode("utf-8")
        self.provenance = {
            "model_id": model_id,
            "revision": revision,
            "config_sha256": _sha256_bytes(config_json),
            "tokenizer_config_sha256": _sha256_bytes(tokenizer_json),
            "device": self.device,
            "batch_size": self.batch_size,
            "passage_token_budget": self.passage_token_budget,
            "max_length": self.max_length,
            "score": "sigmoid(relevance_logits)",
        }

    def _truncate(self, text: str) -> str:
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = encoded.get("input_ids", [])
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(value) for value in ids[: self.passage_token_budget]]
        if not ids:
            return text[:1]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def pack_context(self, passages: Sequence[str]) -> str:
        return "\n\n".join(
            f"[{index}] {self._truncate(str(passage))}"
            for index, passage in enumerate(passages, start=1)
        )

    def score_contexts(self, question: str, contexts: Sequence[str]) -> list[float]:
        if not contexts:
            return []
        scores: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            batch = list(contexts[start : start + self.batch_size])
            encoded = self.tokenizer(
                questions=[question] * len(batch),
                texts=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                output = self.model(**encoded)
            logits = output.relevance_logits.detach().float().cpu().numpy()
            scores.extend(float(1.0 / (1.0 + math.exp(-max(min(float(value), 60.0), -60.0)))) for value in logits)
        return scores


def build_gain_artifact(
    cases: Sequence[Mapping[str, Any]],
    reader: ContextReader,
    *,
    k: int = DEFAULT_K,
    b4_size: int = DEFAULT_B4_SIZE,
) -> dict[str, Any]:
    """Compute and validate the fixed-Reader residual gain artifact."""

    if len(cases) != 100:
        raise TREGError("TREG replay is frozen to 100 cases")
    rows: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases, start=1):
        question = case.get("question")
        top50 = case.get("top_50")
        if not isinstance(question, str) or not question.strip():
            raise TREGError(f"case {case_number} question is empty")
        if not isinstance(top50, list) or len(top50) != 50:
            raise TREGError(f"case {case_number} must contain Top-50")
        candidates = coerce_candidates(top50)
        base = base_indices(candidates, b4_size=b4_size)
        base_ids = [candidates[index].passage_id for index in base]
        base_rows = sorted((candidates[index] for index in base), key=lambda item: (item.retrieved_rank, item.passage_id))
        residual = [candidate for index, candidate in enumerate(candidates) if index not in set(base)]
        base_context = reader.pack_context([candidate.text for candidate in base_rows])
        contexts = [base_context]
        for candidate in residual:
            ordered = list(base_rows) + [candidate]
            contexts.append(reader.pack_context([item.text for item in ordered]))
        scores = list(reader.score_contexts(str(question), contexts))
        if len(scores) != 1 + len(residual):
            raise TREGError(f"Reader returned invalid score count for case {case_number}")
        base_score = _finite(scores[0], f"Reader base score case {case_number}")
        gains = []
        for candidate, score in zip(residual, scores[1:]):
            with_score = _finite(score, f"Reader candidate score {candidate.passage_id}")
            gains.append(
                {
                    "id": candidate.passage_id,
                    "base_score": round(base_score, 10),
                    "with_candidate_score": round(with_score, 10),
                    "gain": round(with_score - base_score, 10),
                }
            )
        rows.append(
            {
                "case_number": case_number,
                "question_id": str(case.get("question_id", "")),
                "candidate_ids": [candidate.passage_id for candidate in candidates],
                "b4_ids": base_ids,
                "gains": gains,
            }
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "diagnostic_only": True,
        "protocol": {
            "case_count": len(cases),
            "top50_count": 50,
            "k": k,
            "b4_size": b4_size,
            "b4_definition": "top Answer Scorer candidates, ties by retrieved_rank then id",
            "gain_definition": "Reader(q,B4+p)-Reader(q,B4)",
            "context_order": "ascending_original_retrieval_rank",
            "residual_order": "candidate Top-50 order",
            "gold_used": False,
            "silver_used": False,
        },
        "provenance": {
            **dict(reader.provenance),
            "model_id": str(reader.model_id),
            "revision": str(reader.revision),
            "code_revision": _git_revision(),
        },
        "cases": rows,
    }
    validate_gain_artifact(
        artifact,
        cases,
        expected_reader_model_id=READER_MODEL_ID,
        expected_reader_revision=READER_REVISION,
        k=k,
        b4_size=b4_size,
    )
    return artifact


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_B4_SIZE",
    "DEFAULT_K",
    "FixedDPRReader",
    "OnlineCandidate",
    "READER_MODEL_ID",
    "READER_REVISION",
    "SCHEMA_VERSION",
    "SELECTOR_ID",
    "SELECTOR_VERSION",
    "SelectionResult",
    "TREGError",
    "base_indices",
    "build_gain_artifact",
    "coerce_candidates",
    "select",
    "validate_gain_artifact",
]
