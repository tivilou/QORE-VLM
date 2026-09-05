"""Canonical no-GPU preflight for the Q-DES selector hypothesis.

This module is an observation-only probe.  It validates the detailed 50-case
artifact, computes a deterministic typed question/passage signal, and compares
that signal with generic relevance signals.  It does not run a selector,
retriever, Generator, evaluator, model, or panel call.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .question_slot_parser import QuestionSlotParser, QuestionSlots


SCHEMA_VERSION = "qdes_no_gpu_preflight.v2"
EXPECTED_CASE_COUNT = 50
EXPECTED_TOP50_COUNT = 50
PROXY_TOP_K = 5
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

_TYPE_CUES = {
    "person": {"person", "people", "man", "woman", "men", "women", "actor", "author", "writer", "president", "king", "queen", "scientist", "leader"},
    "location": {"place", "location", "country", "city", "state", "region", "island", "river", "capital", "province"},
    "date": {"year", "date", "time", "century", "decade", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"},
    "quantity": {"number", "count", "total", "many", "much", "million", "billion", "percent", "percentage"},
    "explanation": {"because", "reason", "cause", "due", "result", "purpose"},
    "procedure": {"how", "method", "way", "process", "used", "use"},
    "entity": {"name", "title", "book", "film", "movie", "song", "term", "definition"},
}

_OPERATOR_CUES = {
    "who": {"who", "whom", "whose"},
    "where": {"where", "located", "place", "country", "city"},
    "when": {"when", "year", "date", "time", "century"},
    "why": {"why", "because", "reason", "cause"},
    "how": {"how", "method", "way", "process"},
    "count": {"number", "count", "total", "many", "much"},
    "what": {"what", "name", "title", "definition"},
    "which": {"which", "first", "last", "largest", "smallest", "oldest", "youngest"},
}


class PreflightError(ValueError):
    """Raised when the input cannot satisfy the fixed preflight contract."""


@dataclass(frozen=True)
class CandidateView:
    """The only candidate fields visible to the typed signal."""

    passage_id: str
    text: str
    retrieved_rank: int
    retrieval_score: float
    answer_scorer_score: float


def _tokens(value: str) -> List[str]:
    return _TOKEN_RE.findall(value.lower())


def _token_set(value: str) -> set[str]:
    return set(_tokens(value))


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PreflightError(f"{field} must be a finite number")
    return number


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _git_revision() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _positive_consensus(candidate: Dict[str, Any]) -> Optional[bool]:
    """Read the silver label only in the offline evaluation branch."""

    evidence = candidate.get("evidence")
    if not isinstance(evidence, dict):
        return None
    value = evidence.get("positive_consensus")
    if isinstance(value, bool):
        return value
    models = evidence.get("models")
    if not isinstance(models, dict):
        return None
    positive_votes = 0
    observed = 0
    for judgment in models.values():
        if not isinstance(judgment, dict):
            continue
        label = judgment.get("label")
        if isinstance(label, str):
            observed += 1
            if label in {"direct", "partial"}:
                positive_votes += 1
    return positive_votes >= 2 if observed else None


def _question_positions(root: Dict[str, Any], cases: Sequence[Dict[str, Any]]) -> List[int]:
    protocol = root.get("protocol")
    positions = protocol.get("question_global_indices") if isinstance(protocol, dict) else None
    if positions is not None:
        if not isinstance(positions, list) or len(positions) != EXPECTED_CASE_COUNT:
            raise PreflightError("protocol.question_global_indices must contain 50 positions")
        parsed = []
        for index, value in enumerate(positions, 1):
            if isinstance(value, bool) or not isinstance(value, int):
                raise PreflightError(f"question_global_indices[{index}] must be an integer")
            parsed.append(value)
        return parsed
    return [
        int(case.get("dataset_position", case.get("case_number", index)))
        for index, case in enumerate(cases, 1)
    ]


def _validate_cases(root: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    cases = root.get("cases")
    if not isinstance(cases, list):
        raise PreflightError("expected detailed case-study root field 'cases'")
    if len(cases) != EXPECTED_CASE_COUNT:
        raise PreflightError(f"expected exactly {EXPECTED_CASE_COUNT} cases, got {len(cases)}")

    positions = _question_positions(root, cases)
    normalized: List[Dict[str, Any]] = []
    qid_hashes: List[str] = []
    seen_ids: set[str] = set()
    case_numbers: List[int] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise PreflightError(f"cases[{index}] must be an object")
        case_number = case.get("case_number")
        if case_number != index + 1:
            raise PreflightError("case_number must be the fixed sequence 1..50")
        case_numbers.append(case_number)
        question_id = case.get("question_id")
        question = case.get("question")
        if not isinstance(question_id, str) or not question_id.strip():
            raise PreflightError(f"cases[{index}].question_id must be non-empty")
        if not isinstance(question, str) or not question.strip():
            raise PreflightError(f"cases[{index}].question must be non-empty")
        if question_id in seen_ids:
            raise PreflightError(f"duplicate question_id: {question_id}")
        seen_ids.add(question_id)
        qid_hashes.append(_sha256_text(question_id))

        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50_COUNT:
            raise PreflightError(f"cases[{index}].top_50 must contain exactly 50 candidates")
        candidates: List[CandidateView] = []
        labels: List[Optional[bool]] = []
        seen_passages: set[str] = set()
        ranks: List[int] = []
        nonempty_text = 0
        for passage_index, item in enumerate(top50):
            if not isinstance(item, dict):
                raise PreflightError(f"cases[{index}].top_50[{passage_index}] must be an object")
            passage_id = item.get("id", item.get("passage_id"))
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                text = item.get("passage")
            if not isinstance(passage_id, (str, int)) or not str(passage_id).strip():
                raise PreflightError(f"cases[{index}].top_50[{passage_index}].id must be non-empty")
            passage_id = str(passage_id)
            if passage_id in seen_passages:
                raise PreflightError(f"duplicate passage id in case {index + 1}: {passage_id}")
            seen_passages.add(passage_id)
            if not isinstance(text, str) or not text.strip():
                raise PreflightError(f"empty passage text in case {index + 1}, rank {passage_index + 1}")
            nonempty_text += 1
            rank_value = item.get("retrieved_rank", item.get("rank", passage_index + 1))
            if isinstance(rank_value, bool) or not isinstance(rank_value, int):
                raise PreflightError(f"retrieved_rank must be an integer in case {index + 1}")
            ranks.append(rank_value)
            candidates.append(CandidateView(
                passage_id=passage_id,
                text=text,
                retrieved_rank=rank_value,
                retrieval_score=_finite_number(item.get("retrieval_score", 0.0), "retrieval_score"),
                answer_scorer_score=_finite_number(item.get("answer_scorer_score", item.get("answer_scorer", 0.0)), "answer_scorer_score"),
            ))
            labels.append(_positive_consensus(item))
        if sorted(ranks) != list(range(1, EXPECTED_TOP50_COUNT + 1)):
            raise PreflightError(f"case {index + 1} retrieved ranks must be exactly 1..50")
        normalized.append({
            "case_number": case_number,
            "question_id": question_id,
            "question": question,
            "candidates": candidates,
            "labels": labels,
            "nonempty_text": nonempty_text,
        })

    if case_numbers != list(range(1, EXPECTED_CASE_COUNT + 1)):
        raise PreflightError("fixed case numbering is not contiguous")
    return normalized, positions, qid_hashes


def typed_coverage(question_slots: QuestionSlots, passage_text: str) -> float:
    """Compute typed coverage from question-only slots and passage text."""

    passage_tokens = _token_set(passage_text)
    if not passage_tokens or not question_slots.success:
        return 0.0
    slots = question_slots.slots
    subject_tokens = set(_tokens(slots.get("subject", "")))
    relation_tokens = set(_tokens(slots.get("relation", "")))
    subject = _coverage(subject_tokens, passage_tokens)
    relation = _coverage(relation_tokens, passage_tokens)

    answer_type = slots.get("answer_type", "")
    type_cues = _TYPE_CUES.get(answer_type, set())
    type_score = _coverage(type_cues, passage_tokens) if type_cues else 0.0

    operator = slots.get("operator", "")
    operator_cues = _OPERATOR_CUES.get(operator, set())
    operator_score = _coverage(operator_cues, passage_tokens) if operator_cues else 0.0

    # Relation and subject carry the question-specific anchors; type/operator
    # cues add an explicit typed component without consulting any label.
    return min(1.0, 0.45 * relation + 0.25 * subject + 0.20 * type_score + 0.10 * operator_score)


def _coverage(needles: Iterable[str], haystack: set[str]) -> float:
    values = {token for token in needles if token}
    return (sum(token in haystack for token in values) / len(values)) if values else 0.0


def generic_question_overlap(question: str, passage_text: str) -> float:
    """A lexical relevance control using the whole question, untyped."""

    question_tokens = {token for token in _tokens(question) if len(token) > 2}
    return _coverage(question_tokens, _token_set(passage_text))


def _rank_auc(scores: Sequence[float], labels: Sequence[Optional[bool]]) -> Optional[float]:
    pairs = [(float(score), bool(label)) for score, label in zip(scores, labels) if label is not None]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(pairs), key=lambda item: (item[1][0], item[0]))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        score = ordered[index][1][0]
        while end < len(ordered) and ordered[end][1][0] == score:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += sum(average_rank for _, (_, label) in ordered[index:end] if label)
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _discordance(left: Sequence[float], right: Sequence[float]) -> float:
    comparable = 0
    discordant = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            left_delta = left[i] - left[j]
            right_delta = right[i] - right[j]
            if abs(left_delta) < 1e-12 or abs(right_delta) < 1e-12:
                continue
            comparable += 1
            if left_delta * right_delta < 0:
                discordant += 1
    return discordant / comparable if comparable else 0.0


def _top_k_indices(scores: Sequence[float], candidates: Sequence[CandidateView]) -> List[int]:
    return sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), candidates[index].retrieved_rank, candidates[index].passage_id),
    )[:PROXY_TOP_K]


def _summarize(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    present = [float(value) for value in values if value is not None]
    return {
        "mean": (sum(present) / len(present)) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
        "n": len(present),
    }


def analyze(root: Dict[str, Any], input_sha256: Optional[str] = None) -> Dict[str, Any]:
    cases, positions, qid_hashes = _validate_cases(root)
    parser = QuestionSlotParser()
    per_question: List[Dict[str, Any]] = []
    parser_successes = 0
    typed_scores_all: List[float] = []
    generic_scores_all: List[float] = []
    answer_scores_all: List[float] = []
    retrieval_scores_all: List[float] = []
    labels_all: List[Optional[bool]] = []
    typed_aucs: List[Optional[float]] = []
    generic_aucs: List[Optional[float]] = []
    answer_aucs: List[Optional[float]] = []
    retrieval_aucs: List[Optional[float]] = []
    typed_proxy_hits = 0
    answer_proxy_hits = 0
    nondegenerate_cases = 0
    disagreement_cases = 0
    evidence_label_cases = 0
    evidence_label_count_total = 0

    for case, position, qid_hash in zip(cases, positions, qid_hashes):
        slots = parser.parse(case["question"])
        parser_successes += int(slots.success)
        candidates: List[CandidateView] = case["candidates"]
        labels: List[Optional[bool]] = case["labels"]
        if any(label is not None for label in labels):
            evidence_label_cases += 1
        evidence_label_count_total += sum(label is not None for label in labels)
        typed_scores = [typed_coverage(slots, candidate.text) for candidate in candidates]
        generic_scores = [generic_question_overlap(case["question"], candidate.text) for candidate in candidates]
        answer_scores = [candidate.answer_scorer_score for candidate in candidates]
        retrieval_scores = [candidate.retrieval_score for candidate in candidates]
        typed_scores_all.extend(typed_scores)
        generic_scores_all.extend(generic_scores)
        answer_scores_all.extend(answer_scores)
        retrieval_scores_all.extend(retrieval_scores)
        labels_all.extend(labels)

        typed_auc = _rank_auc(typed_scores, labels)
        generic_auc = _rank_auc(generic_scores, labels)
        answer_auc = _rank_auc(answer_scores, labels)
        retrieval_auc = _rank_auc(retrieval_scores, labels)
        typed_aucs.append(typed_auc)
        generic_aucs.append(generic_auc)
        answer_aucs.append(answer_auc)
        retrieval_aucs.append(retrieval_auc)

        typed_indices = _top_k_indices(typed_scores, candidates)
        answer_indices = _top_k_indices(answer_scores, candidates)
        typed_positive = sum(bool(labels[index]) for index in typed_indices if labels[index] is not None)
        answer_positive = sum(bool(labels[index]) for index in answer_indices if labels[index] is not None)
        typed_proxy_hits += int(typed_positive > 0)
        answer_proxy_hits += int(answer_positive > 0)
        typed_var = _variance(typed_scores)
        disagreement = _discordance(typed_scores, answer_scores)
        nondegenerate_cases += int(typed_var > 1e-8)
        disagreement_cases += int(disagreement > 0.05)

        per_question.append({
            "case_number": case["case_number"],
            "dataset_position": position,
            "question_id_sha256": qid_hash,
            "parser_success": slots.success,
            "parser_confidence": round(slots.parser_confidence, 6),
            "slot_types": sorted(slots.slots.keys()),
            "nonempty_top50_text_count": case["nonempty_text"],
            "typed_coverage_nonzero_fraction": round(sum(score > 0 for score in typed_scores) / len(typed_scores), 6),
            "typed_coverage_variance": round(typed_var, 8),
            "typed_answer_scorer_disagreement": round(disagreement, 6),
            "evidence_label_count": sum(label is not None for label in labels),
            "positive_consensus_count": sum(bool(label) for label in labels if label is not None),
            "typed_proxy_top5_positive_count": typed_positive,
            "answer_scorer_proxy_top5_positive_count": answer_positive,
            "typed_auc": typed_auc,
            "generic_overlap_auc": generic_auc,
            "answer_scorer_auc": answer_auc,
            "retrieval_auc": retrieval_auc,
        })

    typed_auc_all = _rank_auc(typed_scores_all, labels_all)
    generic_auc_all = _rank_auc(generic_scores_all, labels_all)
    answer_auc_all = _rank_auc(answer_scores_all, labels_all)
    retrieval_auc_all = _rank_auc(retrieval_scores_all, labels_all)
    parser_success_rate = parser_successes / EXPECTED_CASE_COUNT
    typed_nonzero_rate = sum(score > 0 for score in typed_scores_all) / len(typed_scores_all)
    typed_auc_gain = (typed_auc_all - answer_auc_all) if typed_auc_all is not None and answer_auc_all is not None else None
    generic_auc_gain = (typed_auc_all - generic_auc_all) if typed_auc_all is not None and generic_auc_all is not None else None
    gates = {
        "schema_gate": {
            "pass": True,
            "reason": "50 contiguous cases with 50 unique ranked candidates each",
        },
        "parser_gate": {
            "pass": parser_success_rate >= 0.80,
            "value": parser_success_rate,
            "threshold": 0.80,
            "reason": "question-only typed parse; no unconditional fallback",
        },
        "typed_signal_gate": {
            "pass": typed_nonzero_rate >= 0.50 and nondegenerate_cases / EXPECTED_CASE_COUNT >= 0.80,
            "typed_nonzero_rate": typed_nonzero_rate,
            "nondegenerate_case_rate": nondegenerate_cases / EXPECTED_CASE_COUNT,
            "thresholds": {"typed_nonzero_rate": 0.50, "nondegenerate_case_rate": 0.80},
        },
        "separability_gate": {
            "pass": bool(typed_auc_all is not None and typed_auc_all >= 0.55 and typed_auc_gain is not None and typed_auc_gain >= 0.02),
            "typed_auc": typed_auc_all,
            "answer_scorer_auc": answer_auc_all,
            "typed_gain_vs_answer_scorer": typed_auc_gain,
            "minimum_typed_auc": 0.55,
            "minimum_gain": 0.02,
        },
        "distinctness_gate": {
            "pass": disagreement_cases / EXPECTED_CASE_COUNT >= 0.50 and generic_auc_gain is not None,
            "cases_with_material_order_disagreement": disagreement_cases,
            "material_disagreement_case_rate": disagreement_cases / EXPECTED_CASE_COUNT,
            "typed_gain_vs_generic_overlap": generic_auc_gain,
            "threshold": 0.50,
        },
        "label_availability_gate": {
            "pass": evidence_label_count_total == EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT and typed_auc_all is not None,
            "cases_with_passage_labels": evidence_label_cases,
            "labeled_candidates": evidence_label_count_total,
            "required_candidates": EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT,
            "required_cases": EXPECTED_CASE_COUNT,
        },
        "leakage_gate": {
            "pass": True,
            "selection_feedback": False,
            "panel_feedback": False,
            "gold_feedback": False,
            "generator_feedback": False,
            "evaluator_feedback": False,
        },
    }
    gates["overall"] = {"pass": all(bool(gate.get("pass")) for name, gate in gates.items() if name != "overall")}

    metrics = {
        "case_count": EXPECTED_CASE_COUNT,
        "top50_count_per_case": EXPECTED_TOP50_COUNT,
        "parser_success_count": parser_successes,
        "parser_success_rate": parser_success_rate,
        "parser_fallback_count": EXPECTED_CASE_COUNT - parser_successes,
        "parser_fallback_rate": 1.0 - parser_success_rate,
        "typed_coverage_nonzero_rate": typed_nonzero_rate,
        "typed_coverage_nondegenerate_case_rate": nondegenerate_cases / EXPECTED_CASE_COUNT,
        "typed_answer_scorer_material_disagreement_case_rate": disagreement_cases / EXPECTED_CASE_COUNT,
        "passage_label_coverage_case_rate": evidence_label_cases / EXPECTED_CASE_COUNT,
        "passage_label_coverage_candidate_rate": evidence_label_count_total / (EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT),
        "auc_micro": {
            "typed_coverage": typed_auc_all,
            "generic_question_overlap": generic_auc_all,
            "answer_scorer": answer_auc_all,
            "retrieval_score": retrieval_auc_all,
        },
        "auc_macro": {
            "typed_coverage": _summarize(typed_aucs),
            "generic_question_overlap": _summarize(generic_aucs),
            "answer_scorer": _summarize(answer_aucs),
            "retrieval_score": _summarize(retrieval_aucs),
        },
        "typed_gain_vs_answer_scorer": typed_auc_gain,
        "typed_gain_vs_generic_overlap": generic_auc_gain,
        "proxy_top5_positive_case_rate": {
            "typed_coverage": typed_proxy_hits / EXPECTED_CASE_COUNT,
            "answer_scorer": answer_proxy_hits / EXPECTED_CASE_COUNT,
        },
    }
    compact_core = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "gates": gates,
        "question_id_sha256": qid_hashes,
        "dataset_positions": positions,
        "per_question": per_question,
    }
    replay_digest = _sha256_text(_stable_json(compact_core))
    return {
        **compact_core,
        "replay_digest": replay_digest,
        "input_sha256": input_sha256,
        "consumed_selection_fields": ["question", "top_50[].id", "top_50[].text", "top_50[].retrieved_rank", "top_50[].retrieval_score", "top_50[].answer_scorer_score"],
        "offline_evaluation_fields": ["top_50[].evidence.positive_consensus"],
        "forbidden_selection_fields": ["gold_answers", "top_50[].evidence", "selectors", "prediction", "generation_panel", "metrics"],
    }


def run_preflight(case_study_path: str | Path, panel_analysis_path: Optional[str | Path] = None, output_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Run the canonical preflight and optionally write one compact JSON file."""

    path = Path(case_study_path)
    raw = path.read_bytes()
    try:
        root = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse UTF-8 JSON: {path}") from exc
    if not isinstance(root, dict):
        raise PreflightError("case-study root must be a JSON object")
    result = analyze(root, _sha256_bytes(raw))
    if panel_analysis_path is not None:
        panel_path = Path(panel_analysis_path)
        try:
            panel_root = json.loads(panel_path.read_text(encoding="utf-8"))
            panel_cases = panel_root.get("per_case") if isinstance(panel_root, dict) else None
            result["panel_schema_check"] = {
                "provided": True,
                "sha256": _sha256_bytes(panel_path.read_bytes()),
                "case_count": len(panel_cases) if isinstance(panel_cases, list) else None,
                "used_for_selection": False,
                "pass": isinstance(panel_cases, list) and len(panel_cases) == EXPECTED_CASE_COUNT,
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            result["panel_schema_check"] = {"provided": True, "used_for_selection": False, "pass": False}
    else:
        result["panel_schema_check"] = {"provided": False, "used_for_selection": False, "pass": True}
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    return result


def write_artifacts(result: Dict[str, Any], output_dir: str | Path, input_path: str | Path) -> None:
    """Write compact result, summary, and provenance files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "replay_digest": result["replay_digest"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "panel_schema_check": result.get("panel_schema_check"),
    }
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_study_path_name": Path(input_path).name,
        "case_study_sha256": result.get("input_sha256"),
        "code_revision": _git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_used": False,
        "model_used": False,
        "selection_executed": False,
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
