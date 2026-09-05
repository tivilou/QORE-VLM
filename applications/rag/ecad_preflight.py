"""Canonical no-GPU preflight for Evidence Conflict-Aware Diversification.

The preflight is intentionally stricter than an ECAD selector prototype.  A
pairwise NLI score artifact is required for the conflict gate.  Without it, a
deterministic lexical proxy is reported as a negative control and cannot pass.
No conflict score is fed to the production selector.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .qdes_preflight import (
    CandidateView,
    EXPECTED_CASE_COUNT,
    EXPECTED_TOP50_COUNT,
    PreflightError,
    _rank_auc,
    _sha256_bytes,
    _stable_json,
    _validate_cases,
)


SCHEMA_VERSION = "ecad_no_gpu_preflight.v1"
PAIRWISE_NLI_SCHEMA = "ecad_pairwise_nli.v1"
CONFLICT_THRESHOLD = 0.50
CONFLICT_PENALTY = 0.20
MIN_AUC = 0.55
MIN_AUC_GAIN = 0.02
MIN_SELECTION_CHANGE_RATE = 0.20
MIN_CONFLICT_REDUCTION = 0.30
MAX_POSITIVE_RETENTION_DROP = 0.05
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_NEGATION = {"not", "no", "never", "none", "without", "neither", "n't"}
_OPPOSITE_PAIRS = (
    ("increase", "decrease"), ("increased", "decreased"),
    ("before", "after"), ("early", "late"), ("first", "last"),
    ("largest", "smallest"), ("more", "less"), ("won", "lost"),
    ("true", "false"), ("alive", "dead"), ("male", "female"),
)


@dataclass(frozen=True)
class PairScore:
    left: str
    right: str
    contradiction: float
    entailment: float
    neutral: float


def _tokens(value: str) -> Set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _number_tokens(value: str) -> Set[str]:
    return {token for token in _TOKEN_RE.findall(value.lower()) if any(char.isdigit() for char in token)}


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"{field} must be a finite probability")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PreflightError(f"{field} must be in [0, 1]")
    return number


def _git_revision() -> Optional[str]:
    override = os.environ.get("ECAD_CODE_REVISION")
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _contradictory_label(candidate: Dict[str, Any]) -> Optional[bool]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, dict):
        return None
    consensus = evidence.get("consensus_label")
    if isinstance(consensus, str):
        return consensus == "contradictory"
    models = evidence.get("models")
    if not isinstance(models, dict):
        return None
    observed = 0
    votes = 0
    for judgment in models.values():
        if not isinstance(judgment, dict):
            continue
        label = judgment.get("label")
        if isinstance(label, str):
            observed += 1
            votes += int(label == "contradictory")
    return votes >= 2 if observed else None


def _positive_label(candidate: Dict[str, Any]) -> Optional[bool]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, dict):
        return None
    value = evidence.get("positive_consensus")
    if isinstance(value, bool):
        return value
    models = evidence.get("models")
    if not isinstance(models, dict):
        return None
    observed = 0
    votes = 0
    for judgment in models.values():
        if not isinstance(judgment, dict):
            continue
        label = judgment.get("label")
        if isinstance(label, str):
            observed += 1
            votes += int(label in {"direct", "partial"})
    return votes >= 2 if observed else None


def _lexical_conflict(left: str, right: str) -> float:
    """Conservative proxy; never eligible for the NLI gate."""
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    anchors = left_tokens & right_tokens
    if not anchors:
        return 0.0
    score = 0.0
    if (_number_tokens(left) & _number_tokens(right)) == set() and (_number_tokens(left) or _number_tokens(right)):
        score += 0.35
    if bool(left_tokens & _NEGATION) != bool(right_tokens & _NEGATION):
        score += 0.25
    for first, second in _OPPOSITE_PAIRS:
        if (first in left_tokens and second in right_tokens) or (second in left_tokens and first in right_tokens):
            score += 0.40
    return min(1.0, score * min(1.0, len(anchors) / 3.0))


def _load_pairwise_nli(path: Optional[Path]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if path is None:
        return None, {"provided": False, "pass": False, "reason": "pairwise NLI artifact is required"}
    try:
        raw = path.read_bytes()
        root = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, {"provided": True, "pass": False, "reason": f"cannot parse pairwise NLI artifact: {exc}"}
    if not isinstance(root, dict) or root.get("schema_version") != PAIRWISE_NLI_SCHEMA:
        return None, {"provided": True, "pass": False, "reason": f"expected schema {PAIRWISE_NLI_SCHEMA}"}
    provenance = root.get("provenance")
    provenance_ok = (
        isinstance(provenance, dict)
        and isinstance(provenance.get("model_id"), str) and bool(provenance["model_id"])
        and isinstance(provenance.get("revision"), str) and bool(provenance["revision"])
        and isinstance(provenance.get("config_sha256"), str) and len(provenance["config_sha256"]) == 64
    )
    if not provenance_ok:
        return None, {"provided": True, "pass": False, "reason": "model_id, revision, and config_sha256 are required"}
    cases = root.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        return None, {"provided": True, "pass": False, "reason": "pairwise artifact must contain 50 cases"}
    return root, {
        "provided": True,
        "pass": True,
        "sha256": _sha256_bytes(raw),
        "model_id": provenance["model_id"],
        "revision": provenance["revision"],
        "config_sha256": provenance["config_sha256"],
    }


def _pair_scores_for_case(
    pair_root: Optional[Dict[str, Any]],
    case_number: int,
    candidates: Sequence[CandidateView],
) -> Tuple[Dict[Tuple[str, str], PairScore], Dict[str, Any]]:
    expected_ids = {candidate.passage_id for candidate in candidates}
    pair_map: Dict[Tuple[str, str], PairScore] = {}
    if pair_root is None:
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1:]:
                key = tuple(sorted((left.passage_id, right.passage_id)))
                pair_map[key] = PairScore(
                    key[0], key[1], _lexical_conflict(left.text, right.text), 0.0, 0.0
                )
        return pair_map, {"source": "lexical_proxy", "valid": True}

    entries = pair_root.get("cases", [])[case_number - 1]
    if not isinstance(entries, dict) or entries.get("case_number") != case_number:
        raise PreflightError(f"pairwise NLI case {case_number} has invalid case_number")
    pairs = entries.get("pairs")
    expected_pair_count = EXPECTED_TOP50_COUNT * (EXPECTED_TOP50_COUNT - 1) // 2
    if not isinstance(pairs, list) or len(pairs) != expected_pair_count:
        raise PreflightError(f"pairwise NLI case {case_number} must contain {expected_pair_count} pairs")
    for index, item in enumerate(pairs):
        if not isinstance(item, dict):
            raise PreflightError(f"pairwise NLI case {case_number} pair {index} must be an object")
        left = item.get("left_id", item.get("i"))
        right = item.get("right_id", item.get("j"))
        if not isinstance(left, (str, int)) or not isinstance(right, (str, int)):
            raise PreflightError(f"pairwise NLI case {case_number} has invalid ids")
        left, right = str(left), str(right)
        if left == right or left not in expected_ids or right not in expected_ids:
            raise PreflightError(f"pairwise NLI case {case_number} references an unknown/self id")
        key = tuple(sorted((left, right)))
        if key in pair_map:
            raise PreflightError(f"duplicate pair in case {case_number}: {key}")
        pair_map[key] = PairScore(
            key[0], key[1],
            _finite_probability(item.get("contradiction", item.get("contradiction_score")), "contradiction"),
            _finite_probability(item.get("entailment", item.get("entailment_score", 0.0)), "entailment"),
            _finite_probability(item.get("neutral", item.get("neutral_score", 0.0)), "neutral"),
        )
    if len(pair_map) != expected_pair_count:
        raise PreflightError(f"pairwise NLI case {case_number} does not cover every unordered pair")
    return pair_map, {"source": "pairwise_nli", "valid": True}


def _selected_ids(raw_case: Dict[str, Any], selector_id: str) -> List[str]:
    selectors = raw_case.get("selectors")
    if not isinstance(selectors, list):
        return []
    for selector in selectors:
        if not isinstance(selector, dict) or selector.get("selector_id") != selector_id:
            continue
        selected = selector.get("selected_top_5")
        if not isinstance(selected, list):
            return []
        ids = []
        for item in selected:
            if isinstance(item, dict) and isinstance(item.get("id"), (str, int)):
                ids.append(str(item["id"]))
        return ids
    return []


def _pair_conflict(ids: Sequence[str], pair_map: Dict[Tuple[str, str], PairScore]) -> Tuple[float, int]:
    scores = []
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            pair = pair_map.get(tuple(sorted((left, right))))
            if pair is not None:
                scores.append(pair.contradiction)
    return (sum(scores) / len(scores) if scores else 0.0, sum(score >= CONFLICT_THRESHOLD for score in scores))


def _conflict_burdens(candidates: Sequence[CandidateView], pair_map: Dict[Tuple[str, str], PairScore]) -> Dict[str, float]:
    burdens = {}
    for candidate in candidates:
        scores = []
        for other in candidates:
            if other.passage_id == candidate.passage_id:
                continue
            pair = pair_map.get(tuple(sorted((candidate.passage_id, other.passage_id))))
            if pair is not None:
                scores.append(pair.contradiction)
        burdens[candidate.passage_id] = sum(scores) / len(scores) if scores else 0.0
    return burdens


def _probe_selection(candidates: Sequence[CandidateView], burdens: Dict[str, float]) -> List[str]:
    as_values = [candidate.answer_scorer_score for candidate in candidates]
    low, high = min(as_values), max(as_values)
    span = high - low
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -((candidate.answer_scorer_score - low) / span if span else 0.0)
            + CONFLICT_PENALTY * burdens.get(candidate.passage_id, 0.0),
            candidate.retrieved_rank,
            candidate.passage_id,
        ),
    )
    return [candidate.passage_id for candidate in ordered[:5]]


def _labels_from_raw_case(raw_case: Dict[str, Any]) -> Tuple[Dict[str, Optional[bool]], Dict[str, Optional[bool]]]:
    labels: Dict[str, Optional[bool]] = {}
    contradictory: Dict[str, Optional[bool]] = {}
    for item in raw_case.get("top_50", []):
        if not isinstance(item, dict):
            continue
        passage_id = item.get("id", item.get("passage_id"))
        if isinstance(passage_id, (str, int)):
            labels[str(passage_id)] = _positive_label(item)
            contradictory[str(passage_id)] = _contradictory_label(item)
    return labels, contradictory


def analyze(root: Dict[str, Any], pair_root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cases, positions, qid_hashes = _validate_cases(root)
    raw_cases = root["cases"]
    if pair_root is not None and pair_root.get("schema_version") != PAIRWISE_NLI_SCHEMA:
        raise PreflightError(f"expected pairwise schema {PAIRWISE_NLI_SCHEMA}")

    all_conflict_burdens: List[float] = []
    all_answer_scores: List[float] = []
    all_contradictory: List[Optional[bool]] = []
    per_question: List[Dict[str, Any]] = []
    qore_conflict_rates: List[float] = []
    probe_conflict_rates: List[float] = []
    qore_pair_counts: List[int] = []
    probe_pair_counts: List[int] = []
    qore_positive_counts: List[int] = []
    probe_positive_counts: List[int] = []
    changed_cases = 0
    opportunity_cases = 0
    positive_label_count_total = 0
    positive_label_observed_total = 0
    contradictory_label_count_total = 0
    contradictory_label_observed_total = 0

    for case, raw_case, position, qid_hash in zip(cases, raw_cases, positions, qid_hashes):
        candidates: List[CandidateView] = case["candidates"]
        pair_map, source = _pair_scores_for_case(pair_root, case["case_number"], candidates)
        burdens = _conflict_burdens(candidates, pair_map)
        labels, contradictory = _labels_from_raw_case(raw_case)
        positive_label_count_total += sum(value is not None for value in labels.values())
        positive_label_observed_total += sum(value is True for value in labels.values())
        contradictory_label_count_total += sum(value is not None for value in contradictory.values())
        contradictory_label_observed_total += sum(value is True for value in contradictory.values())
        qore_ids = _selected_ids(raw_case, "qore_as")
        if len(qore_ids) != 5:
            qore_ids = [candidate.passage_id for candidate in sorted(candidates, key=lambda item: (item.retrieved_rank, item.passage_id))[:5]]
        probe_ids = _probe_selection(candidates, burdens)
        qore_conflict, qore_pairs = _pair_conflict(qore_ids, pair_map)
        probe_conflict, probe_pairs = _pair_conflict(probe_ids, pair_map)
        qore_positive = sum(labels.get(passage_id) is True for passage_id in qore_ids)
        probe_positive = sum(labels.get(passage_id) is True for passage_id in probe_ids)
        changed = set(qore_ids) != set(probe_ids)
        changed_cases += int(changed)
        opportunity_cases += int(qore_conflict > 0.0 or any(value >= CONFLICT_THRESHOLD for value in burdens.values()))
        qore_conflict_rates.append(qore_conflict)
        probe_conflict_rates.append(probe_conflict)
        qore_pair_counts.append(qore_pairs)
        probe_pair_counts.append(probe_pairs)
        qore_positive_counts.append(qore_positive)
        probe_positive_counts.append(probe_positive)
        for candidate in candidates:
            all_conflict_burdens.append(burdens[candidate.passage_id])
            all_answer_scores.append(candidate.answer_scorer_score)
            all_contradictory.append(contradictory.get(candidate.passage_id))
        per_question.append({
            "case_number": case["case_number"],
            "dataset_position": position,
            "question_id_sha256": qid_hash,
            "score_source": source["source"],
            "n_pairs": len(pair_map),
            "conflict_edge_count": sum(pair.contradiction >= CONFLICT_THRESHOLD for pair in pair_map.values()),
            "conflict_edge_rate": sum(pair.contradiction >= CONFLICT_THRESHOLD for pair in pair_map.values()) / len(pair_map),
            "conflict_score_variance": _variance([pair.contradiction for pair in pair_map.values()]),
            "qore_selected_conflict_pair_count": qore_pairs,
            "probe_selected_conflict_pair_count": probe_pairs,
            "qore_selected_conflict_score": qore_conflict,
            "probe_selected_conflict_score": probe_conflict,
            "probe_changed_set": changed,
            "qore_positive_consensus_count": qore_positive,
            "probe_positive_consensus_count": probe_positive,
            "positive_label_count": sum(value is not None for value in labels.values()),
            "contradictory_label_count": sum(value is not None for value in contradictory.values()),
        })

    conflict_auc = _rank_auc(all_conflict_burdens, all_contradictory)
    answer_auc = _rank_auc(all_answer_scores, all_contradictory)
    auc_gain = conflict_auc - answer_auc if conflict_auc is not None and answer_auc is not None else None
    qore_mean_conflict = sum(qore_conflict_rates) / len(qore_conflict_rates)
    probe_mean_conflict = sum(probe_conflict_rates) / len(probe_conflict_rates)
    reduction = (qore_mean_conflict - probe_mean_conflict) / qore_mean_conflict if qore_mean_conflict > 0 else None
    qore_mean_positive = sum(qore_positive_counts) / len(qore_positive_counts)
    probe_mean_positive = sum(probe_positive_counts) / len(probe_positive_counts)
    retention_delta = probe_mean_positive - qore_mean_positive
    pair_source = "pairwise_nli" if pair_root is not None else "lexical_proxy"

    gates = {
        "pairwise_nli_gate": {
            "pass": pair_root is not None,
            "required": True,
            "score_source": pair_source,
            "reason": "lexical proxy is diagnostic-only and cannot establish NLI conflict structure" if pair_root is None else "pairwise NLI artifact supplied",
        },
        "graph_non_degenerate_gate": {
            "pass": pair_root is not None and opportunity_cases >= 5 and any(item["conflict_score_variance"] > 1e-6 for item in per_question),
            "opportunity_case_count": opportunity_cases,
            "minimum_opportunity_cases": 5,
        },
        "independent_signal_gate": {
            "pass": bool(pair_root is not None and conflict_auc is not None and answer_auc is not None and conflict_auc >= MIN_AUC and auc_gain is not None and auc_gain >= MIN_AUC_GAIN),
            "conflict_burden_auc": conflict_auc,
            "answer_scorer_auc": answer_auc,
            "auc_gain": auc_gain,
            "minimum_auc": MIN_AUC,
            "minimum_gain": MIN_AUC_GAIN,
        },
        "matched_ablation_gate": {
            "pass": bool(pair_root is not None and positive_label_count_total == EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT and opportunity_cases >= 5 and changed_cases / EXPECTED_CASE_COUNT >= MIN_SELECTION_CHANGE_RATE and reduction is not None and reduction >= MIN_CONFLICT_REDUCTION and retention_delta >= -MAX_POSITIVE_RETENTION_DROP),
            "selection_change_rate": changed_cases / EXPECTED_CASE_COUNT,
            "conflict_reduction": reduction,
            "positive_retention_delta": retention_delta,
            "thresholds": {
                "selection_change_rate": MIN_SELECTION_CHANGE_RATE,
                "conflict_reduction": MIN_CONFLICT_REDUCTION,
                "max_positive_retention_drop": MAX_POSITIVE_RETENTION_DROP,
            },
        },
        "silver_label_availability_gate": {
            "pass": positive_label_count_total == EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT,
            "labeled_candidates": positive_label_count_total,
            "positive_candidates": positive_label_observed_total,
            "required_candidates": EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT,
            "reason": "all candidate-level silver labels are required for matched positive-retention ablation",
        },
        "contradictory_label_availability_gate": {
            "pass": (
                contradictory_label_count_total == EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT
                and contradictory_label_observed_total > 0
                and contradictory_label_observed_total < contradictory_label_count_total
            ),
            "labeled_candidates": contradictory_label_count_total,
            "contradictory_candidates": contradictory_label_observed_total,
            "required_candidates": EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT,
            "reason": "independent conflict AUC requires complete candidate-level contradictory/non-contradictory labels",
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
    gates["overall"] = {"pass": all(bool(value.get("pass")) for name, value in gates.items() if name != "overall")}
    metrics = {
        "case_count": EXPECTED_CASE_COUNT,
        "top50_count_per_case": EXPECTED_TOP50_COUNT,
        "pair_count_total": sum(item["n_pairs"] for item in per_question),
        "score_source": pair_source,
        "silver_labeled_candidate_count": positive_label_count_total,
        "silver_positive_candidate_count": positive_label_observed_total,
        "silver_label_candidate_rate": positive_label_count_total / (EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT),
        "contradictory_labeled_candidate_count": contradictory_label_count_total,
        "contradictory_candidate_count": contradictory_label_observed_total,
        "contradictory_label_candidate_rate": contradictory_label_count_total / (EXPECTED_CASE_COUNT * EXPECTED_TOP50_COUNT),
        "conflict_edge_rate_mean": sum(item["conflict_edge_rate"] for item in per_question) / EXPECTED_CASE_COUNT,
        "conflict_edge_rate_min": min(item["conflict_edge_rate"] for item in per_question),
        "conflict_edge_rate_max": max(item["conflict_edge_rate"] for item in per_question),
        "conflict_burden_auc": conflict_auc,
        "answer_scorer_auc_against_contradictory": answer_auc,
        "conflict_auc_gain_vs_answer_scorer": auc_gain,
        "qore_mean_conflict_score": qore_mean_conflict,
        "probe_mean_conflict_score": probe_mean_conflict,
        "probe_conflict_reduction": reduction,
        "qore_mean_positive_consensus_count": qore_mean_positive,
        "probe_mean_positive_consensus_count": probe_mean_positive,
        "probe_positive_retention_delta": retention_delta,
        "probe_selection_change_rate": changed_cases / EXPECTED_CASE_COUNT,
        "qore_conflict_pair_count_total": sum(qore_pair_counts),
        "probe_conflict_pair_count_total": sum(probe_pair_counts),
    }
    compact_core = {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "gates": gates,
        "question_id_sha256": qid_hashes,
        "dataset_positions": positions,
        "per_question": per_question,
    }
    return {
        **compact_core,
        "replay_digest": hashlib.sha256(_stable_json(compact_core).encode("utf-8")).hexdigest(),
        "selection_fields": ["question", "top_50[].id", "top_50[].text", "top_50[].retrieved_rank", "top_50[].retrieval_score", "top_50[].answer_scorer_score"],
        "offline_evaluation_fields": ["top_50[].evidence.consensus_label", "top_50[].evidence.positive_consensus", "selectors[].selected_top_5"],
        "forbidden_selection_fields": ["gold_answers", "top_50[].evidence", "selectors", "prediction", "generation_panel", "metrics"],
    }


def run_preflight(case_study_path: str | Path, pairwise_nli_path: Optional[str | Path] = None) -> Dict[str, Any]:
    case_path = Path(case_study_path)
    raw = case_path.read_bytes()
    try:
        root = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse UTF-8 case-study JSON: {case_path}") from exc
    pair_path = Path(pairwise_nli_path) if pairwise_nli_path is not None else None
    pair_root, pair_check = _load_pairwise_nli(pair_path)
    result = analyze(root, pair_root)
    result["input_sha256"] = _sha256_bytes(raw)
    result["pairwise_nli_check"] = pair_check
    result["replay_digest"] = hashlib.sha256(_stable_json({key: result[key] for key in ("schema_version", "metrics", "gates", "question_id_sha256", "dataset_positions", "per_question")}).encode("utf-8")).hexdigest()
    return result


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def write_artifacts(result: Dict[str, Any], output_dir: str | Path, case_study_path: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "replay_digest": result["replay_digest"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "pairwise_nli_check": result["pairwise_nli_check"],
    }
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_study_path_name": Path(case_study_path).name,
        "case_study_sha256": result.get("input_sha256"),
        "pairwise_nli_artifact_sha256": result.get("pairwise_nli_check", {}).get("sha256"),
        "pairwise_nli_model_id": result.get("pairwise_nli_check", {}).get("model_id"),
        "pairwise_nli_revision": result.get("pairwise_nli_check", {}).get("revision"),
        "code_revision": _git_revision(),
        "python_version": platform.python_version(),
        "gpu_used": False,
        "model_used": False,
        "selector_executed": False,
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
