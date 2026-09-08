"""Fixed-50, no-GPU preflight for the support-graph selector.

Selection is completed from a sanitized online view before any silver panel
labels are read.  The labels only score the frozen selection afterward.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .qdes_preflight import (
    EXPECTED_CASE_COUNT,
    EXPECTED_TOP50_COUNT,
    PreflightError,
    _sha256_bytes,
    _stable_json,
    _validate_cases,
)
from .support_graph_selector import (
    DEFAULT_ANCHOR_COUNT,
    DEFAULT_K,
    SELECTOR_ID,
    SELECTOR_VERSION,
    SelectionInputError,
    select,
)


SCHEMA_VERSION = "support_graph_no_gpu_preflight.v1"
BASELINE_SELECTOR_IDS = (
    "qore_as",
    "topk_as",
    "mmr_as",
    "submodular_as",
    "spectral_dpp_as",
)


def _git_revision() -> Optional[str]:
    override = os.environ.get("SUPPORT_GRAPH_CODE_REVISION")
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


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_online_view(candidate: Any) -> dict[str, Any]:
    return {
        "id": candidate.passage_id,
        "text": candidate.text,
        "retrieved_rank": candidate.retrieved_rank,
        "retrieval_score": candidate.retrieval_score,
        "answer_scorer_score": candidate.answer_scorer_score,
    }


def _silver_label(candidate: Mapping[str, Any], field: str) -> Optional[bool]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, dict):
        return None
    value = evidence.get(field)
    if isinstance(value, bool):
        return value
    consensus = evidence.get("consensus_label")
    if field == "positive_consensus" and isinstance(consensus, str):
        return consensus in {"direct", "partial"}
    if field == "direct_consensus" and isinstance(consensus, str):
        return consensus == "direct"
    models = evidence.get("models")
    if not isinstance(models, dict):
        return None
    labels = [
        judgment.get("label")
        for judgment in models.values()
        if isinstance(judgment, dict) and isinstance(judgment.get("label"), str)
    ]
    if not labels:
        return None
    if field == "positive_consensus":
        return sum(label in {"direct", "partial"} for label in labels) >= 2
    if field == "direct_consensus":
        return sum(label == "direct" for label in labels) >= 2
    return None


def _labels(raw_case: Mapping[str, Any]) -> tuple[dict[str, Optional[bool]], dict[str, Optional[bool]]]:
    broad: dict[str, Optional[bool]] = {}
    direct: dict[str, Optional[bool]] = {}
    for item in raw_case.get("top_50", []):
        if not isinstance(item, dict):
            continue
        passage_id = item.get("id", item.get("passage_id"))
        if isinstance(passage_id, (str, int)):
            key = str(passage_id)
            broad[key] = _silver_label(item, "positive_consensus")
            direct[key] = _silver_label(item, "direct_consensus")
    return broad, direct


def _raw_selector_ids(raw_case: Mapping[str, Any], selector_id: str) -> list[str]:
    selectors = raw_case.get("selectors")
    if not isinstance(selectors, list):
        return []
    for selector in selectors:
        if not isinstance(selector, dict) or selector.get("selector_id") != selector_id:
            continue
        selected = selector.get("selected_top_5")
        if not isinstance(selected, list):
            return []
        values = [item.get("id") for item in selected if isinstance(item, dict)]
        if len(values) != DEFAULT_K or not all(isinstance(value, (str, int)) for value in values):
            return []
        return [str(value) for value in values]
    return []


def _topk_indices(candidates: Sequence[Any]) -> tuple[int, ...]:
    return tuple(sorted(
        range(len(candidates)),
        key=lambda index: (
            -candidates[index].answer_scorer_score,
            candidates[index].retrieved_rank,
            candidates[index].passage_id,
        ),
    )[:DEFAULT_K])


def _indices_from_ids(candidates: Sequence[Any], ids: Sequence[str]) -> tuple[int, ...]:
    by_id = {candidate.passage_id: index for index, candidate in enumerate(candidates)}
    indices = tuple(by_id[value] for value in ids if value in by_id)
    return indices if len(indices) == DEFAULT_K else _topk_indices(candidates)


def _selection_counts(
    indices: Sequence[int],
    candidates: Sequence[Any],
    broad: Mapping[str, Optional[bool]],
    direct: Mapping[str, Optional[bool]],
) -> dict[str, int]:
    ids = [candidates[index].passage_id for index in indices]
    broad_values = [broad.get(value) for value in ids]
    direct_values = [direct.get(value) for value in ids]
    return {
        "broad_positive_count": sum(value is True for value in broad_values),
        "direct_positive_count": sum(value is True for value in direct_values),
        "broad_case_hit": int(any(value is True for value in broad_values)),
        "direct_case_hit": int(any(value is True for value in direct_values)),
    }


def _summary(per_question: Sequence[dict[str, Any]], key: str, *, oracle_broad: int,
             oracle_direct: int, broad_cases: int, direct_cases: int) -> dict[str, Any]:
    broad_total = sum(row[f"{key}_broad_positive_count"] for row in per_question)
    direct_total = sum(row[f"{key}_direct_positive_count"] for row in per_question)
    broad_hits = sum(row[f"{key}_broad_case_hit"] for row in per_question)
    direct_hits = sum(row[f"{key}_direct_case_hit"] for row in per_question)
    available_broad = sum(row["top50_broad_positive_count"] > 0 for row in per_question)
    available_direct = sum(row["top50_direct_positive_count"] > 0 for row in per_question)
    broad_misses = sum(
        row["top50_broad_positive_count"] > 0 and row[f"{key}_broad_positive_count"] == 0
        for row in per_question
    )
    direct_misses = sum(
        row["top50_direct_positive_count"] > 0 and row[f"{key}_direct_positive_count"] == 0
        for row in per_question
    )
    return {
        "selected_broad_positive_count": broad_total,
        "selected_direct_positive_count": direct_total,
        "selected_broad_case_hits": broad_hits,
        "selected_direct_case_hits": direct_hits,
        "available_broad_case_count": available_broad,
        "available_direct_case_count": available_direct,
        "broad_selector_miss_count": broad_misses,
        "direct_selector_miss_count": direct_misses,
        "broad_positive_regret": oracle_broad - broad_total,
        "direct_positive_regret": oracle_direct - direct_total,
        "broad_case_regret": broad_cases - broad_hits,
        "direct_case_regret": direct_cases - direct_hits,
        "case_count": len(per_question),
        "selection_count_per_case": DEFAULT_K,
    }


def analyze(root: dict[str, Any], *, input_sha256: Optional[str] = None,
            config_sha256: Optional[str] = None) -> dict[str, Any]:
    cases, positions, question_hashes = _validate_cases(root)
    raw_cases = root["cases"]

    # Selection pass: no silver labels or stored selector outputs are visible.
    selection_records: list[dict[str, Any]] = []
    for case, position, question_hash in zip(cases, positions, question_hashes):
        online = [_candidate_online_view(candidate) for candidate in case["candidates"]]
        try:
            selection = select(case["question"], online)
        except SelectionInputError as exc:
            raise PreflightError(f"case {case['case_number']} selector input rejected: {exc}") from exc
        candidates = case["candidates"]
        selection_records.append({
            "case_number": case["case_number"],
            "dataset_position": position,
            "question_id_sha256": question_hash,
            "selected_indices": list(selection.selected_indices),
            "anchor_indices": list(selection.anchor_indices),
            "support_indices": list(selection.support_indices),
            "support_scores": [round(value, 8) for value in selection.support_scores],
            "fallback": selection.fallback,
            "fallback_reason": selection.fallback_reason,
            "selected_ranks": [candidates[index].retrieved_rank for index in selection.selected_indices],
            "selected_id_sha256": [_hash_text(candidates[index].passage_id) for index in selection.selected_indices],
        })

    # Evaluation pass: panel labels and stored baseline selections are diagnostic only.
    top50_broad_total = top50_direct_total = oracle_broad = oracle_direct = 0
    top50_broad_cases = top50_direct_cases = 0
    per_question: list[dict[str, Any]] = []
    for raw_case, case, record in zip(raw_cases, cases, selection_records):
        broad, direct = _labels(raw_case)
        candidates = case["candidates"]
        top50_broad = sum(value is True for value in broad.values())
        top50_direct = sum(value is True for value in direct.values())
        top50_broad_total += top50_broad
        top50_direct_total += top50_direct
        oracle_broad += min(DEFAULT_K, top50_broad)
        oracle_direct += min(DEFAULT_K, top50_direct)
        top50_broad_cases += int(top50_broad > 0)
        top50_direct_cases += int(top50_direct > 0)
        selected_counts = _selection_counts(record["selected_indices"], candidates, broad, direct)
        row = {
            "case_number": record["case_number"],
            "dataset_position": record["dataset_position"],
            "question_id_sha256": record["question_id_sha256"],
            "selected_id_sha256": record["selected_id_sha256"],
            "selected_ranks": record["selected_ranks"],
            "anchor_count": len(record["anchor_indices"]),
            "support_count": len(record["support_indices"]),
            "support_scores": record["support_scores"],
            "fallback": record["fallback"],
            "fallback_reason": record["fallback_reason"],
            "top50_broad_positive_count": top50_broad,
            "top50_direct_positive_count": top50_direct,
            "support_graph_broad_positive_count": selected_counts["broad_positive_count"],
            "support_graph_direct_positive_count": selected_counts["direct_positive_count"],
            "support_graph_broad_case_hit": selected_counts["broad_case_hit"],
            "support_graph_direct_case_hit": selected_counts["direct_case_hit"],
        }
        for selector_id in BASELINE_SELECTOR_IDS:
            ids = _raw_selector_ids(raw_case, selector_id)
            indices = _indices_from_ids(candidates, ids)
            counts = _selection_counts(indices, candidates, broad, direct)
            row[f"{selector_id}_broad_positive_count"] = counts["broad_positive_count"]
            row[f"{selector_id}_direct_positive_count"] = counts["direct_positive_count"]
            row[f"{selector_id}_broad_case_hit"] = counts["broad_case_hit"]
            row[f"{selector_id}_direct_case_hit"] = counts["direct_case_hit"]
        per_question.append(row)

    summaries = {
        "support_graph": _summary(per_question, "support_graph", oracle_broad=oracle_broad,
                                   oracle_direct=oracle_direct, broad_cases=top50_broad_cases,
                                   direct_cases=top50_direct_cases),
    }
    for selector_id in BASELINE_SELECTOR_IDS:
        summaries[selector_id] = _summary(per_question, selector_id, oracle_broad=oracle_broad,
                                          oracle_direct=oracle_direct, broad_cases=top50_broad_cases,
                                          direct_cases=top50_direct_cases)

    candidate_summary = summaries["support_graph"]
    topk = summaries["topk_as"]
    replay_core = {
        "selector_id": SELECTOR_ID,
        "selector_version": SELECTOR_VERSION,
        "metrics": {
            "case_count": len(cases),
            "top50_count_per_case": EXPECTED_TOP50_COUNT,
            "selector_k": DEFAULT_K,
            "anchor_count": DEFAULT_ANCHOR_COUNT,
            "support_passage_weight": 0.70,
            "support_question_weight": 0.30,
            "selectors": summaries,
            "oracle_cap": {
                "broad_positive_passage_count": oracle_broad,
                "direct_positive_passage_count": oracle_direct,
                "broad_positive_case_count": top50_broad_cases,
                "direct_positive_case_count": top50_direct_cases,
            },
            "top50_positive_candidate_count": {
                "broad_positive_passage_count": top50_broad_total,
                "direct_positive_passage_count": top50_direct_total,
            },
            "fallback_case_count": sum(item["fallback"] for item in selection_records),
        },
        "question_id_sha256": question_hashes,
        "dataset_positions": positions,
        "per_question": per_question,
    }
    gates: dict[str, Any] = {
        "schema_gate": {
            "pass": len(cases) == EXPECTED_CASE_COUNT and all(
                len(case["candidates"]) == EXPECTED_TOP50_COUNT for case in cases
            ),
            "cases": len(cases),
            "top50_per_case": EXPECTED_TOP50_COUNT,
        },
        "scope_gate": {
            "pass": True,
            "gpu_used": False,
            "model_used": False,
            "retrieval_run": False,
            "generator_run": False,
            "selector_executed": True,
        },
        "leakage_gate": {
            "pass": True,
            "selection_feedback": False,
            "gold_feedback": False,
            "panel_feedback": False,
            "generation_feedback": False,
            "evaluator_feedback": False,
            "selection_fields": [
                "question", "top_50[].id", "top_50[].text", "top_50[].retrieved_rank",
                "top_50[].retrieval_score", "top_50[].answer_scorer_score",
            ],
            "offline_evaluation_fields": ["top_50[].evidence", "selectors[].selected_top_5"],
        },
        "broad_improvement_gate": {
            "pass": candidate_summary["selected_broad_positive_count"] > topk["selected_broad_positive_count"],
            "support_graph": candidate_summary["selected_broad_positive_count"],
            "topk_baseline": topk["selected_broad_positive_count"],
            "required": "strictly greater than Top-k + Answer Scorer",
        },
        "direct_retention_gate": {
            "pass": candidate_summary["selected_direct_positive_count"] >= topk["selected_direct_positive_count"],
            "support_graph": candidate_summary["selected_direct_positive_count"],
            "topk_baseline": topk["selected_direct_positive_count"],
            "required": "not below Top-k + Answer Scorer",
        },
        "selector_miss_gate": {
            "pass": candidate_summary["broad_selector_miss_count"] <= 5,
            "support_graph": candidate_summary["broad_selector_miss_count"],
            "available_broad_case_count": candidate_summary["available_broad_case_count"],
            "threshold": 5,
        },
        "deterministic_replay_gate": {"pass": True, "checked_by_runner": True},
    }
    gates["overall"] = {
        "pass": all(value.get("pass", False) for name, value in gates.items() if name != "overall")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "selector_id": SELECTOR_ID,
        "selector_version": SELECTOR_VERSION,
        "metrics": replay_core["metrics"],
        "gates": gates,
        "question_id_sha256": question_hashes,
        "dataset_positions": positions,
        "per_question": per_question,
        "input_sha256": input_sha256,
        "config_sha256": config_sha256,
        "replay_digest": hashlib.sha256(_stable_json(replay_core).encode("utf-8")).hexdigest(),
        "selection_fields": gates["leakage_gate"]["selection_fields"],
        "offline_evaluation_fields": gates["leakage_gate"]["offline_evaluation_fields"],
        "forbidden_selection_fields": [
            "gold_answers", "top_50[].evidence", "selectors", "prediction", "generation_panel", "metrics",
        ],
    }


def run_preflight(case_study_path: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    case_path = Path(case_study_path)
    raw = case_path.read_bytes()
    try:
        root = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse UTF-8 case-study JSON: {case_path}") from exc
    if not isinstance(root, dict):
        raise PreflightError("case-study root must be an object")
    config_sha256 = _sha256_bytes(Path(config_path).read_bytes()) if config_path is not None else None
    return analyze(root, input_sha256=_sha256_bytes(raw), config_sha256=config_sha256)


def write_artifacts(result: dict[str, Any], output_dir: str | Path, case_study_path: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "selector_id": SELECTOR_ID,
        "replay_digest": result["replay_digest"],
        "metrics": result["metrics"],
        "gates": result["gates"],
    }, ensure_ascii=True, indent=2), encoding="utf-8")
    (directory / "run_metadata.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "case_study_path_name": Path(case_study_path).name,
        "case_study_sha256": result.get("input_sha256"),
        "config_sha256": result.get("config_sha256"),
        "code_revision": _git_revision(),
        "python_version": platform.python_version(),
        "gpu_used": False,
        "model_used": False,
        "retrieval_run": False,
        "generator_run": False,
        "selector_executed": True,
    }, ensure_ascii=True, indent=2), encoding="utf-8")


__all__ = ["SCHEMA_VERSION", "analyze", "run_preflight", "write_artifacts"]
