#!/usr/bin/env python3
"""Measure the frozen Generator ceiling of a three-model silver-oracle Top-5.

The run reuses two existing 50-question case studies and their completed
three-model evidence panels.  It does not retrieve, score passages, or download
Wiki-DPR.  QORE, Top-k, and silver-oracle memberships are all reordered by the
original retrieval rank before the same frozen Generator is called.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import yaml


_SCRIPT_PATH = Path(__file__).resolve()
for _candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_candidate / "configs").is_dir() and (_candidate / "applications").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break


class OracleRunError(RuntimeError):
    """Raised when the preregistered diagnostic contract cannot be satisfied."""


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_SOURCE_IDS = ("nq_1950_1999", "nq_3150_3199")
EXPECTED_ARMS = ("qore_common_order", "topk_common_order", "silver_oracle_common_order")
EXPECTED_PLUGINS = (
    "registered_case_panel_loader",
    "silver_oracle_membership",
    "common_context_order",
    "frozen_generator_observer",
    "oracle_result_renderer",
)
FORBIDDEN_COMPACT_FIELDS = frozenset(
    {
        "question",
        "passages",
        "gold_answers",
        "prediction",
        "raw_prompt",
        "prompt",
        "text",
        "token_ids",
        "selected_ids",
        "retrieved_ids",
    }
)
MAX_COMPACT_BYTES = 1_048_576


def _root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise OracleRunError("cannot locate project root")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleRunError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleRunError(f"JSON root must be an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise OracleRunError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleRunError("config root must be a mapping")
    return value


def _source_specs(phase: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    specs = phase.get("input_sources")
    if not isinstance(specs, list) or len(specs) != 2:
        raise OracleRunError("exactly two registered 50-question sources are required")
    normalized = tuple(dict(item) for item in specs if isinstance(item, Mapping))
    if len(normalized) != 2 or tuple(item.get("id") for item in normalized) != EXPECTED_SOURCE_IDS:
        raise OracleRunError("input source ids or order changed")
    for spec in normalized:
        if int(spec.get("case_count", -1)) != 50:
            raise OracleRunError("each registered source must contain 50 cases")
        for field in ("case_study_sha256", "evidence_panel_sha256"):
            value = str(spec.get(field, ""))
            if len(value) != 64:
                raise OracleRunError(f"{spec['id']} has invalid {field}")
    return normalized


def validate_contract(config_path: Path, plan_path: Path) -> dict[str, Any]:
    """Validate the preregistration without loading data, model, or Wiki-DPR."""

    phase = _load_yaml(config_path).get("phase")
    if not isinstance(phase, Mapping):
        raise OracleRunError("phase section is missing")
    if phase.get("name") != "silver_oracle_top5_100" or int(phase.get("schema_version", -1)) != 1:
        raise OracleRunError("unexpected experiment identity")
    if phase.get("diagnostic_only") is not True or phase.get("evidence_tier") != "L0_diagnostic":
        raise OracleRunError("silver oracle must remain L0 diagnostic")
    if phase.get("selection_mutation") is not False or phase.get("retrieval_or_scoring_calls") != 0:
        raise OracleRunError("production selection and retrieval/scoring must remain frozen")
    specs = _source_specs(phase)
    vote_bundle = phase.get("compact_evidence_bundle")
    if vote_bundle != {
        "path": "configs/experiments/silver_oracle_top5_100_votes.json",
        "sha256": "db364b18294437950e98ce0f6b526e1313d52131a31f9e9b69a199f7d9aff3bb",
        "contains_raw_content": False,
        "alignment": "registered_case_number_and_original_top50_retrieval_rank",
    }:
        raise OracleRunError("compact evidence bundle identity changed")
    policy = phase.get("oracle_policy")
    if policy != {
        "k": 5,
        "judge_count": 3,
        "consensus_votes": 2,
        "priority": ["direct_consensus", "positive_consensus", "answer_scorer_fill"],
        "positive_labels": ["direct", "partial"],
        "tie_break": ["direct_votes", "positive_votes", "mean_confidence", "answer_scorer_score", "retrieved_rank", "candidate_id"],
        "target_conditioned": True,
        "deployable": False,
    }:
        raise OracleRunError("oracle policy changed")
    comparison = phase.get("comparison")
    if not isinstance(comparison, Mapping) or tuple(comparison.get("arms", ())) != EXPECTED_ARMS:
        raise OracleRunError("comparison arm allowlist changed")
    if comparison.get("context_order") != "ascending_original_retrieval_rank" or comparison.get("generator_calls") != 300:
        raise OracleRunError("common ordering or Generator call count changed")
    generator = phase.get("generator")
    if not isinstance(generator, Mapping) or generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION:
        raise OracleRunError("Generator identity changed")
    if generator.get("decoding") != "greedy" or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("use_chat_template") is not True:
        raise OracleRunError("Generator decoding contract changed")
    integrity = phase.get("input_integrity")
    expected_integrity = {
        "case_count": 100,
        "candidate_count": 5000,
        "direct_consensus_candidates": 289,
        "positive_consensus_candidates": 603,
        "cases_with_direct_consensus": 77,
        "cases_with_positive_consensus": 87,
    }
    if integrity != expected_integrity:
        raise OracleRunError("input integrity counts changed")
    expected_memberships = {
        "qore_common_order": {"direct": 119, "broad": 177},
        "topk_common_order": {"direct": 138, "broad": 203},
        "silver_oracle_common_order": {"direct": 222, "broad": 307},
    }
    if phase.get("membership_integrity") != expected_memberships:
        raise OracleRunError("membership integrity counts changed")
    outputs = phase.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("root") != "exchange/five_ideas/silver_oracle_top5_100":
        raise OracleRunError("output root changed")
    if outputs.get("complete_files") != ["case_study.json", "case_study.md"]:
        raise OracleRunError("complete output contract changed")
    if outputs.get("compact_files") != ["result.json", "run_metadata.json", "upload_manifest.json"]:
        raise OracleRunError("compact output contract changed")
    if outputs.get("complete_files_exchange_only") is not True or int(outputs.get("github_max_bytes", -1)) != MAX_COMPACT_BYTES:
        raise OracleRunError("publication policy changed")
    if set(outputs.get("forbidden_compact_fields", ())) != FORBIDDEN_COMPACT_FIELDS:
        raise OracleRunError("compact privacy field set changed")
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v1" or plan.get("project") != "Q-DUET-VLM" or plan.get("authorization") != "implemented":
        raise OracleRunError("plugin plan identity is invalid")
    discovery = plan.get("discovery", {})
    composition = plan.get("composition", {})
    if discovery.get("mode") != "explicit_allowlist" or tuple(discovery.get("allowlist", ())) != EXPECTED_PLUGINS:
        raise OracleRunError("plugin discovery allowlist changed")
    if composition.get("mode") != "sequential" or tuple(composition.get("order", ())) != EXPECTED_PLUGINS:
        raise OracleRunError("plugin composition order changed")
    if plan.get("reproducibility", {}).get("config_path") != "configs/experiments/silver_oracle_top5_100.yaml":
        raise OracleRunError("plan/config link changed")
    return {
        "status": "valid",
        "phase": phase["name"],
        "evidence_tier": phase["evidence_tier"],
        "source_ids": [spec["id"] for spec in specs],
        "case_count": 100,
        "arms": list(EXPECTED_ARMS),
        "generator_calls": 300,
        "model_loaded": False,
        "wiki_dpr_started": False,
    }


def _resolve_path(root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            output.append(path)
            seen.add(key)
    return output


def _glob_candidates(root: Path, patterns: Sequence[str]) -> list[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in root.glob(str(pattern)) if path.is_file())
    return _unique_paths(sorted(candidates))


def _find_hash_match(
    root: Path,
    *,
    expected_sha256: str,
    explicit: Path | None,
    exact_candidates: Sequence[str],
    glob_patterns: Sequence[str],
    label: str,
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.resolve())
    candidates.extend(_resolve_path(root, str(value)) for value in exact_candidates)
    candidates.extend(_glob_candidates(root, [str(value) for value in glob_patterns]))
    checked: list[str] = []
    for path in _unique_paths(candidates):
        if not path.is_file():
            continue
        checked.append(path.name)
        if _sha256(path) == expected_sha256:
            return path
    suffix = f"; checked {', '.join(checked[:8])}" if checked else ""
    raise OracleRunError(
        f"cannot locate registered {label} with sha256 {expected_sha256}{suffix}"
    )


def _detail_matches(payload: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    source = payload.get("source")
    return (
        isinstance(source, Mapping)
        and source.get("case_study_sha256") == spec["case_study_sha256"]
        and source.get("panel_sha256") == spec["evidence_panel_sha256"]
        and int(source.get("case_count", -1)) == 50
    )


def _find_detail_source(
    root: Path, spec: Mapping[str, Any], explicit: Path | None
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.resolve())
    candidates.extend(
        _resolve_path(root, str(value))
        for value in spec.get("merged_detail_candidates", [])
    )
    for path in _unique_paths(candidates):
        if not path.is_file():
            continue
        payload = _load_json(path)
        if _detail_matches(payload, spec):
            return path, payload
        if explicit is not None and path == explicit.resolve():
            raise OracleRunError(f"explicit detail source has wrong registered identity: {path}")
    return None


def _normalized_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    from applications.rag.silver_oracle_selector import evidence_counts

    stats = evidence_counts({"evidence": evidence})
    models = evidence.get("models")
    normalized: dict[str, Any] = {
        "model_labels": list(stats["model_labels"]),
        "mean_confidence": round(float(stats["mean_confidence"]), 6),
        "direct_votes": int(stats["direct_votes"]),
        "positive_votes": int(stats["positive_votes"]),
        "direct_consensus": bool(stats["direct_consensus"]),
        "positive_consensus": bool(stats["positive_consensus"]),
    }
    if isinstance(models, Mapping):
        normalized["models"] = {
            str(model): {
                "label": str(row.get("label")),
                "confidence": float(row.get("confidence")),
            }
            for model, row in models.items()
            if isinstance(row, Mapping)
        }
    return normalized


def _panel_evidence(panel: Mapping[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    evidence_panel = panel.get("evidence_panel")
    aggregate = evidence_panel.get("aggregate") if isinstance(evidence_panel, Mapping) else None
    per_case = aggregate.get("per_case") if isinstance(aggregate, Mapping) else None
    if not isinstance(per_case, list) or len(per_case) != 50:
        raise OracleRunError("raw evidence panel must contain 50 aggregate per_case rows")
    output: dict[int, dict[str, dict[str, Any]]] = {}
    for row in per_case:
        if not isinstance(row, Mapping):
            raise OracleRunError("evidence per_case row must be a mapping")
        case_number = int(row.get("case_number", -1))
        labels = row.get("candidate_labels")
        if case_number in output or not isinstance(labels, Mapping) or len(labels) != 50:
            raise OracleRunError("evidence panel case numbering or candidate count is invalid")
        output[case_number] = {
            str(candidate_id): _normalized_evidence(summary)
            for candidate_id, summary in labels.items()
            if isinstance(summary, Mapping)
        }
        if len(output[case_number]) != 50:
            raise OracleRunError("evidence panel contains an invalid candidate summary")
    if set(output) != set(range(1, 51)):
        raise OracleRunError("evidence panel must contain case numbers 1..50")
    return output


def _load_compact_vote_bundle(
    root: Path, phase: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    specification = phase["compact_evidence_bundle"]
    path = _resolve_path(root, str(specification["path"]))
    if not path.is_file() or _sha256(path) != specification["sha256"]:
        raise OracleRunError("registered compact evidence bundle is missing or has wrong SHA-256")
    payload = _load_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "silver_oracle_rank_aligned_vote_bundle"
        or payload.get("diagnostic_only") is not True
        or payload.get("contains_raw_content") is not False
        or payload.get("vote_tuple")
        != ["direct_votes", "positive_votes", "mean_confidence"]
    ):
        raise OracleRunError("compact evidence bundle schema is invalid")
    sources = payload.get("sources")
    if not isinstance(sources, list) or [row.get("id") for row in sources if isinstance(row, Mapping)] != list(EXPECTED_SOURCE_IDS):
        raise OracleRunError("compact evidence bundle source allowlist changed")
    return {str(row["id"]): row for row in sources}, {
        "path_name": path.name,
        "sha256": specification["sha256"],
        "bytes": path.stat().st_size,
        "contains_raw_content": False,
    }


def _rank_aligned_vote_evidence(
    case_payload: Mapping[str, Any],
    source_spec: Mapping[str, Any],
    bundled_source: Mapping[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    if (
        bundled_source.get("case_study_sha256") != source_spec["case_study_sha256"]
        or bundled_source.get("evidence_panel_sha256") != source_spec["evidence_panel_sha256"]
    ):
        raise OracleRunError("compact vote bundle source provenance mismatch")
    cases = case_payload.get("cases")
    vote_cases = bundled_source.get("cases")
    if not isinstance(cases, list) or not isinstance(vote_cases, list) or len(cases) != 50 or len(vote_cases) != 50:
        raise OracleRunError("compact vote bundle must align to 50 source cases")
    output: dict[int, dict[str, dict[str, Any]]] = {}
    for case_number, (case, votes) in enumerate(zip(cases, vote_cases), start=1):
        top_50 = case.get("top_50") if isinstance(case, Mapping) else None
        if not isinstance(top_50, list) or not isinstance(votes, list) or len(top_50) != 50 or len(votes) != 50:
            raise OracleRunError("compact vote bundle rank alignment is invalid")
        by_candidate: dict[str, dict[str, Any]] = {}
        for passage, vote in zip(top_50, votes):
            if not isinstance(passage, Mapping) or not isinstance(vote, list) or len(vote) != 3:
                raise OracleRunError("compact vote tuple is invalid")
            direct_votes, positive_votes, confidence = vote
            if (
                isinstance(direct_votes, bool)
                or isinstance(positive_votes, bool)
                or not isinstance(direct_votes, int)
                or not isinstance(positive_votes, int)
                or not 0 <= direct_votes <= positive_votes <= 3
            ):
                raise OracleRunError("compact vote counts are invalid")
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise OracleRunError("compact mean confidence is invalid")
            labels = (
                ["direct"] * direct_votes
                + ["partial"] * (positive_votes - direct_votes)
                + ["irrelevant"] * (3 - positive_votes)
            )
            by_candidate[str(passage["id"])] = {
                "model_labels": labels,
                "mean_confidence": confidence,
                "direct_votes": direct_votes,
                "positive_votes": positive_votes,
                "direct_consensus": direct_votes >= 2,
                "positive_consensus": positive_votes >= 2,
            }
        output[case_number] = by_candidate
    return output


def _normalize_source_cases(
    payload: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    evidence_by_case: Mapping[int, Mapping[str, Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 50:
        raise OracleRunError(f"{spec['id']} must contain exactly 50 cases")
    start = int(spec["global_start"])
    normalized: list[dict[str, Any]] = []
    for offset, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise OracleRunError("case row must be a mapping")
        case_number = int(case.get("case_number", -1))
        if case_number != offset + 1:
            raise OracleRunError(f"{spec['id']} case numbers must be 1..50")
        global_index = start + offset
        question_id = str(case.get("question_id", ""))
        if question_id != f"nq_{global_index}":
            raise OracleRunError(
                f"{spec['id']} expected question nq_{global_index}, found {question_id}"
            )
        top_50 = case.get("top_50")
        if not isinstance(top_50, list) or len(top_50) != 50:
            raise OracleRunError(f"{question_id} must contain exactly 50 candidates")
        candidate_ids = [str(item.get("id", "")) for item in top_50 if isinstance(item, Mapping)]
        if len(candidate_ids) != 50 or any(not value for value in candidate_ids) or len(set(candidate_ids)) != 50:
            raise OracleRunError(f"{question_id} candidate ids are invalid")
        expected_ranks = list(range(1, 51))
        ranks = [int(item.get("retrieved_rank", -1)) for item in top_50]
        if ranks != expected_ranks:
            raise OracleRunError(f"{question_id} retrieved ranks must be 1..50")
        passages: list[dict[str, Any]] = []
        for item in top_50:
            passage = dict(item)
            if evidence_by_case is not None:
                evidence = evidence_by_case[case_number].get(str(item["id"]))
                if not isinstance(evidence, Mapping):
                    raise OracleRunError(f"panel is missing candidate {item['id']}")
                passage["evidence"] = dict(evidence)
            elif isinstance(item.get("evidence"), Mapping):
                passage["evidence"] = _normalized_evidence(item["evidence"])
            else:
                raise OracleRunError(f"merged detail is missing evidence for {item['id']}")
            passages.append(passage)
        selectors = case.get("selectors")
        if not isinstance(selectors, list):
            raise OracleRunError(f"{question_id} selectors are missing")
        baseline_ids: dict[str, list[str]] = {}
        for selector_id in ("qore_as", "topk_as"):
            matches = [row for row in selectors if isinstance(row, Mapping) and row.get("selector_id") == selector_id]
            if len(matches) != 1 or not isinstance(matches[0].get("selected_top_5"), list):
                raise OracleRunError(f"{question_id} is missing baseline {selector_id}")
            ids = [str(item.get("id", "")) for item in matches[0]["selected_top_5"] if isinstance(item, Mapping)]
            if len(ids) != 5 or len(set(ids)) != 5 or not set(ids).issubset(candidate_ids):
                raise OracleRunError(f"{question_id} has invalid {selector_id} membership")
            baseline_ids[selector_id] = ids
        answers = case.get("gold_answers")
        if not isinstance(answers, list) or not answers:
            raise OracleRunError(f"{question_id} has no gold answers")
        normalized.append(
            {
                "source_id": str(spec["id"]),
                "source_case_number": case_number,
                "global_index": global_index,
                "question_id": question_id,
                "question": str(case.get("question", "")),
                "gold_answers": [str(answer) for answer in answers],
                "top_50": passages,
                "baseline_memberships": baseline_ids,
            }
        )
    return normalized


def _load_registered_source(
    root: Path,
    spec: Mapping[str, Any],
    *,
    detail_override: Path | None = None,
    case_override: Path | None = None,
    panel_override: Path | None = None,
    bundled_source: Mapping[str, Any] | None = None,
    bundle_provenance: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detail = _find_detail_source(root, spec, detail_override)
    if detail is not None:
        path, payload = detail
        return _normalize_source_cases(payload, spec), {
            "source_id": spec["id"],
            "mode": "merged_detail",
            "path_name": path.name,
            "merged_sha256": _sha256(path),
            "case_study_sha256": spec["case_study_sha256"],
            "evidence_panel_sha256": spec["evidence_panel_sha256"],
        }
    case_path = _find_hash_match(
        root,
        expected_sha256=str(spec["case_study_sha256"]),
        explicit=case_override,
        exact_candidates=spec.get("case_study_candidates", []),
        glob_patterns=spec.get("case_study_globs", []),
        label=f"{spec['id']} case study",
    )
    case_payload = _load_json(case_path)
    if bundled_source is not None:
        evidence = _rank_aligned_vote_evidence(case_payload, spec, bundled_source)
        return _normalize_source_cases(
            case_payload, spec, evidence_by_case=evidence
        ), {
            "source_id": spec["id"],
            "mode": "registered_compact_vote_join",
            "case_path_name": case_path.name,
            "case_study_sha256": _sha256(case_path),
            "evidence_panel_sha256": spec["evidence_panel_sha256"],
            "compact_vote_bundle": dict(bundle_provenance or {}),
        }
    panel_path = _find_hash_match(
        root,
        expected_sha256=str(spec["evidence_panel_sha256"]),
        explicit=panel_override,
        exact_candidates=spec.get("evidence_panel_candidates", []),
        glob_patterns=spec.get("evidence_panel_globs", []),
        label=f"{spec['id']} evidence panel",
    )
    evidence = _panel_evidence(_load_json(panel_path))
    return _normalize_source_cases(
        case_payload, spec, evidence_by_case=evidence
    ), {
        "source_id": spec["id"],
        "mode": "raw_join",
        "case_path_name": case_path.name,
        "case_study_sha256": _sha256(case_path),
        "panel_path_name": panel_path.name,
        "evidence_panel_sha256": _sha256(panel_path),
    }


def _validate_integrity(
    cases: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> dict[str, int]:
    from applications.rag.silver_oracle_selector import count_consensus

    if len(cases) != int(expected["case_count"]):
        raise OracleRunError("combined source case count mismatch")
    question_ids = [str(case["question_id"]) for case in cases]
    if len(set(question_ids)) != len(question_ids):
        raise OracleRunError("combined sources contain duplicate questions")
    candidate_count = 0
    direct = 0
    broad = 0
    direct_cases = 0
    broad_cases = 0
    for case in cases:
        counts = count_consensus(case["top_50"])
        candidate_count += len(case["top_50"])
        direct += counts["direct"]
        broad += counts["broad"]
        direct_cases += counts["direct"] > 0
        broad_cases += counts["broad"] > 0
    observed = {
        "case_count": len(cases),
        "candidate_count": candidate_count,
        "direct_consensus_candidates": direct,
        "positive_consensus_candidates": broad,
        "cases_with_direct_consensus": direct_cases,
        "cases_with_positive_consensus": broad_cases,
    }
    if observed != dict(expected):
        raise OracleRunError(f"registered input integrity mismatch: {observed}")
    return observed


def load_registered_cases(
    root: Path, phase: Mapping[str, Any], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    specs = _source_specs(phase)
    bundled_sources, bundle_provenance = _load_compact_vote_bundle(root, phase)
    overrides = (
        (args.detail_source_a, args.case_source_a, args.panel_source_a),
        (args.detail_source_b, args.case_source_b, args.panel_source_b),
    )
    cases: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for spec, override in zip(specs, overrides):
        source_cases, source_provenance = _load_registered_source(
            root,
            spec,
            detail_override=override[0],
            case_override=override[1],
            panel_override=override[2],
            bundled_source=bundled_sources.get(str(spec["id"])),
            bundle_provenance=bundle_provenance,
        )
        cases.extend(source_cases)
        provenance.append(source_provenance)
    integrity = _validate_integrity(cases, phase["input_integrity"])
    return cases, provenance, integrity


def _resolve_generator(
    root: Path, specification: Mapping[str, Any], override: str | None
) -> dict[str, str]:
    candidates: list[tuple[Path, str]] = []
    if override:
        candidates.append(
            (Path(os.path.expandvars(os.path.expanduser(override))), "cli_override")
        )
    for raw in specification.get("model_path_candidates", []):
        path = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        candidates.append((path if path.is_absolute() else root / path, "project_candidate"))
    cache_roots = [Path.home() / ".cache/huggingface/hub"]
    if os.environ.get("HF_HOME"):
        cache_roots.insert(0, Path(os.environ["HF_HOME"]) / "hub")
    for cache in cache_roots:
        candidates.append(
            (cache / "models--NousResearch--Meta-Llama-3-8B-Instruct", "hf_cache")
        )
    attempted: list[str] = []
    for candidate, resolution in candidates:
        candidate = candidate.resolve()
        attempted.append(str(candidate))
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate if (candidate / "config.json").is_file() else snapshot
        if (resolved / "config.json").is_file():
            return {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_path": str(resolved),
                "config_sha256": _sha256(resolved / "config.json"),
                "resolution": resolution,
            }
    raise OracleRunError("frozen Generator not found; attempted: " + ", ".join(attempted))


def _passage_text(passage: Mapping[str, Any]) -> str:
    value = passage.get("passage")
    if value:
        return str(value)
    title = str(passage.get("title") or "")
    text = str(passage.get("text") or "")
    return f"{title}. {text}" if title else text


def build_arm_passages(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build three membership arms with one neutral retrieval-rank ordering."""

    from applications.rag.silver_oracle_selector import select

    top_50 = list(case["top_50"])
    by_id = {str(item["id"]): item for item in top_50}
    oracle = select(top_50, k=5)
    memberships = {
        "qore_common_order": list(case["baseline_memberships"]["qore_as"]),
        "topk_common_order": list(case["baseline_memberships"]["topk_as"]),
        "silver_oracle_common_order": [str(top_50[index]["id"]) for index in oracle.selected_indices],
    }
    arms: dict[str, dict[str, Any]] = {}
    for arm_id in EXPECTED_ARMS:
        ids = memberships[arm_id]
        if len(ids) != 5 or len(set(ids)) != 5 or not set(ids).issubset(by_id):
            raise OracleRunError(f"invalid membership for {arm_id}")
        passages = sorted((by_id[value] for value in ids), key=lambda item: int(item["retrieved_rank"]))
        arms[arm_id] = {
            "passages": passages,
            "context_order": "ascending_original_retrieval_rank",
            "oracle_priority_trace": [dict(item) for item in oracle.trace]
            if arm_id == "silver_oracle_common_order"
            else [],
        }
    return arms


def _validate_membership_integrity(
    cases: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    from applications.rag.silver_oracle_selector import count_consensus

    observed = {
        arm_id: {"direct": 0, "broad": 0} for arm_id in EXPECTED_ARMS
    }
    for case in cases:
        arms = build_arm_passages(case)
        for arm_id in EXPECTED_ARMS:
            counts = count_consensus(arms[arm_id]["passages"])
            observed[arm_id]["direct"] += counts["direct"]
            observed[arm_id]["broad"] += counts["broad"]
    normalized_expected = {
        str(arm_id): {
            "direct": int(counts["direct"]),
            "broad": int(counts["broad"]),
        }
        for arm_id, counts in expected.items()
    }
    if observed != normalized_expected:
        raise OracleRunError(f"registered membership integrity mismatch: {observed}")
    return observed


def _evaluate_arm(
    *,
    question: str,
    answers: Sequence[str],
    passages: Sequence[Mapping[str, Any]],
    generator: Any,
) -> tuple[str, dict[str, float], float]:
    from applications.rag.evaluation import evaluate_answer

    started = time.perf_counter()
    prediction = str(generator.generate(question, [_passage_text(item) for item in passages]))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metrics = evaluate_answer(prediction, list(answers))
    return prediction, {"em": float(metrics["em"]), "f1": float(metrics["f1"])}, elapsed_ms


def _bootstrap_ci(values: Sequence[float], *, seed: int, reps: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(reps, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _forbidden_fields(value: Any, path: str = "$root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_COMPACT_FIELDS:
                found.append(f"{path}.{key}")
            found.extend(_forbidden_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_fields(child, f"{path}[{index}]"))
    return found


def _compact_generator_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Keep reproducible model identity without publishing a collaborator path."""

    return {
        key: identity[key]
        for key in ("model_id", "revision", "config_sha256", "resolution")
        if key in identity
    }


def summarize(
    cases: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    phase: Mapping[str, Any],
) -> dict[str, Any]:
    from applications.rag.silver_oracle_selector import count_consensus

    rows_by_arm: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in EXPECTED_ARMS}
    for case in cases:
        for arm in case["arms"]:
            rows_by_arm[str(arm["arm_id"])].append(arm)
    arm_summaries: dict[str, Any] = {}
    for arm_id, rows in rows_by_arm.items():
        arm_summaries[arm_id] = {
            "case_count": len(rows),
            "mean_em": round(float(np.mean([row["metrics"]["em"] for row in rows])), 6),
            "mean_f1": round(float(np.mean([row["metrics"]["f1"] for row in rows])), 6),
            "mean_direct_consensus_in_top5": round(float(np.mean([row["evidence_counts"]["direct"] for row in rows])), 6),
            "mean_positive_consensus_in_top5": round(float(np.mean([row["evidence_counts"]["broad"] for row in rows])), 6),
            "mean_generation_ms": round(float(np.mean([row["generation_ms"] for row in rows])), 3),
        }
    oracle = rows_by_arm["silver_oracle_common_order"]
    paired: dict[str, Any] = {}
    reps = int(phase["gates"]["paired_bootstrap_repetitions"])
    seed = int(phase["gates"]["paired_bootstrap_seed"])
    for baseline_id in ("qore_common_order", "topk_common_order"):
        baseline = rows_by_arm[baseline_id]
        f1_deltas = [float(o["metrics"]["f1"]) - float(b["metrics"]["f1"]) for o, b in zip(oracle, baseline)]
        em_deltas = [float(o["metrics"]["em"]) - float(b["metrics"]["em"]) for o, b in zip(oracle, baseline)]
        positives = sorted((value for value in f1_deltas if value > 0), reverse=True)
        positive_total = sum(positives)
        paired[f"silver_oracle_minus_{baseline_id.removesuffix('_common_order')}"] = {
            "mean_em_delta": round(float(np.mean(em_deltas)), 6),
            "mean_f1_delta": round(float(np.mean(f1_deltas)), 6),
            "f1_delta_ci95": [round(value, 6) for value in _bootstrap_ci(f1_deltas, seed=seed, reps=reps)],
            "positive_f1_cases": sum(value > 0 for value in f1_deltas),
            "unchanged_f1_cases": sum(value == 0 for value in f1_deltas),
            "negative_f1_cases": sum(value < 0 for value in f1_deltas),
            "largest_three_positive_gain_share": round(sum(positives[:3]) / positive_total, 6) if positive_total > 0 else None,
        }
        seed += 1
    strata = {
        "retrieval_miss_no_positive_consensus": 0,
        "no_qore_membership_headroom": 0,
        "oracle_more_evidence_f1_up": 0,
        "oracle_more_evidence_f1_same": 0,
        "oracle_more_evidence_f1_down": 0,
    }
    for case in cases:
        top_counts = count_consensus(case["top_50"])
        arms = {str(row["arm_id"]): row for row in case["arms"]}
        qore = arms["qore_common_order"]
        oracle_row = arms["silver_oracle_common_order"]
        if top_counts["broad"] == 0:
            strata["retrieval_miss_no_positive_consensus"] += 1
        evidence_more = (
            oracle_row["evidence_counts"]["broad"] > qore["evidence_counts"]["broad"]
            or oracle_row["evidence_counts"]["direct"] > qore["evidence_counts"]["direct"]
        )
        if not evidence_more:
            strata["no_qore_membership_headroom"] += 1
        else:
            delta = float(oracle_row["metrics"]["f1"]) - float(qore["metrics"]["f1"])
            suffix = "up" if delta > 0 else "down" if delta < 0 else "same"
            strata[f"oracle_more_evidence_f1_{suffix}"] += 1
    qore_delta = paired["silver_oracle_minus_qore"]
    topk_delta = paired["silver_oracle_minus_topk"]
    thresholds = phase["gates"]
    gate_checks = {
        "oracle_minus_qore_f1_delta": qore_delta["mean_f1_delta"] >= float(thresholds["oracle_minus_qore_f1_delta_min"]),
        "oracle_minus_qore_ci_lower": qore_delta["f1_delta_ci95"][0] > float(thresholds["oracle_minus_qore_f1_ci95_lower_min"]),
        "oracle_minus_qore_em": qore_delta["mean_em_delta"] >= float(thresholds["oracle_minus_qore_em_delta_min"]),
        "oracle_minus_topk_direction": topk_delta["mean_f1_delta"] > float(thresholds["oracle_minus_topk_f1_delta_min"]),
        "distributed_gain": qore_delta["positive_f1_cases"] >= int(thresholds["positive_f1_case_count_min"]),
    }
    passed = all(gate_checks.values())
    qore_f1_delta = float(qore_delta["mean_f1_delta"])
    if passed:
        decision = "selector_responsive_ceiling_confirmed"
    elif qore_f1_delta > 0.01:
        decision = "borderline_selector_responsiveness"
    else:
        decision = "silver_evidence_retention_not_validated_as_f1_surrogate"
    return {
        "schema_version": 1,
        "artifact_type": "silver_oracle_top5_100_compact_result",
        "status": "completed",
        "evidence_tier": "L0_diagnostic",
        "diagnostic_only": True,
        "case_count": len(cases),
        "generator_call_count": len(cases) * len(EXPECTED_ARMS),
        "arm_summaries": arm_summaries,
        "paired_deltas": paired,
        "diagnostic_strata": strata,
        "gate": {"pass": passed, "checks": gate_checks, "thresholds": dict(thresholds)},
        "decision": decision,
        "limitations": [
            "The oracle uses gold-answer-conditioned three-model silver labels and is not deployable.",
            "This is a fixed 100-question diagnostic ceiling, not L1/L2 evidence or a SOTA claim.",
            "The experiment tests whether better Top-5 evidence membership transfers through the frozen Generator; it does not validate a new online selector.",
        ],
        "validation": {
            "contract": dict(contract),
            "common_context_order": True,
            "retrieval_calls": 0,
            "answer_scorer_calls": 0,
            "production_selector_mutated": False,
        },
    }


def _build_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Silver-oracle Top-5 Generator Ceiling (100 Questions)",
        "",
        "> Private L0 diagnostic. The oracle uses gold-answer-conditioned three-model labels and is not a deployable selector or a SOTA result.",
        "",
        f"- Generator: `{payload['generator']['model_id']}` revision `{payload['generator']['revision']}`",
        "- Arms: QORE membership, Top-k membership, silver-oracle membership.",
        "- All three arms are ordered by original retrieval rank before generation.",
        "- Retrieval and Answer Scorer calls: zero.",
        "",
    ]
    for case in payload["cases"]:
        lines.extend(
            [
                f"## Case {case['global_case_number']}: {case['question_id']}",
                "",
                f"**Question:** {case['question']}",
                "",
                "**Gold answers:** " + "; ".join(case["gold_answers"]),
                "",
                "### Registered Top-50",
                "",
            ]
        )
        for passage in case["top_50"]:
            evidence = passage["evidence"]
            lines.extend(
                [
                    f"{passage['retrieved_rank']}. **{passage.get('title') or '(untitled)'}** "
                    f"(id `{passage['id']}`, retrieval `{float(passage['retrieval_score']):.6f}`, "
                    f"Answer Scorer `{float(passage['answer_scorer_score']):.6f}`, "
                    f"direct votes `{evidence['direct_votes']}`, positive votes `{evidence['positive_votes']}`)",
                    str(passage.get("text") or passage.get("passage") or ""),
                    "",
                ]
            )
        lines.extend(["### Frozen-Generator arms", ""])
        for arm in case["arms"]:
            lines.extend(
                [
                    f"#### {arm['arm_id']}",
                    "",
                    f"Prediction: `{arm['prediction']}`",
                    f"EM `{arm['metrics']['em']:.3f}`; F1 `{arm['metrics']['f1']:.3f}`; "
                    f"direct/broad evidence `{arm['evidence_counts']['direct']}/{arm['evidence_counts']['broad']}`",
                    "",
                ]
            )
            for position, passage in enumerate(arm["selected_top_5"], start=1):
                lines.extend(
                    [
                        f"{position}. **{passage.get('title') or '(untitled)'}** "
                        f"(retrieved rank `{passage['retrieved_rank']}`, id `{passage['id']}`)",
                        str(passage.get("text") or passage.get("passage") or ""),
                        "",
                    ]
                )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path | None:
    root = _root()
    config_path = (args.config if args.config.is_absolute() else root / args.config).resolve()
    plan_path = (args.plan if args.plan.is_absolute() else root / args.plan).resolve()
    contract = validate_contract(config_path, plan_path)
    if args.validate_only:
        print(json.dumps(contract, sort_keys=True))
        return None
    phase = _load_yaml(config_path)["phase"]
    cases, source_provenance, integrity = load_registered_cases(root, phase, args)
    membership_integrity = _validate_membership_integrity(
        cases, phase["membership_integrity"]
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "case_count": len(cases),
                    "input_integrity": integrity,
                    "membership_integrity": membership_integrity,
                    "source_provenance": source_provenance,
                    "model_loaded": False,
                    "wiki_dpr_started": False,
                },
                sort_keys=True,
            )
        )
        return None
    identity = _resolve_generator(root, phase["generator"], args.model_path)
    from applications.rag.generation import Generator
    from applications.rag.silver_oracle_selector import count_consensus

    generator = Generator(
        identity["model_path"],
        max_new_tokens=int(phase["generator"]["max_new_tokens"]),
        use_chat_template=True,
    )
    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    target_dir = f"five_ideas/silver_oracle_top5_100/{run_dir.name}"
    started = time.perf_counter()
    output_cases: list[dict[str, Any]] = []
    for global_case_number, case in enumerate(cases, start=1):
        arm_inputs = build_arm_passages(case)
        arm_outputs: list[dict[str, Any]] = []
        for arm_id in EXPECTED_ARMS:
            arm = arm_inputs[arm_id]
            prediction, metrics, elapsed_ms = _evaluate_arm(
                question=str(case["question"]),
                answers=case["gold_answers"],
                passages=arm["passages"],
                generator=generator,
            )
            arm_outputs.append(
                {
                    "arm_id": arm_id,
                    "membership_source": (
                        "stored_qore_as"
                        if arm_id == "qore_common_order"
                        else "stored_topk_as"
                        if arm_id == "topk_common_order"
                        else "gold_conditioned_three_model_silver_oracle"
                    ),
                    "context_order": arm["context_order"],
                    "selected_top_5": [dict(item) for item in arm["passages"]],
                    "oracle_priority_trace": arm["oracle_priority_trace"],
                    "prediction": prediction,
                    "metrics": metrics,
                    "evidence_counts": count_consensus(arm["passages"]),
                    "generation_ms": elapsed_ms,
                }
            )
        output_cases.append(
            {
                "global_case_number": global_case_number,
                "source_id": case["source_id"],
                "source_case_number": case["source_case_number"],
                "global_index": case["global_index"],
                "question_id": case["question_id"],
                "question": case["question"],
                "gold_answers": case["gold_answers"],
                "top_50": case["top_50"],
                "top_50_evidence_counts": count_consensus(case["top_50"]),
                "arms": arm_outputs,
            }
        )
        print(f"  Silver-oracle Generator ceiling: {global_case_number}/100", flush=True)
    complete_payload = {
        "schema_version": 1,
        "artifact_type": "private_silver_oracle_top5_100_case_study",
        "evidence_tier": "L0_diagnostic",
        "diagnostic_only": True,
        "target_conditioned_oracle": True,
        "deployable_selector": False,
        "source_provenance": source_provenance,
        "input_integrity": integrity,
        "membership_integrity": membership_integrity,
        "generator": _compact_generator_identity(identity),
        "common_context_order": "ascending_original_retrieval_rank",
        "retrieval_calls": 0,
        "answer_scorer_calls": 0,
        "cases": output_cases,
        "limitations": [
            "The oracle sees gold-answer-conditioned three-model silver evidence labels.",
            "The artifact measures a retrieval-conditioned selector-realizable Generator ceiling only.",
            "It is not a production selector, SOTA result, or L1/L2 evidence.",
        ],
    }
    _write_json(run_dir / "case_study.json", complete_payload)
    (run_dir / "case_study.md").write_text(
        _build_markdown(complete_payload), encoding="utf-8"
    )
    result = summarize(output_cases, contract=contract, phase=phase)
    if _forbidden_fields(result):
        raise OracleRunError("compact result contains forbidden content fields")
    _write_json(run_dir / "result.json", result)
    if (run_dir / "result.json").stat().st_size > MAX_COMPACT_BYTES:
        raise OracleRunError("compact result exceeds the 1 MiB GitHub threshold")
    complete_files = []
    for name in ("case_study.json", "case_study.md"):
        path = run_dir / name
        complete_files.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "exchange_path": f"{target_dir}/{name}",
            }
        )
    metadata = {
        "schema_version": 1,
        "artifact_type": "silver_oracle_top5_100_run_metadata",
        "status": "completed",
        "evidence_tier": "L0_diagnostic",
        "diagnostic_only": True,
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        },
        "config": {"path_name": config_path.name, "sha256": _sha256(config_path)},
        "plugin_plan": {"path_name": plan_path.name, "sha256": _sha256(plan_path)},
        "source_provenance": source_provenance,
        "input_integrity": integrity,
        "membership_integrity": membership_integrity,
        "generator": _compact_generator_identity(identity),
        "arms": list(EXPECTED_ARMS),
        "generator_call_count": 300,
        "retrieval_calls": 0,
        "answer_scorer_calls": 0,
        "outputs": {
            "target_directory": target_dir,
            "complete_files_exchange_only": True,
            "complete_files": complete_files,
            "compact_result": {
                "name": "result.json",
                "bytes": (run_dir / "result.json").stat().st_size,
                "sha256": _sha256(run_dir / "result.json"),
            },
        },
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "decision": result["decision"],
    }
    if _forbidden_fields(metadata):
        raise OracleRunError("compact metadata contains forbidden content fields")
    _write_json(run_dir / "run_metadata.json", metadata)
    upload_manifest = {
        "schema_version": 1,
        "artifact_type": "silver_oracle_top5_100_upload_manifest",
        "status": "ready_for_authenticated_exchange_upload",
        "target_directory": target_dir,
        "generated_at_utc": timestamp,
        "git_commit": metadata["git"]["commit"],
        "exchange_required_files": ["case_study.json", "case_study.md"],
        "github_allowed_if_unchanged": ["result.json", "run_metadata.json", "upload_manifest.json"],
        "files": complete_files
        + [
            {
                "name": name,
                "bytes": (run_dir / name).stat().st_size,
                "sha256": _sha256(run_dir / name),
                "exchange_path": f"{target_dir}/{name}",
            }
            for name in ("result.json", "run_metadata.json")
        ],
        "publication_rule": "GitHub files must be at most 1 MiB and contain no raw question, passage, answer, prediction, or selected/retrieved ids.",
    }
    if _forbidden_fields(upload_manifest):
        raise OracleRunError("upload manifest contains forbidden content fields")
    _write_json(run_dir / "upload_manifest.json", upload_manifest)
    print(f"Completed: {run_dir}")
    print(f"Exchange target: {target_dir}")
    print("GitHub compact files: result.json run_metadata.json upload_manifest.json")
    print("Exchange complete files: case_study.json case_study.md")
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/experiments/silver_oracle_top5_100.yaml"
    )
    parser.add_argument(
        "--plan", type=Path, default=root / "configs/experiments/silver_oracle_top5_100_plan.json"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--detail-source-a", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--detail-source-b", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-source-a", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-source-b", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--panel-source-a", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--panel-source-b", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OracleRunError, OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
