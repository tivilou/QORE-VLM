#!/usr/bin/env python3
"""Replay a label-blind DPR span/relevance score screen on the fixed 100 x 50.

No DPR model, retriever, Generator, or evaluator is called. The runner consumes
the pinned compact Reader trace and the pinned case-study input; all Silver and
evidence labels are opened only after every method has produced its Top-5.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlsplit

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.rag.dpr_span_relevance_quantum import (
    REDUNDANCY_PENALTY,
    SpanFusionError,
    build_feature_matrix,
    classical_born_scores,
    classical_linear_scores,
    quantum_statevector_scores,
    ranked_indices,
    redundancy_aware_indices,
)


CONFIG_PATH = ROOT / "configs/experiments/dpr_span_relevance_quantum_screen_100.json"
DEFAULT_INPUT = ROOT / "research-web/apps/experiment-results/case-studies/silver-oracle-top5-100-20260915T120525Z-detail.json"
INPUT_EXCHANGE_PATH = "five_ideas/selector_replay_100_historical_input/silver-oracle-top5-100-20260915T120525Z-detail.json"
INPUT_BYTES = 8046318
DEFAULT_INPUT_SHA256 = "669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731"
TRACE_EXCHANGE_PATH = "five_ideas/dpr_reader_tensor_trace_100/20260930T143444Z/candidate_trace.json"
TRACE_BYTES = 3126685
TRACE_SHA256 = "13db49c56c0a5d61edf28156c5308e958f2c279e56867c6e773a135c81d56fa2"
EXCHANGE_URL = "http://117.50.198.37:18083"
MAX_GITHUB_BYTES = 1_048_576
CASE_COUNT = 100
TOP50 = 50
TOP5 = 5
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 314159
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
SCORE_TRACE_COLUMNS = (
    "reader_relevance_logit",
    "feature_relevance_z",
    "feature_span_z",
    "feature_margin_z",
    "feature_boundary_concentration_z",
    "classical_span_fusion_score",
    "quantum_span_interaction_score",
    "classical_matched_born_score",
)
CANONICAL_STAGES = (
    "data", "preprocess", "training", "retrieval", "scoring",
    "selection", "context", "generation", "evaluation", "diagnosis",
)


class ScreenError(RuntimeError):
    """Raised when the registered screen contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScreenError("cannot read JSON artifact: " + str(path)) from exc


def _connection(base_url: str, timeout: float = 300.0) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ScreenError("exchange URL must be a plain HTTP(S) URL")
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    return conn, parsed.path.rstrip("/")


def _download_artifact(base_url: str, token: str, exchange_path: str, destination: Path, expected_size: int, expected_sha256: str, label: str) -> None:
    conn, prefix = _connection(base_url, 900.0)
    try:
        path = prefix + "/files/" + quote(exchange_path, safe="/")
        conn.request("GET", path, headers={"Authorization": "Bearer " + token, "Accept": "application/octet-stream"})
        response = conn.getresponse()
        if response.status != 200:
            raise ScreenError(label + " download failed (HTTP " + str(response.status) + ")")
        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
                handle.write(block)
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise ScreenError("downloaded " + label + " failed its registered size/hash check")
    finally:
        conn.close()


def _download_trace(base_url: str, token: str, destination: Path) -> None:
    _download_artifact(base_url, token, TRACE_EXCHANGE_PATH, destination, TRACE_BYTES, TRACE_SHA256, "Reader trace")


def _download_input(base_url: str, token: str, destination: Path) -> None:
    _download_artifact(base_url, token, INPUT_EXCHANGE_PATH, destination, INPUT_BYTES, DEFAULT_INPUT_SHA256, "fixed 100-case input")


def _create_exchange_directory(base_url: str, token: str, relative_path: str) -> None:
    conn, prefix = _connection(base_url, 60.0)
    try:
        body = json.dumps({"path": relative_path}).encode("utf-8")
        conn.request(
            "POST",
            prefix + "/api/directories",
            body=body,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response = conn.getresponse()
        response_body = response.read(64 * 1024)
        if response.status not in {200, 201}:
            raise ScreenError("exchange directory creation failed (HTTP " + str(response.status) + ")")
        receipt = json.loads(response_body.decode("utf-8")) if response_body else {}
        if receipt.get("path") not in {relative_path, None}:
            raise ScreenError("exchange directory receipt path mismatch")
    finally:
        conn.close()


def _upload_file(base_url: str, token: str, source: Path, remote_path: str) -> dict[str, Any]:
    size = source.stat().st_size
    digest = _sha256(source)
    conn, prefix = _connection(base_url, 900.0)
    try:
        conn.putrequest("PUT", prefix + "/upload/" + quote(remote_path, safe="/"))
        conn.putheader("Authorization", "Bearer " + token)
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("Content-Length", str(size))
        conn.putheader("Accept", "application/json")
        conn.endheaders()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                conn.send(block)
        response = conn.getresponse()
        raw = response.read(64 * 1024)
        if response.status not in {200, 201}:
            raise ScreenError("exchange upload failed (HTTP " + str(response.status) + ")")
        receipt = json.loads(raw.decode("utf-8")) if raw else {}
        if (
            receipt.get("path") != remote_path
            or int(receipt.get("size_bytes", -1)) != size
            or receipt.get("sha256") != digest
        ):
            raise ScreenError("exchange upload receipt mismatch")
        return receipt
    finally:
        conn.close()


def _load_input(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ScreenError("fixed 100-case input is absent or unsafe: " + str(path))
    if _sha256(path) != DEFAULT_INPUT_SHA256:
        raise ScreenError("fixed 100-case input SHA-256 does not match the registered artifact")
    bundle = _load_json(path)
    cases = bundle.get("cases") if isinstance(bundle, Mapping) else None
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ScreenError("fixed input must contain exactly 100 cases")
    for case_number, case in enumerate(cases, 1):
        if not isinstance(case, Mapping) or not isinstance(case.get("top_50"), list) or len(case["top_50"]) != TOP50:
            raise ScreenError("case " + str(case_number) + " does not have exactly 50 candidates")
        identifiers = [str(item.get("id", "")) for item in case["top_50"] if isinstance(item, Mapping)]
        ranks = [item.get("retrieved_rank") for item in case["top_50"] if isinstance(item, Mapping)]
        if len(identifiers) != TOP50 or len(set(identifiers)) != TOP50 or ranks != list(range(1, TOP50 + 1)):
            raise ScreenError("case " + str(case_number) + " failed candidate ID/rank invariants")
    return bundle


def _load_trace(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ScreenError("Reader trace is absent or unsafe: " + str(path))
    if path.stat().st_size != TRACE_BYTES or _sha256(path) != TRACE_SHA256:
        raise ScreenError("Reader trace size/SHA-256 does not match the registered artifact")
    trace = _load_json(path)
    if not isinstance(trace, Mapping) or trace.get("schema_version") != "rag.dpr_reader_tensor_trace_100.candidate_trace.v1":
        raise ScreenError("unexpected DPR Reader candidate-trace schema")
    input_info = trace.get("input", {})
    if input_info.get("sha256") != DEFAULT_INPUT_SHA256:
        raise ScreenError("Reader trace was not produced from the registered fixed input")
    rows = trace.get("cases")
    if not isinstance(rows, list) or len(rows) != CASE_COUNT * TOP50:
        raise ScreenError("Reader trace must contain exactly 5,000 rows")
    return dict(trace)


def _stored_ids(case: Mapping[str, Any], selector_id: str) -> list[str]:
    selectors = case.get("selectors")
    if not isinstance(selectors, list):
        raise ScreenError("case is missing the stored baseline selector list")
    for selector in selectors:
        if isinstance(selector, Mapping) and selector.get("selector_id") == selector_id:
            selected = selector.get("selected_top_5")
            if not isinstance(selected, list) or len(selected) != TOP5:
                raise ScreenError("stored baseline is not exactly five passages: " + selector_id)
            ids = [str(item.get("id", "")) for item in selected if isinstance(item, Mapping)]
            if len(ids) != TOP5 or len(set(ids)) != TOP5:
                raise ScreenError("stored baseline IDs are invalid: " + selector_id)
            return ids
    raise ScreenError("stored baseline is missing: " + selector_id)


def _project_online_candidates(case: Mapping[str, Any], trace_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Allowlist score inputs; evidence and answer fields do not cross this boundary."""

    top50 = case["top_50"]
    if len(trace_rows) != TOP50:
        raise ScreenError("each question must map to 50 trace rows")
    result = []
    for index, (candidate, row) in enumerate(zip(top50, trace_rows)):
        identifier = str(candidate.get("id", ""))
        rank = int(candidate.get("retrieved_rank", -1))
        if (
            int(row.get("case_number", -1)) != int(case.get("case_number", -2))
            or int(row.get("candidate_index", -1)) != index
            or str(row.get("candidate_id", "")) != identifier
            or int(row.get("retrieved_rank", -1)) != rank
        ):
            raise ScreenError("trace/input candidate identity or order mismatch")
        online = {
            "candidate_id": identifier,
            "retrieved_rank": rank,
            "title": str(candidate.get("title") or ""),
            "text": str(candidate.get("text") or ""),
        }
        for key in ("relevance_logit", "span_logit", "span_margin", "start_entropy", "end_entropy"):
            online[key] = row[key]
        result.append(online)
    return result


def _case_predictions(case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trace_rows = candidates
    features = build_feature_matrix(trace_rows)
    reader_scores = np.asarray([float(row["relevance_logit"]) for row in trace_rows], dtype=np.float64)
    classical_scores = classical_linear_scores(features)
    quantum_scores = quantum_statevector_scores(features)
    born_scores = classical_born_scores(features)
    parity_error = float(np.max(np.abs(quantum_scores - born_scores)))
    if parity_error > 1e-10:
        raise ScreenError("quantum/classical Born parity failed: " + str(parity_error))

    reader_indices = ranked_indices(reader_scores, trace_rows)
    topk_ids = _stored_ids(case, "topk_common_order")
    reader_ids = [str(trace_rows[index]["candidate_id"]) for index in reader_indices]
    if reader_ids != topk_ids:
        raise ScreenError("Reader trace Top-5 does not exactly match the registered Top-k baseline")
    qore_ids = _stored_ids(case, "qore_common_order")
    universe = {str(row["candidate_id"]) for row in trace_rows}
    if not set(qore_ids).issubset(universe) or not set(topk_ids).issubset(universe):
        raise ScreenError("stored baseline contains IDs outside the fixed Top-50")

    selections: dict[str, dict[str, Any]] = {
        "answer_scorer_topk": {"indices": reader_indices, "scores": reader_scores.tolist()},
        "qore_answer_scorer": {"ids": qore_ids},
    }
    score_arms = {
        "classical_span_fusion": classical_scores,
        "quantum_span_interaction": quantum_scores,
        "classical_matched_born": born_scores,
    }
    for method_id, scores in score_arms.items():
        selections[method_id] = {"indices": ranked_indices(scores, trace_rows), "scores": scores.tolist()}
        dedup_indices = redundancy_aware_indices(scores, trace_rows)
        selections[method_id + "_dedup"] = {"indices": dedup_indices, "scores": scores.tolist()}
    selection_steps: dict[str, dict[int, int]] = {}
    for method_id, selection in selections.items():
        if "indices" in selection:
            selection_steps[method_id] = {index: rank for rank, index in enumerate(selection["indices"], 1)}
        else:
            id_to_index = {str(row["candidate_id"]): index for index, row in enumerate(trace_rows)}
            selection_steps[method_id] = {
                id_to_index[identifier]: rank
                for rank, identifier in enumerate(selection["ids"], 1)
                if identifier in id_to_index
            }
    score_trace_rows = []
    for index, row in enumerate(trace_rows):
        score_trace_rows.append({
            "case_number": int(case["case_number"]),
            "candidate_index": index,
            "candidate_id": str(row["candidate_id"]),
            "retrieved_rank": int(row["retrieved_rank"]),
            "reader_relevance_logit": round(float(reader_scores[index]), 8),
            "features": [round(float(value), 8) for value in features[index]],
            "classical_span_fusion_score": round(float(classical_scores[index]), 8),
            "quantum_span_interaction_score": round(float(quantum_scores[index]), 8),
            "classical_matched_born_score": round(float(born_scores[index]), 8),
            "numeric_values": [
                round(float(reader_scores[index]), 8),
                *[round(float(value), 8) for value in features[index]],
                round(float(classical_scores[index]), 8),
                round(float(quantum_scores[index]), 8),
                round(float(born_scores[index]), 8),
            ],
            "selection_rank": {
                method_id: selection_steps[method_id].get(index)
                for method_id in sorted(selection_steps)
            },
        })
    return {
        "candidate_rows": list(trace_rows),
        "selections": selections,
        "score_trace_rows": score_trace_rows,
        "quantum_classical_parity_max_abs": parity_error,
        "circuit": {
            "qubits": 4,
            "data_encoding": "RY(arccos(x_i))",
            "entangler": "directed CNOT ring (0->1,1->2,2->3,3->0)",
            "observables": ["Z_i for i=0..3", "Z_i Z_(i+1 mod 4) for i=0..3"],
            "trainable_parameters": 0,
            "two_qubit_gates": 4,
            "two_qubit_depth": 2,
            "score": "0.5 * mean(local Z) + 0.5 * mean(ring ZZ)",
        },
    }


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(TOKEN_RE.findall(left.lower()))
    right_tokens = set(TOKEN_RE.findall(right.lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _posthoc_case_metrics(
    case: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Read Silver/panel labels only after all selector decisions are frozen."""

    top50 = list(case["top_50"])
    candidate_by_id = {str(item.get("id", "")): item for item in top50}
    silver_ids = set(_stored_ids(case, "silver_oracle_common_order"))
    broad_by_id = {}
    direct_by_id = {}
    for candidate in top50:
        evidence = candidate.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        broad = evidence.get("positive_consensus")
        direct = evidence.get("direct_consensus")
        if not isinstance(broad, bool):
            broad = evidence.get("consensus_label") in {"direct", "partial"}
        if not isinstance(direct, bool):
            direct = evidence.get("consensus_label") == "direct"
        identifier = str(candidate.get("id", ""))
        broad_by_id[identifier] = bool(broad)
        direct_by_id[identifier] = bool(direct)
    top50_broad = sum(int(value) for value in broad_by_id.values())
    top50_direct = sum(int(value) for value in direct_by_id.values())

    metrics: dict[str, Any] = {}
    for method_id, selection in prediction["selections"].items():
        if "ids" in selection:
            selected_ids = list(selection["ids"])
            selected_scores = []
        else:
            indices = list(selection["indices"])
            selected_ids = [str(prediction["candidate_rows"][index]["candidate_id"]) for index in indices]
            selected_scores = [round(float(selection["scores"][index]), 8) for index in indices]
        if len(selected_ids) != TOP5 or len(set(selected_ids)) != TOP5:
            raise ScreenError(method_id + " did not select five unique candidates")
        selected = [candidate_by_id[identifier] for identifier in selected_ids]
        pairwise = [_jaccard(str(selected[i].get("text", "")), str(selected[j].get("text", "")))
                    for i in range(TOP5) for j in range(i + 1, TOP5)]
        titles = [str(item.get("title") or "") for item in selected]
        metrics[method_id] = {
            "selected_ids": selected_ids,
            "selected_ranks": [int(item.get("retrieved_rank", -1)) for item in selected],
            "selected_scores": selected_scores,
            "silver_oracle_overlap_count": len(set(selected_ids) & silver_ids),
            "selected_broad_positive_count": sum(int(broad_by_id[identifier]) for identifier in selected_ids),
            "selected_direct_positive_count": sum(int(direct_by_id[identifier]) for identifier in selected_ids),
            "top50_broad_positive_count": top50_broad,
            "top50_direct_positive_count": top50_direct,
            "retrieval_miss": top50_broad == 0,
            "selector_miss": top50_broad > 0 and not any(broad_by_id[identifier] for identifier in selected_ids),
            "token_jaccard_mean": float(np.mean(pairwise)) if pairwise else None,
            "near_duplicate_pair_count_jaccard_ge_0_8": sum(1 for value in pairwise if value >= 0.8),
            "unique_title_count": len(set(titles)),
        }
    return metrics


def _bootstrap_delta(values: Sequence[float], baseline: Sequence[float]) -> dict[str, float]:
    left = np.asarray(values, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != (CASE_COUNT,) or right.shape != (CASE_COUNT,):
        raise ScreenError("paired bootstrap requires 100 case-level values")
    delta = left - right
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, CASE_COUNT, size=(BOOTSTRAP_SAMPLES, CASE_COUNT))
    means = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "paired_bootstrap_95_low": float(np.quantile(means, 0.025)),
        "paired_bootstrap_95_high": float(np.quantile(means, 0.975)),
        "wins": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses": int(np.sum(delta < 0)),
    }


def _summarize(case_results: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    method_ids = list(case_results[0])
    topk = [float(case_results[i]["answer_scorer_topk"]["silver_oracle_overlap_count"]) for i in range(CASE_COUNT)]
    qore = [float(case_results[i]["qore_answer_scorer"]["silver_oracle_overlap_count"]) for i in range(CASE_COUNT)]
    summaries: dict[str, Any] = {}
    for method_id in method_ids:
        rows = [case_results[i][method_id] for i in range(CASE_COUNT)]
        overlap = [float(row["silver_oracle_overlap_count"]) for row in rows]
        mean_value = lambda key: float(np.mean([float(row[key]) for row in rows]))
        summary = {
            "mean_silver_oracle_overlap_out_of_5": mean_value("silver_oracle_overlap_count"),
            "mean_selected_broad_positive_count_out_of_5": mean_value("selected_broad_positive_count"),
            "mean_selected_direct_positive_count_out_of_5": mean_value("selected_direct_positive_count"),
            "mean_token_jaccard": mean_value("token_jaccard_mean"),
            "mean_near_duplicate_pairs_per_top5": mean_value("near_duplicate_pair_count_jaccard_ge_0_8"),
            "mean_unique_titles_in_top5": mean_value("unique_title_count"),
            "paired_delta_vs_answer_scorer_topk": _bootstrap_delta(overlap, topk),
            "paired_delta_vs_qore_answer_scorer": _bootstrap_delta(overlap, qore),
        }
        summaries[method_id] = summary
    parity = max(float(item["quantum_classical_parity_max_abs"]) for item in predictions)
    summaries["quantum_classical_parity_max_abs"] = parity
    return summaries


def _display(summary: str, fields: Sequence[Mapping[str, Any]] = (), visualizations: Sequence[Mapping[str, Any]] = (), default_view: str = "table") -> dict[str, Any]:
    return {
        "summary": summary,
        "default_view": default_view,
        "visible_fields": list(fields),
        "visualizations": list(visualizations),
    }


def _field(field_id: str, label: str, help_text: str, value: Any) -> dict[str, Any]:
    return {"id": field_id, "label": label, "help": help_text, "value": value}


def _not_applicable_stage(sample_id: str, canonical_stage: str, step: int, reason: str) -> dict[str, Any]:
    return {
        "stage_id": sample_id + "-" + canonical_stage,
        "stage": canonical_stage,
        "canonical_stage": canonical_stage,
        "branch": None,
        "lane": "replay",
        "status": "not_applicable",
        "step": step,
        "checkpoint": None,
        "purpose": "Record that this phase is outside the frozen selector-only replay.",
        "summary": reason,
        "reason": reason,
        "input_refs": [],
        "output_refs": [],
        "inputs": {},
        "outputs": {},
        "display": _display(reason, default_view="not_available"),
        "help": {"status": "No operation for this phase is part of this experiment."},
    }


def read_score_matrix(path: Path, case_number: int) -> tuple[list[str], np.ndarray]:
    """Return the real 50x8 score matrix and candidate IDs for one case."""

    doc = _load_json(path)
    if not isinstance(doc, Mapping) or doc.get("schema_version") != "rag.dpr_span_relevance_quantum_screen_100.score_trace.v1":
        raise ScreenError("score matrix reader received an unsupported trace")
    sample_id = "case-{:03d}".format(case_number)
    for sample in doc.get("samples", []):
        if sample.get("sample_id") == sample_id:
            matrix = np.asarray(sample.get("numeric_matrix"), dtype=np.float64)
            identifiers = [str(value) for value in sample.get("candidate_ids", [])]
            if matrix.shape != (TOP50, len(SCORE_TRACE_COLUMNS)) or len(identifiers) != TOP50:
                raise ScreenError("score trace matrix has invalid coverage or shape")
            if not np.all(np.isfinite(matrix)):
                raise ScreenError("score trace matrix contains non-finite values")
            return identifiers, matrix
    raise ScreenError("score trace is missing " + sample_id)


def _build_sample_trace(
    run_id: str,
    input_path: Path,
    input_bytes: int,
    score_trace_ref: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    case_results: Sequence[Mapping[str, Any]],
    config_hash: str,
) -> dict[str, Any]:
    score_artifact = {
        "path": str(score_trace_ref["path"]),
        "sha256": str(score_trace_ref["sha256"]),
        "bytes": int(score_trace_ref["bytes"]),
        "visibility": str(score_trace_ref["visibility"]),
    }
    samples = []
    method_ids = list(case_results[0])
    for index, (case, prediction, metrics) in enumerate(zip(cases, predictions, case_results), 1):
        sample_id = "case-{:03d}".format(index)
        sample_rows = prediction["score_trace_rows"]
        candidate_ids = [str(row["candidate_id"]) for row in sample_rows]
        matrix = [list(row["numeric_values"]) for row in sample_rows]
        if len(matrix) != TOP50 or any(len(row) != len(SCORE_TRACE_COLUMNS) for row in matrix):
            raise ScreenError("cannot build complete score representation for " + sample_id)
        selection_outputs = {}
        for method_id in method_ids:
            selection_outputs[method_id] = list(metrics[method_id]["selected_ids"])
        artifact_ref = dict(score_artifact)
        representation = {
            "id": "reader_selector_score_matrix",
            "stage_id": sample_id + "-scoring",
            "status": "observed",
            "meaning": "Complete per-candidate Reader and selector score values in the fixed Top-50 order.",
            "shape": [TOP50, len(SCORE_TRACE_COLUMNS)],
            "dtype": "float64",
            "axis_semantics": [
                "row axis: candidate_index 0..49 in retrieved_rank order",
                "column axis: " + ", ".join(SCORE_TRACE_COLUMNS),
            ],
            "portal_preview": {"first_candidate_id": candidate_ids[0], "first_row": matrix[0]},
            "artifact_refs": [artifact_ref],
            "value_capture": {
                "status": "complete",
                "shape": [TOP50, len(SCORE_TRACE_COLUMNS)],
                "logical_element_count": TOP50 * len(SCORE_TRACE_COLUMNS),
                "stored_element_count": TOP50 * len(SCORE_TRACE_COLUMNS),
                "artifact_refs": [artifact_ref],
                "reader_projection": {
                    "status": "available",
                    "kind": "domain_specific",
                    "coordinate_access": "bounded_tiles",
                    "reader_ref": "scripts/collab/five_ideas/run_dpr_span_relevance_quantum_screen_100.py::read_score_matrix",
                    "source_sha256": [score_artifact["sha256"]],
                },
            },
            "display": _display(
                "50 个候选 x 8 项 Reader/selector 数值；按检索 rank 对齐。",
                fields=[
                    _field("columns", "数值列", "每列对应 Reader 原始 relevance、4 个归一化特征或一个固定评分臂。", list(SCORE_TRACE_COLUMNS)),
                    _field("shape", "矩阵形状", "第一维是候选段，第二维是特征/评分列。", [TOP50, len(SCORE_TRACE_COLUMNS)]),
                ],
                visualizations=[{
                    "id": "reader-selector-score-matrix",
                    "kind": "heatmap",
                    "title": "Reader 与筛选分数",
                    "help": "按候选 rank 查看不同 Reader 信号和筛选评分的变化。",
                    "data_ref": "score_trace:" + sample_id + "/numeric_matrix",
                    "artifact_refs": [artifact_ref],
                }],
            ),
        }
        stages = []
        stages.append({
            "stage_id": sample_id + "-data",
            "stage": "Fixed candidate input",
            "canonical_stage": "data",
            "branch": None,
            "lane": "replay",
            "status": "observed",
            "step": 1,
            "checkpoint": None,
            "purpose": "Bind this sample to its unchanged 50 retrieved candidates.",
            "summary": "Fixed Top-50 input; candidate text is not copied into output artifacts.",
            "reason": None,
            "input_refs": ["source:detail-json#cases/{}".format(index - 1)],
            "output_refs": ["sample:{}:candidate-identity".format(sample_id)],
            "inputs": {"input_sha256": DEFAULT_INPUT_SHA256},
            "outputs": {"case_number": index, "candidate_count": TOP50, "candidate_ids": candidate_ids},
            "display": _display(
                "固定问题样本及其 50 个候选段。",
                fields=[
                    _field("case_number", "样本编号", "固定 100 题数据中的顺序编号。", index),
                    _field("candidate_count", "候选数", "进入 Top-5 筛选的冻结候选段数量。", TOP50),
                ],
            ),
            "help": {"candidate_ids": "稳定 ID 用于验证各评分方法始终从同一候选集合选择。"},
            "artifact_refs": [],
            "target_access": "outside_generation_input",
        })
        stages.append(_not_applicable_stage(sample_id, "preprocess", 2, "This replay consumes the already materialized fixed candidate input."))
        stages.append(_not_applicable_stage(sample_id, "training", 3, "The Reader and fixed scoring rules are not trained or fitted in this screen."))
        stages.append(_not_applicable_stage(sample_id, "retrieval", 4, "Retrieval is frozen; the existing Top-50 is reused."))
        stages.append({
            "stage_id": sample_id + "-scoring",
            "stage": "DPR Reader feature scoring",
            "canonical_stage": "scoring",
            "branch": None,
            "lane": "replay",
            "status": "observed",
            "step": 5,
            "checkpoint": "facebook/dpr-reader-single-nq-base@38f47a4986084c53447ba92ab0a83076b58d86a8",
            "purpose": "Fuse frozen Reader relevance, span strength, boundary margin, and concentration.",
            "summary": "One complete 50x8 numeric trace is preserved for this question.",
            "reason": None,
            "input_refs": ["sample:{}:candidate-identity".format(sample_id), "external:" + TRACE_EXCHANGE_PATH],
            "output_refs": ["score_trace:" + sample_id + "/numeric_matrix"],
            "inputs": {"candidate_count": TOP50, "features": list(SCORE_TRACE_COLUMNS[:5])},
            "outputs": {"shape": [TOP50, len(SCORE_TRACE_COLUMNS)], "method_scores": list(SCORE_TRACE_COLUMNS[5:])},
            "display": _display(
                "DPR Reader 信号及经典/量子评分臂。",
                fields=[
                    _field("reader_signals", "Reader 输入信号", "原始 trace 提供 relevance、span、span margin 与 start/end entropy。", list(SCORE_TRACE_COLUMNS[:5])),
                    _field("method_scores", "筛选评分", "这些分数只决定 Top-5 顺序；Silver 标签不参与计算。", list(SCORE_TRACE_COLUMNS[5:])),
                ],
                visualizations=[{
                    "id": "score-comparison",
                    "kind": "line",
                    "title": "候选评分比较",
                    "help": "在相同 50 个候选上对比基线、经典融合和量子交互评分。",
                    "data_ref": "score_trace:" + sample_id + "/numeric_matrix",
                    "artifact_refs": [artifact_ref],
                }],
            ),
            "help": {"quantum_score": "4 qubit 固定环形纠缠评分，0 个训练参数；其 Born 期望必须与解析经典控制一致。"},
            "artifact_refs": [artifact_ref],
            "target_access": "outside_generation_input",
        })
        stages.append({
            "stage_id": sample_id + "-selection",
            "stage": "Independent Top-5 selections",
            "canonical_stage": "selection",
            "branch": None,
            "lane": "replay",
            "status": "observed",
            "step": 6,
            "checkpoint": None,
            "purpose": "Select exactly five IDs per registered baseline and score arm.",
            "summary": "All methods use the same 50 IDs; the redundancy penalty is a matched ablation.",
            "reason": None,
            "input_refs": ["score_trace:" + sample_id + "/numeric_matrix"],
            "output_refs": ["selection:" + sample_id],
            "inputs": {"candidate_count": TOP50, "k": TOP5},
            "outputs": {"selected_top5_by_method": selection_outputs},
            "display": _display(
                "逐方法显示从固定候选中实际选出的 5 个 passage ID。",
                fields=[
                    _field("method_ids", "筛选方法", "包括 Answer Scorer、QORE、经典融合、量子交互与冗余消融。", method_ids),
                    _field("selected_ids", "Top-5 ID", "每个列表恰含五个唯一候选 ID。", selection_outputs),
                ],
            ),
            "help": {"selected_top5_by_method": "所有选择在 post-hoc 读取 silver/evidence 标签之前完成。"},
            "artifact_refs": [artifact_ref],
            "target_access": "outside_generation_input",
        })
        stages.append(_not_applicable_stage(sample_id, "context", 7, "No Generator context is assembled in this selector-only replay."))
        stages.append(_not_applicable_stage(sample_id, "generation", 8, "No language-model answer is generated in this screen."))
        stages.append(_not_applicable_stage(sample_id, "evaluation", 9, "No generated answer exists, so EM/F1 are not evaluated."))
        stages.append({
            "stage_id": sample_id + "-diagnosis",
            "stage": "Post-hoc Silver/evidence diagnosis",
            "canonical_stage": "diagnosis",
            "branch": None,
            "lane": "replay",
            "status": "observed",
            "step": 10,
            "checkpoint": None,
            "purpose": "Compare frozen selections with Silver/evidence labels only after selection.",
            "summary": "Selection diagnostics are not gold answer accuracy or full-data evidence.",
            "reason": None,
            "input_refs": ["selection:" + sample_id, "external:detail-json#cases/{}".format(index - 1)],
            "output_refs": ["result.json#cases/{}".format(index - 1)],
            "inputs": {"labels_available_before_selection": False},
            "outputs": metrics,
            "display": _display(
                "选择后才显示 Silver overlap、direct/broad evidence 与冗余指标。",
                fields=[
                    _field("silver_overlap", "Silver overlap", "Top-5 与三模型 Silver oracle Top-5 的 ID 交集数量，仅作 L0 诊断。", {key: value["silver_oracle_overlap_count"] for key, value in metrics.items()}),
                    _field("direct_evidence", "Direct evidence", "三模型共同判断为直接支持问题答案的段落数；标签只在此阶段读取。", {key: value["selected_direct_positive_count"] for key, value in metrics.items()}),
                    _field("redundancy", "片段冗余", "Top-5 内的平均 token Jaccard 和近重复对数。", {key: value["token_jaccard_mean"] for key, value in metrics.items()}),
                ],
            ),
            "help": {"silver_oracle_overlap_count": "这是 Silver 选择重合，不等同于 gold evidence recall 或最终答题正确率。"},
            "artifact_refs": [],
            "target_access": "outside_generation_input",
        })
        samples.append({
            "sample_id": sample_id,
            "lane": "replay",
            "split": "fixed_100_silver_panel",
            "checkpoint": "frozen DPR Reader trace",
            "roles": {
                "fixed_candidate_set": "source:detail-json#cases/{}".format(index - 1),
                "reader_features": "score_trace:" + sample_id + "/numeric_matrix",
                "top5_selection": "selection:" + sample_id,
                "posthoc_diagnostic": "result.json#cases/{}".format(index - 1),
            },
            "representations": [representation],
            "stages": stages,
        })
    trace = {
        "schema_version": "sample-trace.v2",
        "trace_id": "dpr-span-relevance-100-" + run_id,
        "trace_completeness": "complete",
        "task_contract": {
            "task_type": "fixed_top50_to_top5_selector_replay",
            "required_roles": ["fixed_candidate_set", "reader_features", "top5_selection", "posthoc_diagnostic"],
            "required_representations": ["reader_selector_score_matrix"],
            "required_value_representations": ["reader_selector_score_matrix"],
            "required_splits": ["fixed_100_silver_panel"],
            "required_checkpoints": ["frozen DPR Reader trace"],
            "required_lanes": ["replay"],
            "checkpoint_policy": "The DPR Reader model/revision is frozen; this replay performs no new model inference.",
        },
        "experiment": {
            "run_id": run_id,
            "dataset": "NQ-Open historical Silver case study, fixed 100 questions x 50 passages",
            "code_revision": _git_revision() or "unavailable",
            "config_sha256": config_hash,
            "model_identity": {
                "model_id": "facebook/dpr-reader-single-nq-base",
                "revision": "38f47a4986084c53447ba92ab0a83076b58d86a8",
                "config_sha256": "383c5dbe735db108976bcaef700483481def480bfe24ed8ff5243cd1ce4751",
            },
            "seed": BOOTSTRAP_SEED,
            "input_sha256": DEFAULT_INPUT_SHA256,
            "reader_trace_sha256": TRACE_SHA256,
            "input_artifacts": [
                {"path": INPUT_EXCHANGE_PATH, "sha256": DEFAULT_INPUT_SHA256, "bytes": input_bytes},
                {"path": TRACE_EXCHANGE_PATH, "sha256": TRACE_SHA256, "bytes": TRACE_BYTES},
            ],
        },
        "sample_selection": {
            "rule": "Include every sample in the registered 100-case fixed diagnostic; preserve case order.",
            "population_scope": "The two previously registered 50-question Silver case-study slices; fixed Top-50 per question.",
            "seed": BOOTSTRAP_SEED,
            "selected_count": CASE_COUNT,
            "selected_sample_ids": [sample["sample_id"] for sample in samples],
        },
        "coverage": {
            "data": "observed",
            "preprocess": "not_applicable",
            "training": "not_applicable",
            "retrieval": "not_applicable",
            "scoring": "observed",
            "selection": "observed",
            "context": "not_applicable",
            "generation": "not_applicable",
            "evaluation": "not_applicable",
            "diagnosis": "observed",
        },
        "samples": samples,
    }
    return trace


def validate_sample_trace_semantics(
    trace: Mapping[str, Any],
    score_trace: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
) -> None:
    """Project-owned readiness checks for complete values, IDs, stage order, and leakage."""

    if trace.get("schema_version") != "sample-trace.v2" or len(trace.get("samples", [])) != CASE_COUNT:
        raise ScreenError("sample-trace.v2 schema/case coverage check failed")
    if list(trace.get("coverage", {}).keys()) != list(CANONICAL_STAGES):
        raise ScreenError("sample-trace.v2 canonical coverage order failed")
    if len(score_trace.get("samples", [])) != CASE_COUNT:
        raise ScreenError("score trace does not cover all 100 samples")
    forbidden = {"question", "text", "gold_answers", "gold_answer", "evidence", "positive_consensus", "direct_consensus", "silver_oracle_ids"}

    def walk_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden.intersection(value.keys())
            if overlap:
                raise ScreenError("compact trace contains a forbidden content/label field: " + sorted(overlap)[0])
            for item in value.values():
                walk_keys(item)
        elif isinstance(value, list):
            for item in value:
                walk_keys(item)

    walk_keys(score_trace)
    walk_keys(trace)
    for index, sample in enumerate(trace["samples"], 1):
        sample_id = "case-{:03d}".format(index)
        if sample.get("sample_id") != sample_id or [stage.get("canonical_stage") for stage in sample.get("stages", [])] != list(CANONICAL_STAGES):
            raise ScreenError("sample-trace stage order/identity failed for " + sample_id)
        representation = sample.get("representations", [{}])[0]
        if representation.get("id") != "reader_selector_score_matrix" or representation.get("shape") != [TOP50, len(SCORE_TRACE_COLUMNS)]:
            raise ScreenError("numeric representation declaration failed for " + sample_id)
        capture = representation.get("value_capture", {})
        projection = capture.get("reader_projection", {})
        refs = representation.get("artifact_refs", [])
        if (
            capture.get("status") != "complete"
            or capture.get("logical_element_count") != TOP50 * len(SCORE_TRACE_COLUMNS)
            or capture.get("stored_element_count") != TOP50 * len(SCORE_TRACE_COLUMNS)
            or capture.get("shape") != [TOP50, len(SCORE_TRACE_COLUMNS)]
            or projection.get("status") != "available"
            or projection.get("source_sha256") != [refs[0].get("sha256") if refs else None]
        ):
            raise ScreenError("complete value-capture contract failed for " + sample_id)
        identifiers, matrix = read_score_matrix_from_document(score_trace, index)
        data_stage = sample["stages"][0]
        if len(set(identifiers)) != TOP50 or identifiers != data_stage.get("outputs", {}).get("candidate_ids"):
            raise ScreenError("sample score values do not preserve candidate IDs for " + sample_id)
        if matrix.shape != (TOP50, len(SCORE_TRACE_COLUMNS)) or not np.all(np.isfinite(matrix)):
            raise ScreenError("complete finite score-value capture failed for " + sample_id)
        selected = sample["stages"][5].get("outputs", {}).get("selected_top5_by_method", {})
        diagnosis = sample["stages"][9].get("outputs", {})
        if sample["stages"][4].get("target_access") != "outside_generation_input" or sample["stages"][5].get("target_access") != "outside_generation_input":
            raise ScreenError("posthoc target boundary failed for " + sample_id)
        expected = case_results[index - 1]
        for method_id in expected:
            if selected.get(method_id) != expected[method_id]["selected_ids"]:
                raise ScreenError("same-sample selection replay failed for " + sample_id + "/" + method_id)
            if diagnosis.get(method_id) != expected[method_id]:
                raise ScreenError("posthoc metric lineage failed for " + sample_id + "/" + method_id)


def validate_sample_trace_artifacts(
    sample_trace_path: Path,
    score_trace_path: Path,
    expected_sample_sha256: str,
    expected_sample_bytes: int,
    expected_score_sha256: str,
    expected_score_bytes: int,
    predictions: Sequence[Mapping[str, Any]],
    case_results: Sequence[Mapping[str, Any]],
) -> None:
    """Reopen detached trace files and verify their hashes, values, and selections."""

    for path, expected_sha256, expected_bytes, label in (
        (sample_trace_path, expected_sample_sha256, expected_sample_bytes, "sample trace"),
        (score_trace_path, expected_score_sha256, expected_score_bytes, "score trace"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ScreenError(label + " artifact is missing or unsafe")
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
            raise ScreenError(label + " artifact failed its written size/SHA-256 check")

    trace = _load_json(sample_trace_path)
    score_trace = _load_json(score_trace_path)
    if not isinstance(trace, Mapping) or not isinstance(score_trace, Mapping):
        raise ScreenError("detached sample/score traces must be JSON objects")
    validate_sample_trace_semantics(trace, score_trace, case_results)
    if score_trace.get("columns") != list(SCORE_TRACE_COLUMNS) or score_trace.get("online_labels_used") is not False:
        raise ScreenError("detached score trace columns or label boundary changed")

    trace_samples = trace["samples"]
    score_samples = score_trace["samples"]
    if len(predictions) != CASE_COUNT or len(case_results) != CASE_COUNT:
        raise ScreenError("detached value replay requires all 100 predictions and cases")
    for index, (sample, score_sample, prediction) in enumerate(zip(trace_samples, score_samples, predictions), 1):
        sample_id = "case-{:03d}".format(index)
        candidate_ids, matrix = read_score_matrix_from_document(score_trace, index)
        prediction_rows = prediction["score_trace_rows"]
        expected_ids = [str(row["candidate_id"]) for row in prediction_rows]
        expected_ranks = [int(row["retrieved_rank"]) for row in prediction_rows]
        expected_matrix = np.asarray([row["numeric_values"] for row in prediction_rows], dtype=np.float64)
        if candidate_ids != expected_ids or score_sample.get("retrieved_ranks") != expected_ranks:
            raise ScreenError("detached candidate ID/rank coverage changed for " + sample_id)
        if matrix.shape != (TOP50, len(SCORE_TRACE_COLUMNS)) or not np.allclose(matrix, expected_matrix, rtol=0.0, atol=1e-8):
            raise ScreenError("detached score values do not replay the in-memory reader slice for " + sample_id)

        representation = sample["representations"][0]
        refs = representation.get("artifact_refs", [])
        if not refs:
            raise ScreenError("sample trace has no detached score artifact reference for " + sample_id)
        expected_ref = {
            "path": refs[0]["path"],
            "sha256": expected_score_sha256,
            "bytes": expected_score_bytes,
            "visibility": refs[0]["visibility"],
        }
        if refs != [expected_ref] or representation["value_capture"].get("artifact_refs") != [expected_ref]:
            raise ScreenError("sample trace does not reference the verified score artifact for " + sample_id)

        selected = sample["stages"][5]["outputs"]["selected_top5_by_method"]
        selection_rows = score_sample.get("selection_ranks")
        if not isinstance(selection_rows, list) or len(selection_rows) != TOP50:
            raise ScreenError("detached selection-rank coverage failed for " + sample_id)
        if len(set(score_sample.get("candidate_ids", []))) != TOP50:
            raise ScreenError("detached candidate IDs are not unique for " + sample_id)
        for method_id, selected_ids in selected.items():
            ranked_ids = [
                candidate_ids[row_index]
                for row_index, row in sorted(
                    enumerate(selection_rows),
                    key=lambda pair: (pair[1].get(method_id) is None, pair[1].get(method_id) or TOP50 + 1),
                )
                if row.get(method_id) is not None
            ]
            method_ranks = [row.get(method_id) for row in selection_rows if row.get(method_id) is not None]
            if sorted(method_ranks) != list(range(1, TOP5 + 1)) or ranked_ids != selected_ids:
                raise ScreenError("detached score ranks do not reproduce Top-5 for " + sample_id + "/" + method_id)


def read_score_matrix_from_document(score_trace: Mapping[str, Any], case_number: int) -> tuple[list[str], np.ndarray]:
    sample_id = "case-{:03d}".format(case_number)
    for sample in score_trace.get("samples", []):
        if sample.get("sample_id") == sample_id:
            identifiers = [str(value) for value in sample.get("candidate_ids", [])]
            matrix = np.asarray(sample.get("numeric_matrix"), dtype=np.float64)
            if matrix.shape != (TOP50, len(SCORE_TRACE_COLUMNS)) or len(identifiers) != TOP50:
                raise ScreenError("score trace matrix has invalid coverage or shape")
            return identifiers, matrix
    raise ScreenError("score trace document is missing " + sample_id)


def _report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    methods = [key for key, value in summary.items() if isinstance(value, Mapping)]
    lines = [
        "# DPR span/relevance quantum feature screen: fixed 100 x Top-50",
        "",
        "本轮只对已固定的 100 x 50 候选做离线重放。没有重新运行检索/DPR、Generator 或 evaluator；Silver oracle 与三模型证据只在全部 Top-5 产生后用于诊断。",
        "",
        "| 方法 | Silver overlap / 5 | 相对 Top-k Δ (95% paired bootstrap) | direct evidence / 5 | Jaccard 冗余 | 近重复对 / 题 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_id in methods:
        item = summary[method_id]
        delta = item["paired_delta_vs_answer_scorer_topk"]
        lines.append(
            "| {method} | {overlap:.3f} | {delta:+.3f} [{low:+.3f}, {high:+.3f}] | {direct:.3f} | {jac:.3f} | {dup:.3f} |".format(
                method=method_id,
                overlap=item["mean_silver_oracle_overlap_out_of_5"],
                delta=delta["mean_delta"],
                low=delta["paired_bootstrap_95_low"],
                high=delta["paired_bootstrap_95_high"],
                direct=item["mean_selected_direct_positive_count_out_of_5"],
                jac=item["mean_token_jaccard"],
                dup=item["mean_near_duplicate_pairs_per_top5"],
            )
        )
    lines.extend([
        "",
        "## 量子线路与解释边界",
        "",
        "- 线路：4 qubit，RY(arccos(x)) 角度编码，单层有向 CNOT 环，8 个局部/相邻 ZZ 观测量；0 个可训练参数，4 个 CNOT，抽象双比特深度 2。",
        "- `classical_matched_born` 逐行解析计算同一 Born 期望，必须与 statevector 线路完全一致；其结果用于实现/归因控制，不作为独立量子收益。",
        "- 因此本轮若胜过 DPR，只能说明 Reader span/relevance 交互信号值得继续，不构成量子优势证据；量子与经典匹配控制相同是预期结果。",
        "- Silver overlap 是 L0 选择诊断，不是 gold 准确率，也不报告生成 EM/F1。Top-50、K=5、Reader、Generator、evaluator 均未改变。",
        "- 继续门槛：span 机制的经典/量子匹配臂相对 Top-k 和 QORE 均改善且冗余不恶化；否则关闭这一 span-fusion 假设。只有后续可训练电路在预注册的独立标签/切分和匹配经典参数预算下胜出，才讨论量子机制收益。",
        "",
        "量子/经典 Born 最大绝对差：`{:.3e}`。".format(summary["quantum_classical_parity_max_abs"]),
        "",
    ])
    return "\n".join(lines)


def _run(input_path: Path, trace_path: Path, output_root: Path, exchange_url: str, token: str, upload: bool) -> Path:
    if not input_path.is_file() or _sha256(input_path) != DEFAULT_INPUT_SHA256:
        raise ScreenError("fixed 100-case input is missing or has a different hash")
    bundle = _load_input(input_path)
    trace = _load_trace(trace_path)
    raw_rows = trace["cases"]
    rows_by_case: dict[int, list[Mapping[str, Any]]] = {i: [] for i in range(1, CASE_COUNT + 1)}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise ScreenError("Reader trace row is not an object")
        case_number = int(row.get("case_number", 0))
        candidate_index = int(row.get("candidate_index", -1))
        if case_number not in rows_by_case or candidate_index < 0 or candidate_index >= TOP50:
            raise ScreenError("Reader trace contains an invalid case/index")
        rows_by_case[case_number].append(row)
    if any(len(rows_by_case[number]) != TOP50 for number in rows_by_case):
        raise ScreenError("Reader trace does not cover exactly 50 rows per question")
    for case_number in rows_by_case:
        rows_by_case[case_number].sort(key=lambda row: int(row["candidate_index"]))

    predictions = []
    for case, trace_rows in zip(bundle["cases"], (rows_by_case[index] for index in range(1, CASE_COUNT + 1))):
        online_candidates = _project_online_candidates(case, trace_rows)
        predictions.append(_case_predictions(case, online_candidates))

    # Post-hoc labels are first accessed here, after all 100 x all-method selections.
    case_results = [
        _posthoc_case_metrics(case, prediction)
        for case, prediction in zip(bundle["cases"], predictions)
    ]
    summary = _summarize(case_results, predictions)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root.resolve() / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = output_root.resolve() / (timestamp + "_" + str(suffix))
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = output_dir.name
    exchange_target = "five_ideas/dpr_reader_span_relevance_screen_100/" + run_id
    config_hash = _sha256(CONFIG_PATH)
    score_trace_samples = []
    for index, prediction in enumerate(predictions, 1):
        rows = prediction["score_trace_rows"]
        score_trace_samples.append({
            "sample_id": "case-{:03d}".format(index),
            "candidate_ids": [str(row["candidate_id"]) for row in rows],
            "retrieved_ranks": [int(row["retrieved_rank"]) for row in rows],
            "numeric_matrix": [list(row["numeric_values"]) for row in rows],
            "selection_ranks": [dict(row["selection_rank"]) for row in rows],
        })
    score_trace_row_count = sum(len(sample["candidate_ids"]) for sample in score_trace_samples)
    score_trace = {
        "schema_version": "rag.dpr_span_relevance_quantum_screen_100.score_trace.v1",
        "input_sha256": DEFAULT_INPUT_SHA256,
        "reader_trace_sha256": TRACE_SHA256,
        "online_labels_used": False,
        "columns": list(SCORE_TRACE_COLUMNS),
        "samples": score_trace_samples,
    }
    trace_temporary = tempfile.TemporaryDirectory(prefix="qore-span-score-trace-")
    score_trace_path = Path(trace_temporary.name) / "score_trace.json"
    score_trace_path.write_text(json.dumps(score_trace, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    score_trace_size = score_trace_path.stat().st_size
    score_trace_sha256 = _sha256(score_trace_path)
    score_trace_remote_path = exchange_target + "/score_trace.json"
    score_trace_storage = "github" if score_trace_size <= MAX_GITHUB_BYTES else "exchange"
    if score_trace_storage == "github":
        shutil.copyfile(score_trace_path, output_dir / "score_trace.json")
    elif not upload:
        trace_temporary.cleanup()
        raise ScreenError("score_trace.json exceeds 1 MiB and needs the authenticated exchange")
    score_trace_ref = {
        "path": "five_ideas/dpr_reader_span_relevance_screen_100/{}/score_trace.json".format(run_id)
        if score_trace_storage == "github" else score_trace_remote_path,
        "sha256": score_trace_sha256,
        "bytes": score_trace_size,
        "visibility": "portal" if score_trace_storage == "github" else "exchange_only",
    }
    sample_trace = _build_sample_trace(
        run_id,
        input_path,
        input_path.stat().st_size,
        score_trace_ref,
        bundle["cases"],
        predictions,
        case_results,
        config_hash,
    )
    sample_trace_path = Path(trace_temporary.name) / "sample_trace.json"
    sample_trace_path.write_text(json.dumps(sample_trace, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    sample_trace_size = sample_trace_path.stat().st_size
    sample_trace_sha256 = _sha256(sample_trace_path)
    validate_sample_trace_artifacts(
        sample_trace_path,
        score_trace_path,
        sample_trace_sha256,
        sample_trace_size,
        score_trace_sha256,
        score_trace_size,
        predictions,
        case_results,
    )
    sample_trace_storage = "github" if sample_trace_size <= MAX_GITHUB_BYTES else "exchange"
    sample_trace_remote_path = exchange_target + "/sample_trace.json"
    if sample_trace_storage == "github":
        shutil.copyfile(sample_trace_path, output_dir / "sample_trace.json")
    elif not upload:
        trace_temporary.cleanup()
        raise ScreenError("sample_trace.json exceeds 1 MiB and needs the authenticated exchange")
    result = {
        "schema_version": "rag.dpr_span_relevance_quantum_screen_100.result.v1",
        "artifact_type": "fixed_top50_to_top5_dpr_span_relevance_feature_screen",
        "evidence_tier": "L0_diagnostic",
        "input": {"path_name": input_path.name, "bytes": input_path.stat().st_size, "sha256": _sha256(input_path)},
        "reader_trace": {"exchange_path": TRACE_EXCHANGE_PATH, "bytes": TRACE_BYTES, "sha256": TRACE_SHA256},
        "protocol": {
            "case_count": CASE_COUNT,
            "top50_count": TOP50,
            "top5_count": TOP5,
            "online_feature_allowlist": ["relevance_logit", "span_logit", "span_margin", "start_entropy", "end_entropy", "candidate title/text for redundancy only"],
            "silver_or_gold_used_for_scoring": False,
            "gold_answers_used": False,
            "retriever_called": False,
            "dpr_reader_called": False,
            "selector_core_changed": False,
            "generator_called": False,
            "evaluator_called": False,
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "paired_bootstrap_seed": BOOTSTRAP_SEED,
            "redundancy_penalty": REDUNDANCY_PENALTY,
        },
        "circuit": predictions[0]["circuit"],
        "summary": summary,
        "score_trace": {
            "schema_version": score_trace["schema_version"],
            "rows": score_trace_row_count,
            "bytes": score_trace_size,
            "sha256": score_trace_sha256,
            "storage": score_trace_storage,
            "path": score_trace_ref["path"],
        },
        "sample_trace": {
            "schema_version": sample_trace["schema_version"],
            "trace_id": sample_trace["trace_id"],
            "completeness": sample_trace["trace_completeness"],
            "sample_count": len(sample_trace["samples"]),
            "bytes": sample_trace_size,
            "sha256": sample_trace_sha256,
            "storage": sample_trace_storage,
            "path": "sample_trace.json" if sample_trace_storage == "github" else sample_trace_remote_path,
        },
        "cases": [
            {"case_number": index + 1, "methods": case_results[index]}
            for index in range(CASE_COUNT)
        ],
        "provenance": {
            "script": str(SCRIPT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "script_sha256": _sha256(SCRIPT_PATH),
            "config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": config_hash,
            "git_revision": _git_revision(),
        },
        "next_step": "Read this Silver diagnostic with the cumulative RAG record; continue only if fixed span features beat both stored baselines without redundancy regression. This fixed circuit cannot support a quantum-advantage claim.",
    }
    result_path = output_dir / "result.json"
    report_path = output_dir / "report.md"
    metadata_path = output_dir / "run_metadata.json"
    manifest_path = output_dir / "upload_manifest.json"
    _write_json(result_path, result)
    report_path.write_text(_report(result), encoding="utf-8")
    metadata = {
        "schema_version": "rag.dpr_span_relevance_quantum_screen_100.run_metadata.v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "input_sha256": DEFAULT_INPUT_SHA256,
        "reader_trace_sha256": TRACE_SHA256,
        "code_sha256": result["provenance"]["script_sha256"],
        "config_sha256": config_hash,
        "exchange_target": exchange_target,
        "output_files": [
            "result.json", "report.md", "run_metadata.json", "upload_manifest.json",
            "score_trace.json" if score_trace_storage == "github" else "18083:score_trace.json",
            "sample_trace.json" if sample_trace_storage == "github" else "18083:sample_trace.json",
        ],
    }
    _write_json(metadata_path, metadata)
    if result_path.stat().st_size > MAX_GITHUB_BYTES:
        raise ScreenError("compact result.json exceeds the 1 MiB GitHub threshold")
    if report_path.stat().st_size > MAX_GITHUB_BYTES:
        raise ScreenError("compact report.md exceeds the 1 MiB GitHub threshold")
    files = [result_path, report_path, metadata_path]
    if score_trace_storage == "github":
        files.append(output_dir / "score_trace.json")
    if sample_trace_storage == "github":
        files.append(output_dir / "sample_trace.json")
    exchange_files = []
    if score_trace_storage == "exchange":
        exchange_files.append({"name": "score_trace.json", "bytes": score_trace_size, "sha256": score_trace_sha256})
    if sample_trace_storage == "exchange":
        exchange_files.append({"name": "sample_trace.json", "bytes": sample_trace_size, "sha256": sample_trace_sha256})
    manifest = {
        "schema_version": "rag.dpr_span_relevance_quantum_screen_100.upload_manifest.v1",
        "artifact_type": "dpr_span_relevance_quantum_screen_100_upload_manifest",
        "status": "ready_for_github_and_authenticated_exchange",
        "target_directory": exchange_target,
        "compact_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files
        ],
        "exchange_files": exchange_files,
        "privacy": {
            "raw_question_or_passage_text_in_result": False,
            "gold_answers_in_result": False,
            "silver_or_evidence_labels_used_online": False,
            "full_reader_trace_copied_to_output": False,
        },
        "provenance": result["provenance"],
    }
    _write_json(manifest_path, manifest)
    if upload:
        if not token:
            trace_temporary.cleanup()
            raise ScreenError("authenticated exchange needs QORE_EXCHANGE_TOKEN")
        try:
            _create_exchange_directory(exchange_url, token, exchange_target)
            receipts = []
            if score_trace_storage == "exchange":
                receipts.append(_upload_file(exchange_url, token, score_trace_path, score_trace_remote_path))
            if sample_trace_storage == "exchange":
                receipts.append(_upload_file(exchange_url, token, sample_trace_path, sample_trace_remote_path))
            receipts.append(_upload_file(exchange_url, token, manifest_path, exchange_target + "/upload_manifest.json"))
            _write_json(output_dir / "upload_receipt.json", {"receipts": receipts})
        finally:
            trace_temporary.cleanup()
    else:
        trace_temporary.cleanup()
    return output_dir


def _git_revision() -> str | None:
    import subprocess
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "five_ideas/dpr_reader_span_relevance_screen_100")
    parser.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", EXCHANGE_URL))
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--no-exchange", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    temporary = None
    try:
        token = os.environ.get(args.token_env, "")
        input_path = args.input or DEFAULT_INPUT
        if not input_path.is_file():
            if args.input is not None:
                raise ScreenError("explicit fixed-input path does not exist")
            if not token:
                raise ScreenError("fixed detail JSON is absent from checkout; set " + args.token_env + " for its automatic 18083 download")
            temporary = tempfile.TemporaryDirectory(prefix="qore-dpr-span-screen-")
            input_path = Path(temporary.name) / Path(INPUT_EXCHANGE_PATH).name
            _download_input(args.exchange_url, token, input_path)
        if input_path.is_symlink() or _sha256(input_path) != DEFAULT_INPUT_SHA256:
            raise ScreenError("fixed 100-case input has a different registered hash or is a symlink")
        trace_path = args.trace
        if trace_path is None:
            local_trace = ROOT / "exchange" / TRACE_EXCHANGE_PATH
            if local_trace.is_file():
                trace_path = local_trace
            else:
                if not token:
                    raise ScreenError("Reader trace is absent locally; set " + args.token_env + " for the automatic 18083 download")
                if temporary is None:
                    temporary = tempfile.TemporaryDirectory(prefix="qore-dpr-span-screen-")
                trace_path = Path(temporary.name) / "candidate_trace.json"
                _download_trace(args.exchange_url, token, trace_path)
        bundle = _load_input(input_path)
        trace = _load_trace(trace_path)
        if args.validate_only:
            print(json.dumps({
                "input_sha256": DEFAULT_INPUT_SHA256,
                "trace_sha256": TRACE_SHA256,
                "case_count": len(bundle["cases"]),
                "candidate_count": len(trace["cases"]),
                "input_downloaded": str(input_path) != str(DEFAULT_INPUT),
                "trace_downloaded": str(trace_path) != str(ROOT / "exchange" / TRACE_EXCHANGE_PATH),
                "model_calls": 0,
                "label_use_online": False,
                "valid": True,
            }, ensure_ascii=False))
            return 0
        output_dir = _run(input_path, trace_path, args.output_root, args.exchange_url, token, not args.no_exchange)
        print(json.dumps({"output_dir": str(output_dir), "result": str(output_dir / "result.json"), "exchange_created": not args.no_exchange}, ensure_ascii=False))
        return 0
    except (ScreenError, SpanFusionError, OSError, ValueError, KeyError, TypeError) as exc:
        print("DPR span/relevance screen failed: " + str(exc), file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
