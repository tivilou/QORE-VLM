#!/usr/bin/env python3
"""Replay retained historical selectors on the registered 100-case Top-50.

The runner rehydrates the exact DPR Top-50 locally, validates identity against
the registered case-study artifact, then runs the retained MMR, Submodular,
Spectral-DPP and explicit QORE-enhancer arms. Evidence labels are never passed
to a selector; they are read only after selection for L0 diagnostic metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
DEFAULT_INPUT = ROOT / "research-web/apps/experiment-results/case-studies/silver-oracle-top5-100-20260915T120525Z-detail.json"
DEFAULT_CONFIG = ROOT / "configs/experiments/selector_replay_100_historical.yaml"
DEFAULT_PLAN = ROOT / "configs/experiments/selector_replay_100_historical_plan.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exchange/five_ideas/selector_replay_100_historical"
EXPECTED_CASES = 100
EXPECTED_TOP50 = 50
K = 5
SEED = 42
SCORE_ABS_TOLERANCE = 1.0e-3


class ReplayError(RuntimeError):
    """Raised when the replay contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=False, capture_output=True,
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"JSON root must be an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        raise ReplayError(f"cannot load YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"YAML root must be an object: {path}")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayError(f"{field} must be finite")
    return result


def _validate_config(config_path: Path, plan_path: Path) -> dict[str, Any]:
    document = _load_yaml(config_path)
    phase = document.get("phase")
    if not isinstance(phase, Mapping):
        raise ReplayError("phase section is missing")
    expected = {
        "name": "selector_replay_100_historical",
        "schema_version": 1,
        "evidence_tier": "L0_diagnostic",
        "diagnostic_only": True,
        "selection_mutation": False,
    }
    for key, value in expected.items():
        if phase.get(key) != value:
            raise ReplayError(f"replay config mismatch for {key}")
    input_spec = phase.get("input")
    if not isinstance(input_spec, Mapping) or int(input_spec.get("case_count", -1)) != EXPECTED_CASES or int(input_spec.get("top50_count", -1)) != EXPECTED_TOP50:
        raise ReplayError("100-case input contract is not frozen")
    retrieval = phase.get("retrieval")
    if retrieval != {
        "corpus_mode": "wiki_dpr",
        "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64,
        "top_k": 50,
        "exact_identity_required": True,
    }:
        raise ReplayError("retrieval contract is not frozen")
    selectors = phase.get("selectors")
    if not isinstance(selectors, list) or len(selectors) < 5:
        raise ReplayError("selector allowlist is incomplete")
    ids = [str(item.get("id")) for item in selectors if isinstance(item, Mapping)]
    required = ["qore_as", "topk_as", "mmr_as", "submodular_as", "spectral_dpp_as"]
    if ids[:5] != required:
        raise ReplayError("baseline selector order changed")
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v1" or plan.get("authorization") != "implemented":
        raise ReplayError("plugin plan is not implemented")
    if plan.get("reproducibility", {}).get("silver_labels_used_online") is not False:
        raise ReplayError("plan permits Silver leakage")
    return dict(phase)


def _stored_ids(case: Mapping[str, Any], selector_id: str) -> list[str]:
    selectors = case.get("selectors")
    if not isinstance(selectors, list):
        raise ReplayError("case selectors must be a list")
    for selector in selectors:
        if not isinstance(selector, Mapping) or selector.get("selector_id") != selector_id:
            continue
        selected = selector.get("selected_top_5")
        if not isinstance(selected, list) or len(selected) != K:
            raise ReplayError(f"stored selector {selector_id} must contain five passages")
        ids = [str(item.get("id", "")) for item in selected if isinstance(item, Mapping)]
        if len(ids) != K or len(set(ids)) != K or any(not item for item in ids):
            raise ReplayError(f"stored selector {selector_id} has invalid IDs")
        return ids
    raise ReplayError(f"stored selector {selector_id} is missing")


def _online_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    identifier = candidate.get("id")
    title = candidate.get("title")
    text = candidate.get("text")
    rank = candidate.get("retrieved_rank")
    if not isinstance(identifier, (str, int)) or not str(identifier).strip():
        raise ReplayError("candidate id is invalid")
    if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
        raise ReplayError("candidate title/text is invalid")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ReplayError("candidate rank is invalid")
    return {
        "id": str(identifier),
        "title": title,
        "text": text,
        "retrieved_rank": rank,
        "retrieval_score": _finite(candidate.get("retrieval_score"), "retrieval_score"),
        "answer_scorer_score": _finite(candidate.get("answer_scorer_score"), "answer_scorer_score"),
    }


def _validate_input(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = bundle.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ReplayError(f"expected exactly {EXPECTED_CASES} cases")
    for number, case in enumerate(cases, start=1):
        if not isinstance(case, Mapping) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ReplayError(f"case {number} has no question")
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise ReplayError(f"case {number} must contain exactly {EXPECTED_TOP50} candidates")
        ids = [_online_candidate(item)["id"] for item in top50 if isinstance(item, Mapping)]
        if len(ids) != EXPECTED_TOP50 or len(set(ids)) != EXPECTED_TOP50:
            raise ReplayError(f"case {number} candidate IDs are not unique")
        _stored_ids(case, "qore_common_order")
        _stored_ids(case, "topk_common_order")
        _stored_ids(case, "silver_oracle_common_order")
    return cases


def _evidence_flags(candidate: Mapping[str, Any]) -> tuple[bool, bool]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, Mapping):
        return False, False
    broad = evidence.get("positive_consensus")
    direct = evidence.get("direct_consensus")
    if not isinstance(broad, bool):
        broad = evidence.get("consensus_label") in {"direct", "partial"}
    if not isinstance(direct, bool):
        direct = evidence.get("consensus_label") == "direct"
    return bool(broad), bool(direct)


def _token_set(value: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value.lower()))


def _redundancy(selected: Sequence[Mapping[str, Any]], embeddings: Sequence[Sequence[float]]) -> dict[str, Any]:
    token_sets = [_token_set(f"{item['title']}. {item['text']}") for item in selected]
    pair_count = 0
    token_sum = 0.0
    cosine_sum = 0.0
    title_pairs = 0
    same_title = 0
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            pair_count += 1
            union = token_sets[left] | token_sets[right]
            token_sum += len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0
            cosine_sum += sum(float(a) * float(b) for a, b in zip(embeddings[left], embeddings[right]))
            title_pairs += 1
            same_title += int(selected[left]["title"] == selected[right]["title"])
    return {
        "token_jaccard_mean": token_sum / pair_count if pair_count else None,
        "embedding_cosine_mean": cosine_sum / pair_count if pair_count else None,
        "same_title_pair_fraction": same_title / title_pairs if title_pairs else None,
        "unique_title_count": len({str(item["title"]) for item in selected}),
    }


def _build_row(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    oracle_ids: set[str],
    embeddings_by_id: Mapping[str, Sequence[float]],
    *,
    elapsed_ms: float,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = {str(item["id"]): item for item in candidates}
    selected = [rows[identifier] for identifier in selected_ids]
    broad = {str(item["id"]): _evidence_flags(item)[0] for item in candidates}
    direct = {str(item["id"]): _evidence_flags(item)[1] for item in candidates}
    selected_vectors = [embeddings_by_id[identifier] for identifier in selected_ids]
    overlap = len(set(selected_ids) & oracle_ids)
    oracle_size = len(oracle_ids)
    precision = overlap / K
    recall = overlap / oracle_size if oracle_size else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0 if oracle_size else None
    canonical = "\n".join(sorted(selected_ids)).encode("utf-8")
    broad_count = sum(int(broad.get(identifier, False)) for identifier in selected_ids)
    return {
        "selected_set_sha256": hashlib.sha256(canonical).hexdigest(),
        "selected_ranks": [int(item["retrieved_rank"]) for item in selected],
        "selected_answer_scorer_scores": [round(float(item["answer_scorer_score"]), 8) for item in selected],
        "selected_broad_positive_count": broad_count,
        "selected_direct_positive_count": sum(int(direct.get(identifier, False)) for identifier in selected_ids),
        "top50_broad_positive_count": sum(broad.values()),
        "top50_direct_positive_count": sum(direct.values()),
        "retrieval_miss": not any(broad.values()),
        "selector_miss": bool(broad) and any(broad.values()) and broad_count == 0,
        "silver_oracle_overlap_count": overlap,
        "silver_oracle_precision": precision,
        "silver_oracle_recall": recall,
        "silver_oracle_set_f1": f1,
        "redundancy": _redundancy(selected, selected_vectors),
        "selection_time_ms": elapsed_ms,
        "selector_diagnostics": dict(diagnostics or {}),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for part in field.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _summary(rows: Sequence[Mapping[str, Any]], method: str, status: str) -> dict[str, Any]:
    overlap = [int(row["silver_oracle_overlap_count"]) for row in rows]
    return {
        "method": method,
        "status": status,
        "case_count": len(rows),
        "mean_silver_oracle_overlap": _mean(rows, "silver_oracle_overlap_count"),
        "median_silver_oracle_overlap": statistics.median(overlap) if overlap else None,
        "mean_silver_oracle_precision": _mean(rows, "silver_oracle_precision"),
        "mean_silver_oracle_recall": _mean(rows, "silver_oracle_recall"),
        "mean_silver_oracle_set_f1": _mean(rows, "silver_oracle_set_f1"),
        "mean_selected_broad_positive_count": _mean(rows, "selected_broad_positive_count"),
        "mean_selected_direct_positive_count": _mean(rows, "selected_direct_positive_count"),
        "retrieval_miss_cases": sum(bool(row["retrieval_miss"]) for row in rows),
        "selector_miss_cases": sum(bool(row["selector_miss"]) for row in rows),
        "mean_token_jaccard_redundancy": _mean(rows, "redundancy.token_jaccard_mean"),
        "mean_embedding_cosine_redundancy": _mean(rows, "redundancy.embedding_cosine_mean"),
        "mean_same_title_pair_fraction": _mean(rows, "redundancy.same_title_pair_fraction"),
        "mean_unique_title_count": _mean(rows, "redundancy.unique_title_count"),
        "mean_selection_time_ms": _mean(rows, "selection_time_ms"),
        "median_selection_time_ms": statistics.median([float(row["selection_time_ms"]) for row in rows]) if rows else None,
        "overlap_distribution": dict(sorted(Counter(overlap).items())),
    }


def _paired(rows: Mapping[str, Sequence[Mapping[str, Any]]], left: str, right: str) -> dict[str, int]:
    left_map = {int(row["case_number"]): row for row in rows[left]}
    right_map = {int(row["case_number"]): row for row in rows[right]}
    values = [(left_map[number]["silver_oracle_overlap_count"], right_map[number]["silver_oracle_overlap_count"]) for number in sorted(set(left_map) & set(right_map))]
    return {
        "left_wins": sum(int(left_value > right_value) for left_value, right_value in values),
        "ties": sum(int(left_value == right_value) for left_value, right_value in values),
        "right_wins": sum(int(left_value < right_value) for left_value, right_value in values),
    }


def _retrieve(manager: Any, query_embedding: Any) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np
    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise ReplayError("Wiki-DPR manager does not expose its indexed dataset")
    scores, retrieved = dataset.get_nearest_examples("embeddings", np.asarray(query_embedding, dtype=np.float32), k=EXPECTED_TOP50)
    required = ("id", "title", "text", "embeddings")
    if any(key not in retrieved for key in required) or len(scores) != EXPECTED_TOP50:
        raise ReplayError("Wiki-DPR retrieval result lacks the exact Top-50 fields")
    records = []
    for index in range(EXPECTED_TOP50):
        records.append({
            "id": str(retrieved["id"][index]),
            "title": str(retrieved["title"][index] or ""),
            "text": str(retrieved["text"][index] or ""),
            "retrieved_rank": index + 1,
            "retrieval_score": float(scores[index]),
        })
    return records, np.asarray(retrieved["embeddings"], dtype=np.float32)


def _validate_retrieval(case: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [_online_candidate(item) for item in case["top_50"]]
    if len(records) != len(expected):
        raise ReplayError("rehydrated Top-50 length mismatch")
    max_score_error = 0.0
    text_mismatches = 0
    id_mismatches = 0
    for stored, observed in zip(expected, records):
        id_mismatches += int(stored["id"] != observed["id"])
        text_mismatches += int(stored["title"] != observed["title"] or stored["text"] != observed["text"])
        max_score_error = max(max_score_error, abs(stored["retrieval_score"] - float(observed["retrieval_score"])))
    if id_mismatches or text_mismatches or max_score_error > SCORE_ABS_TOLERANCE:
        raise ReplayError(
            "rehydrated Top-50 differs from registered artifact: "
            f"id_mismatches={id_mismatches}, text_mismatches={text_mismatches}, "
            f"max_score_error={max_score_error:.6g}"
        )
    return {
        "ids_match": True,
        "texts_match": True,
        "max_retrieval_score_abs_error": max_score_error,
        "candidate_count": len(records),
    }


def _select(spec: Mapping[str, Any], query: Any, embeddings: Any, candidates: Sequence[Mapping[str, Any]], scorer: Any | None) -> tuple[list[int], dict[str, Any]]:
    import numpy as np
    from applications.rag.selector import select_passages
    scores = np.asarray([float(item["answer_scorer_score"]) for item in candidates], dtype=np.float64)
    passages = [f"{item['title']}. {item['text']}" if item["title"] else str(item["text"]) for item in candidates]
    kwargs: dict[str, Any] = {
        "query_embedding": query,
        "passage_embeddings": embeddings,
        "K": K,
        "method": str(spec["method"]),
        "relevance_scores": scores,
        "seed": SEED,
    }
    method = str(spec["method"])
    if method == "mmr":
        kwargs["lambda_mmr"] = float(spec["lambda_mmr"])
    elif method == "submodular":
        kwargs["saturation_alpha"] = float(spec["saturation_alpha"])
        kwargs["lambda_submodular"] = float(spec["lambda_submodular"])
    elif method == "spectral_dpp":
        kwargs["dpp_quality_scale"] = float(spec["dpp_quality_scale"])
        kwargs["dpp_jitter"] = float(spec["dpp_jitter"])
    elif method == "qore":
        kwargs.update({
            "num_reads": int(spec["num_reads"]),
            "lam": float(spec["lam"]),
            "seed": int(spec["seed"]),
            "direct_solve_max_n": int(spec["direct_solve_max_n"]),
            "enhancers": list(spec["enhancers"]),
            "enhancer_configs": dict(spec["enhancer_configs"]),
            "answer_scorer": scorer,
            "passage_texts": passages,
            "question": str(spec["question"]),
        })
    else:
        raise ReplayError(f"unsupported replay method: {method}")
    selected = [int(value) for value in np.asarray(select_passages(**kwargs)).reshape(-1)]
    if len(selected) != K or len(set(selected)) != K or any(value < 0 or value >= len(candidates) for value in selected):
        raise ReplayError(f"selector {spec['id']} returned an invalid selection")
    return selected, {"method": method}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 100题历史 selector Top-50 -> Top-5 replay",
        "",
        "这是固定 Top-50 上的 L0 selector-only replay。选择阶段只使用问题、候选身份/文本、DPR embedding、检索分数和 Answer Scorer 分数；Silver evidence 只在选择完成后用于诊断。没有调用 Generator，也不能单独升级为 L1/L2。",
        "",
        f"- 输入：`{report['input']['path_name']}`，SHA-256 `{report['input']['sha256']}`",
        f"- 规模：{report['protocol']['case_count']} 题 × {report['protocol']['top50_count']} 候选，K={report['protocol']['k']}",
        f"- 重放配置：`{report['protocol']['replay_profile']}`",
        f"- 代码 revision：`{report['provenance'].get('git_revision') or 'unavailable'}`",
        "",
        "## 结果",
        "",
        "| 方法 | 状态 | Silver overlap 均值 | set-F1 均值 | broad 选中均值 | selector miss | embedding 冗余 | 中位耗时 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, summary in report["summaries"].items():
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {method} | {summary['status']} | {fmt(summary['mean_silver_oracle_overlap'])} | "
            f"{fmt(summary['mean_silver_oracle_set_f1'])} | {fmt(summary['mean_selected_broad_positive_count'])} | "
            f"{summary['selector_miss_cases']} | {fmt(summary['mean_embedding_cosine_redundancy'])} | "
            f"{fmt(summary['median_selection_time_ms'])} |"
        )
    lines.extend(["", "## 相对基线的逐题胜负", "", "| 方法 | 相对 QORE：胜/平/负 | 相对 Top-k：胜/平/负 |", "|---|---:|---:|"])
    for method, values in report["paired_comparisons"].items():
        qore = values["vs_qore"]
        topk = values["vs_topk"]
        lines.append(f"| {method} | {qore['left_wins']}/{qore['ties']}/{qore['right_wins']} | {topk['left_wins']}/{topk['ties']}/{topk['right_wins']} |")
    lines.extend(["", "## 输入与边界", "", f"- 重新取得的 Wiki-DPR Top-50 通过逐题 ID、title、正文和检索分数校验；最大分数绝对误差：`{report['diagnostics']['max_retrieval_score_abs_error']:.6g}`。", f"- retrieval miss：{report['diagnostics']['retrieval_miss_cases']}/{report['protocol']['case_count']}；这些题不归因于 selector。", "- QORE/Mobius/Cohesion 是明确命名的 QUBO enhancer arms；原始 QORE 本身已经是 QUBO selector。", "- Differentiable-QUBO/Idea 7 未纳入：checkpoint/training provenance 不在本 replay contract 内，且历史 real-data 结果为负。", "- 完整 selector trace 只通过 18083 exchange 保存；GitHub 只保存 compact 文件。", ""])
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unique_output_dir(root: Path, timestamp: str) -> Path:
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run(args: argparse.Namespace) -> Path:
    config_path = args.config.resolve()
    plan_path = args.plan.resolve()
    phase = _validate_config(config_path, plan_path)
    input_path = args.input.resolve()
    bundle = _load_json(input_path)
    cases = _validate_input(bundle)
    if args.skip_qubo_enhancers:
        selectors = [item for item in phase["selectors"] if item.get("kind") != "qore_enhancer"]
    else:
        selectors = list(phase["selectors"])

    import numpy as np
    from applications.rag.data import make_corpus_manager
    from applications.rag.retrieval import make_encoder
    encoder = make_encoder("dpr")
    manager = make_corpus_manager("wiki_dpr", {
        "wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"],
        "nprobe": int(phase["retrieval"]["nprobe"]),
    })
    manager.build([])
    scorer = None
    if any(item.get("kind") == "qore_enhancer" and "mobius" in item.get("enhancers", []) for item in selectors):
        from applications.rag.answer_scorer import make_answer_scorer
        scorer = make_answer_scorer(backend="dpr")

    method_rows: dict[str, list[dict[str, Any]]] = {}
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    retrieval_miss_cases = 0
    max_score_error = 0.0
    for case_number, case in enumerate(cases, start=1):
        question = str(case["question"])
        query_embedding = encoder.encode_queries([question])[0]
        records, embeddings = _retrieve(manager, query_embedding)
        validation = _validate_retrieval(case, records)
        max_score_error = max(max_score_error, float(validation["max_retrieval_score_abs_error"]))
        candidates = [_online_candidate(item) for item in case["top_50"]]
        embeddings_by_id = {str(record["id"]): [float(value) for value in embeddings[index]] for index, record in enumerate(records)}
        oracle_ids = set(_stored_ids(case, "silver_oracle_common_order"))
        if not any(_evidence_flags(item)[0] for item in candidates):
            retrieval_miss_cases += 1
        case_methods: dict[str, dict[str, Any]] = {}
        trace_methods: dict[str, Any] = {}

        baseline_specs = [
            ("qore_as", "qore_common_order"),
            ("topk_as", "topk_common_order"),
        ]
        for method_id, source_id in baseline_specs:
            selected_ids = _stored_ids(case, source_id)
            row = _build_row(case, candidates, selected_ids, oracle_ids, embeddings_by_id, elapsed_ms=0.0, diagnostics={"source": source_id})
            row["case_number"] = case_number
            method_rows.setdefault(method_id, []).append(row)
            case_methods[method_id] = row
            trace_methods[method_id] = {"selected_ids": selected_ids, "selected_ranks": row["selected_ranks"]}

        for selector in selectors:
            method_id = str(selector["id"])
            if method_id in {"qore_as", "topk_as"}:
                continue
            spec = dict(selector)
            spec["question"] = question
            started_one = time.perf_counter()
            indices, diagnostics = _select(spec, query_embedding, embeddings, candidates, scorer)
            elapsed_ms = (time.perf_counter() - started_one) * 1000.0
            selected_ids = [str(records[index]["id"]) for index in indices]
            row = _build_row(case, candidates, selected_ids, oracle_ids, embeddings_by_id, elapsed_ms=elapsed_ms, diagnostics=diagnostics)
            row["case_number"] = case_number
            method_rows.setdefault(method_id, []).append(row)
            case_methods[method_id] = row
            trace_methods[method_id] = {"selected_ids": selected_ids, "selected_ranks": row["selected_ranks"]}

        traces.append({
            "case_number": case_number,
            "source_id": str(case.get("source_id") or case.get("question_id") or ""),
            "question_id": str(case.get("question_id") or ""),
            "top50_candidate_id_sha256": hashlib.sha256("\n".join(str(item["id"]) for item in candidates).encode("utf-8")).hexdigest(),
            "retrieval_validation": validation,
            "methods": trace_methods,
        })
        if args.progress and (case_number == EXPECTED_CASES or case_number % 10 == 0):
            print(f"  historical selector replay: {case_number}/{EXPECTED_CASES}", flush=True)

    baseline_methods = {"qore_as", "topk_as"}
    summaries = {
        method: _summary(rows, method, "registered_baseline" if method in baseline_methods else "replayed")
        for method, rows in method_rows.items()
    }
    paired = {
        method: {"vs_qore": _paired(method_rows, method, "qore_as"), "vs_topk": _paired(method_rows, method, "topk_as")}
        for method in method_rows if method not in baseline_methods
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root.resolve() if args.output_root else DEFAULT_OUTPUT_ROOT
    output_dir = _unique_output_dir(output_root, timestamp)
    target_directory = f"five_ideas/selector_replay_100_historical/{output_dir.name}"
    report: dict[str, Any] = {
        "schema_version": "rag.selector_replay_100_historical.v1",
        "artifact_type": "selector_only_historical_replay",
        "diagnostic_only": True,
        "input": {"path_name": input_path.name, "sha256": _sha256(input_path)},
        "protocol": {
            "case_count": EXPECTED_CASES,
            "top50_count": EXPECTED_TOP50,
            "k": K,
            "replay_profile": phase["replay_profile"],
            "online_fields": phase["online_fields"],
            "silver_oracle_used_for_selection": False,
            "generator_called": False,
            "retrieval": dict(phase["retrieval"]),
            "selector_allowlist": [str(item["id"]) for item in selectors],
        },
        "provenance": {
            "git_revision": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "script": str(SCRIPT_PATH.relative_to(ROOT)),
            "script_sha256": _sha256(SCRIPT_PATH),
            "config_path": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
            "config_sha256": _sha256(config_path),
            "plan_path": str(plan_path.relative_to(ROOT)) if plan_path.is_relative_to(ROOT) else str(plan_path),
            "plan_sha256": _sha256(plan_path),
        },
        "summaries": summaries,
        "paired_comparisons": paired,
        "diagnostics": {
            "retrieval_miss_cases": retrieval_miss_cases,
            "max_retrieval_score_abs_error": max_score_error,
            "exact_top50_identity_validated": True,
            "case_count_validated": EXPECTED_CASES,
        },
        "blocked_methods": list(phase.get("blocked", [])),
        "next_step": "Only a candidate that beats Top-k on the preregistered paired selector gate can be considered for a later frozen-Generator screen; this replay alone remains L0 diagnostic evidence.",
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    _write_json(output_dir / "selector_trace.json", {
        "schema_version": "rag.selector_replay_100_historical.trace.v1",
        "diagnostic_only": True,
        "input": report["input"],
        "provenance": report["provenance"],
        "target_directory": target_directory,
        "cases": traces,
    })
    metadata = {
        "schema_version": "rag.selector_replay_100_historical.metadata.v1",
        "artifact_type": "selector_replay_100_historical_run_metadata",
        "status": "completed",
        "diagnostic_only": True,
        "selection_mutation": False,
        "generated_at_utc": timestamp,
        "input": report["input"],
        "provenance": report["provenance"],
        "protocol": report["protocol"],
        "outputs": {"root": str(output_root.relative_to(ROOT)) if output_root.is_relative_to(ROOT) else str(output_root), "target_directory": target_directory, "exchange_files": ["selector_trace.json"]},
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "validation": report["diagnostics"],
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    manifest = {
        "schema_version": "rag.selector_replay_100_historical.upload_manifest.v1",
        "artifact_type": "selector_replay_100_historical_upload_manifest",
        "status": "ready_for_authenticated_exchange_upload",
        "target_directory": target_directory,
        "generated_at_utc": timestamp,
        "exchange_files": [{"name": "selector_trace.json", "exchange_path": f"{target_directory}/selector_trace.json"}],
        "compact_files": [
            {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": _sha256(output_dir / name)}
            for name in ("summary.json", "report.md", "run_metadata.json")
        ],
        "privacy": {"raw_passage_text_in_compact": False, "gold_answers_in_compact": False, "silver_labels_used_online": False, "generator_outputs": False},
        "github_policy": "Commit compact files only when they pass the 1 MiB/privacy check; selector_trace.json is exchange-only.",
    }
    _write_json(output_dir / "upload_manifest.json", manifest)
    if not args.no_upload:
        from scripts.collab.lib.exchange_upload import upload_manifest
        receipts = upload_manifest(output_dir / "upload_manifest.json")
        metadata["exchange_upload"] = {"status": "uploaded", "files": receipts}
        _write_json(output_dir / "run_metadata.json", metadata)
        manifest["compact_files"] = [
            {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": _sha256(output_dir / name)}
            for name in ("summary.json", "report.md", "run_metadata.json")
        ]
        _write_json(output_dir / "upload_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "target_directory": target_directory, "uploaded": not args.no_upload}, ensure_ascii=False))
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--skip-qubo-enhancers", action="store_true", help="Replay only stored/QORE-independent selector arms")
    parser.add_argument("--no-upload", action="store_true", help="Keep local artifacts without the automatic 18083 upload")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and registered input without loading models")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    for field in ("input", "config", "plan"):
        path = getattr(args, field)
        if not path.is_absolute():
            setattr(args, field, ROOT / path)
    try:
        if args.validate_only:
            phase = _validate_config(args.config.resolve(), args.plan.resolve())
            bundle = _load_json(args.input.resolve())
            cases = _validate_input(bundle)
            print(json.dumps({"status": "valid", "phase": phase["name"], "case_count": len(cases), "top50_count": EXPECTED_TOP50}, ensure_ascii=False))
            return 0
        run(args)
    except (ReplayError, ValueError, OSError) as exc:
        print(f"historical replay error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
