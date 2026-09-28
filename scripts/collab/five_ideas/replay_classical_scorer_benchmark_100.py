#!/usr/bin/env python3
"""Benchmark classical non-generative answer scorers on the fixed 100 x 50 set.

The runner is selector-only.  It never retrieves, generates, evaluates, or
uses Silver/gold fields while scoring or selecting.  For each allowlisted
scorer it records direct score Top-5 and, when the existing QORE runtime is
available locally, the same scores passed to the current QUBO selector.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.rag.scorer_benchmark import (  # noqa: E402
    DEFAULT_SCORER_SPECS,
    ScorerBenchmarkError,
    ScorerSpec,
    build_scorer,
    rank_indices,
    score_summary,
)


DEFAULT_INPUT = ROOT / "research-web/apps/experiment-results/case-studies/silver-oracle-top5-100-20260915T120525Z-detail.json"
DEFAULT_EXCHANGE_INPUT_PATH = (
    "five_ideas/selector_replay_100_historical_input/"
    "silver-oracle-top5-100-20260915T120525Z-detail.json"
)
DEFAULT_INPUT_BYTES = 8046318
DEFAULT_INPUT_SHA256 = "669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731"
DEFAULT_OUTPUT_ROOT = ROOT / "exchange/five_ideas/classical_scorer_benchmark_100"
DEFAULT_EXCHANGE_TARGET = "five_ideas/classical_scorer_benchmark_100"
EXPECTED_CASES = 100
EXPECTED_TOP50 = 50
K = 5
MAX_GITHUB_BYTES = 1_048_576


class BenchmarkError(RuntimeError):
    """Raised when the fixed benchmark contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("detail JSON root must be an object")
    return value


def _load_specs(config_path: Path | None, selected: str | None) -> list[ScorerSpec]:
    specs = {spec.scorer_id: spec for spec in DEFAULT_SCORER_SPECS}
    if config_path is not None:
        config = _load_json(config_path.resolve())
        raw_specs = config.get("scorers")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise BenchmarkError("config scorers must be a non-empty list")
        specs = {}
        for raw in raw_specs:
            if not isinstance(raw, Mapping):
                raise BenchmarkError("config scorer entries must be objects")
            required = ("scorer_id", "backend", "model_id")
            if any(not isinstance(raw.get(field), str) or not raw[field].strip() for field in required):
                raise BenchmarkError("config scorer is missing scorer_id/backend/model_id")
            specs[str(raw["scorer_id"])] = ScorerSpec(
                scorer_id=str(raw["scorer_id"]),
                backend=str(raw["backend"]),
                model_id=str(raw["model_id"]),
                revision=str(raw["revision"]) if raw.get("revision") else None,
                max_length=int(raw.get("max_length", 512)),
                trust_remote_code=bool(raw.get("trust_remote_code", False)),
            )
    if selected:
        names = [value.strip() for value in selected.split(",") if value.strip()]
        unknown = [name for name in names if name not in specs]
        if unknown:
            raise BenchmarkError(f"unknown scorer ids: {', '.join(unknown)}")
        return [specs[name] for name in names]
    return list(specs.values())


def _prepare_input(input_path: Path | None, exchange_url: str | None, token_env: str) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if input_path is not None:
        return input_path.resolve(), None
    for candidate in (
        DEFAULT_INPUT,
        ROOT / "exchange/" / DEFAULT_EXCHANGE_INPUT_PATH,
        ROOT / ".tmp_case_study_100/silver-oracle-detail.json",
    ):
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve(), None
    try:
        from scripts.collab.lib.exchange_upload import download_exchange_file
    except ImportError as exc:
        raise BenchmarkError("registered input is absent and exchange helper is unavailable; pass --input") from exc
    temporary_dir = tempfile.TemporaryDirectory(prefix="qore-scorer-input-")
    local_path = Path(temporary_dir.name) / Path(DEFAULT_EXCHANGE_INPUT_PATH).name
    try:
        download_exchange_file(
            DEFAULT_EXCHANGE_INPUT_PATH,
            local_path,
            base_url=exchange_url,
            token_env=token_env,
            expected_size=DEFAULT_INPUT_BYTES,
            expected_sha256=DEFAULT_INPUT_SHA256,
        )
    except Exception as exc:
        temporary_dir.cleanup()
        raise BenchmarkError("cannot download registered detail artifact; pass --input") from exc
    return local_path, temporary_dir


def _validate_bundle(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = bundle.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise BenchmarkError(f"expected exactly {EXPECTED_CASES} cases")
    for case_number, case in enumerate(cases, start=1):
        if not isinstance(case, Mapping):
            raise BenchmarkError(f"case {case_number} is not an object")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise BenchmarkError(f"case {case_number} has an empty question")
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise BenchmarkError(f"case {case_number} must contain exactly 50 candidates")
        ids: list[str] = []
        ranks: list[int] = []
        for candidate in top50:
            if not isinstance(candidate, Mapping):
                raise BenchmarkError(f"case {case_number} candidate is not an object")
            identifier = candidate.get("id", candidate.get("passage_id"))
            rank = candidate.get("retrieved_rank", candidate.get("rank"))
            title = candidate.get("title")
            text = candidate.get("text")
            if not isinstance(identifier, (str, int)) or not str(identifier).strip():
                raise BenchmarkError(f"case {case_number} has an invalid candidate id")
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise BenchmarkError(f"case {case_number} has an invalid candidate rank")
            if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
                raise BenchmarkError(f"case {case_number} has invalid candidate text")
            ids.append(str(identifier))
            ranks.append(rank)
        if len(set(ids)) != EXPECTED_TOP50 or ranks != list(range(1, EXPECTED_TOP50 + 1)):
            raise BenchmarkError(f"case {case_number} failed ID/rank identity contract")
    return cases


def _online_text(candidate: Mapping[str, Any]) -> str:
    title = str(candidate.get("title") or "").strip()
    text = str(candidate.get("text") or "").strip()
    return f"{title}. {text}" if title else text


def _online_candidates(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project raw input to the only fields visible to a scorer/selector."""

    result: list[dict[str, Any]] = []
    for candidate in case["top_50"]:
        result.append({
            "id": str(candidate.get("id", candidate.get("passage_id"))),
            "text": _online_text(candidate),
            "retrieved_rank": int(candidate.get("retrieved_rank", candidate.get("rank"))),
            "retrieval_score": float(candidate.get("retrieval_score", 0.0)),
        })
    return result


def _stored_ids(case: Mapping[str, Any], selector_id: str) -> list[str]:
    selectors = case.get("selectors")
    if not isinstance(selectors, list):
        raise BenchmarkError("case selectors must be a list")
    for selector in selectors:
        if isinstance(selector, Mapping) and selector.get("selector_id") == selector_id:
            selected = selector.get("selected_top_5")
            if not isinstance(selected, list) or len(selected) != K:
                raise BenchmarkError(f"stored selector {selector_id} is not length 5")
            ids = [str(item.get("id", "")) for item in selected if isinstance(item, Mapping)]
            if len(ids) != K or len(set(ids)) != K:
                raise BenchmarkError(f"stored selector {selector_id} has invalid IDs")
            return ids
    raise BenchmarkError(f"stored selector {selector_id} is missing")


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


def _posthoc_metrics(case: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], selected_scores: Sequence[float]) -> dict[str, Any]:
    """Read diagnostic labels only after the online selection is complete."""

    top50 = list(case["top_50"])
    oracle_ids = set(_stored_ids(case, "silver_oracle_common_order"))
    selected_ids = [str(item["id"]) for item in selected]
    broad_by_id = {str(item["id"]): _evidence_flags(item)[0] for item in top50}
    direct_by_id = {str(item["id"]): _evidence_flags(item)[1] for item in top50}
    overlap = len(set(selected_ids) & oracle_ids)
    selected_broad = sum(int(broad_by_id.get(identifier, False)) for identifier in selected_ids)
    selected_direct = sum(int(direct_by_id.get(identifier, False)) for identifier in selected_ids)
    top50_broad = sum(int(value) for value in broad_by_id.values())
    top50_direct = sum(int(value) for value in direct_by_id.values())
    tokens = [_token_set(str(item["text"])) for item in selected]
    pairs = [
        len(tokens[left] & tokens[right]) / len(tokens[left] | tokens[right])
        if tokens[left] | tokens[right] else 0.0
        for left in range(len(tokens)) for right in range(left + 1, len(tokens))
    ]
    raw_by_id = {str(item.get("id")): item for item in top50}
    titles = [str(raw_by_id.get(identifier, {}).get("title") or "") for identifier in selected_ids]
    title_pairs = [titles[left] == titles[right] for left in range(len(titles)) for right in range(left + 1, len(titles))]
    return {
        "selected_ids": selected_ids,
        "selected_ranks": [int(item["retrieved_rank"]) for item in selected],
        "selected_scores": [round(float(score), 8) for score in selected_scores],
        "top50_broad_positive_count": top50_broad,
        "top50_direct_positive_count": top50_direct,
        "selected_broad_positive_count": selected_broad,
        "selected_direct_positive_count": selected_direct,
        "retrieval_miss": top50_broad == 0,
        "selector_miss": top50_broad > 0 and selected_broad == 0,
        "silver_oracle_overlap_count": overlap,
        "silver_oracle_precision": overlap / K,
        "token_jaccard_mean": sum(pairs) / len(pairs) if pairs else None,
        "same_title_pair_fraction": sum(title_pairs) / len(title_pairs) if title_pairs else None,
        "unique_title_count": len(set(titles)),
    }


def _summarize(rows: Sequence[Mapping[str, Any]], method: str, status: str = "replayed") -> dict[str, Any]:
    def values(field: str) -> list[float]:
        return [float(row[field]) for row in rows if row.get(field) is not None]

    def mean(field: str) -> float | None:
        items = values(field)
        return sum(items) / len(items) if items else None

    def median(field: str) -> float | None:
        items = values(field)
        return statistics.median(items) if items else None

    overlaps = [int(row["silver_oracle_overlap_count"]) for row in rows]
    return {
        "method": method,
        "status": status,
        "case_count": len(rows),
        "mean_silver_oracle_overlap": mean("silver_oracle_overlap_count"),
        "median_silver_oracle_overlap": median("silver_oracle_overlap_count"),
        "mean_selected_broad_positive_count": mean("selected_broad_positive_count"),
        "mean_selected_direct_positive_count": mean("selected_direct_positive_count"),
        "retrieval_miss_cases": sum(bool(row["retrieval_miss"]) for row in rows),
        "selector_miss_cases": sum(bool(row["selector_miss"]) for row in rows),
        "mean_token_jaccard_redundancy": mean("token_jaccard_mean"),
        "mean_same_title_pair_fraction": mean("same_title_pair_fraction"),
        "mean_unique_title_count": mean("unique_title_count"),
        "mean_selection_time_ms": mean("selection_time_ms"),
        "median_selection_time_ms": median("selection_time_ms"),
        "overlap_distribution": dict(sorted(Counter(overlaps).items())),
    }


def _paired(rows: Mapping[str, Sequence[Mapping[str, Any]]], left: str, right: str) -> dict[str, int]:
    left_map = {int(row["case_number"]): row for row in rows[left]}
    right_map = {int(row["case_number"]): row for row in rows[right]}
    pairs = [
        (left_map[index]["silver_oracle_overlap_count"], right_map[index]["silver_oracle_overlap_count"])
        for index in sorted(set(left_map) & set(right_map))
    ]
    return {
        "left_wins": sum(int(left_value > right_value) for left_value, right_value in pairs),
        "ties": sum(int(left_value == right_value) for left_value, right_value in pairs),
        "right_wins": sum(int(left_value < right_value) for left_value, right_value in pairs),
    }


def _load_qore(args: argparse.Namespace) -> tuple[Any, Any, str | None]:
    try:
        from sentence_transformers import SentenceTransformer
        from applications.rag.selector import select_passages

        device = args.device or ("cuda" if _cuda_available() else "cpu")
        embedder = SentenceTransformer(args.embedding_model, device=device)
        return select_passages, embedder, None
    except Exception as exc:  # dependency/model availability is part of provenance
        return None, None, f"{type(exc).__name__}: {exc}"


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _qore_select(select_passages: Any, embedder: Any, question: str, online: Sequence[Mapping[str, Any]], scores: Sequence[float], args: argparse.Namespace) -> tuple[list[int], float]:
    texts = [str(item["text"]) for item in online]
    started = time.perf_counter()
    query_embedding = np.asarray(embedder.encode(question, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    passage_embeddings = np.asarray(embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    indices = select_passages(
        query_embedding=query_embedding,
        passage_embeddings=passage_embeddings,
        K=K,
        method="qore",
        relevance_scores=np.asarray(scores, dtype=np.float64),
        lam=float(args.qore_lambda),
        num_reads=int(args.qore_num_reads),
        seed=int(args.seed),
        direct_solve_max_n=20,
        qore_prefilter_size=int(args.qore_prefilter_size),
        passage_texts=texts,
        question=question,
    )
    return [int(index) for index in np.asarray(indices).reshape(-1)], (time.perf_counter() - started) * 1000.0


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Classical answer-scorer benchmark: fixed 100 x Top-50",
        "",
        "本轮只比较非生成式 scorer 在固定 Top-50 上的 Top-5。在线输入只有 question、candidate text/id/rank/retrieval score；Silver/gold、答案、Generator 和 evaluator 只在选择完成后作为 L0 诊断读取。",
        "",
        f"- 输入：`{report['input']['path_name']}`，SHA-256 `{report['input']['sha256']}`",
        f"- 规模：{report['protocol']['case_count']} 题 × {report['protocol']['top50_count']} 候选，K={report['protocol']['k']}",
        f"- QORE arm：`{report['protocol']['qore_status']}`",
        "",
        "## Scorer 与选择结果",
        "",
        "| 方法 | 状态 | Silver overlap 均值 | broad 选中均值 | selector miss | 中位耗时 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, summary in report["summaries"].items():
        lines.append(
            f"| {method} | {summary['status']} | "
            f"{summary['mean_silver_oracle_overlap'] if summary['mean_silver_oracle_overlap'] is not None else 'NA'} | "
            f"{summary['mean_selected_broad_positive_count'] if summary['mean_selected_broad_positive_count'] is not None else 'NA'} | "
            f"{summary['selector_miss_cases'] if summary['selector_miss_cases'] is not None else 'NA'} | "
            f"{summary['median_selection_time_ms'] if summary['median_selection_time_ms'] is not None else 'NA'} |"
        )
    lines.extend(["", "Silver overlap 只是事后诊断，不是部署准确率；本轮没有 Generator 输出，因此不报告 EM/F1。", ""])
    lines.extend(["## 相对已注册基线的逐题结果", "", "| 方法 | 相对 QORE：胜/平/负 | 相对旧 Top-k：胜/平/负 |", "|---|---:|---:|"])
    for method, comparisons in report["paired_comparisons"].items():
        lines.append(
            f"| {method} | {comparisons['vs_qore']['left_wins']}/{comparisons['vs_qore']['ties']}/{comparisons['vs_qore']['right_wins']} | "
            f"{comparisons['vs_topk']['left_wins']}/{comparisons['vs_topk']['ties']}/{comparisons['vs_topk']['right_wins']} |"
        )
    lines.extend(["", "## 解释边界", "", "- 旧 QORE/Top-k 是已注册的 DPR Answer Scorer 选择，不是重新运行的标签条件结果。", "- 只有 scorer 直接 Top-5 和匹配的 QORE arm 同时超过旧 Top-k，才允许进入下一轮量子优化候选；单纯超过旧 DPR 不足以证明量子优势。", "- 若模型加载失败、QORE 依赖缺失或输入哈希不符，结果只能作为不完整 screen，不能比较。", ""])
    return "\n".join(lines)


def _output_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root.resolve() / timestamp
    suffix = 1
    while target.exists():
        target = root.resolve() / f"{timestamp}_{suffix}"
        suffix += 1
    target.mkdir(parents=True)
    return target


def run(args: argparse.Namespace) -> Path:
    input_path, temporary_dir = _prepare_input(args.input, args.exchange_url, args.token_env)
    try:
        if not input_path.is_file() or input_path.is_symlink():
            raise BenchmarkError(f"input is missing or symlink: {input_path}")
        bundle = _load_json(input_path)
        cases = _validate_bundle(bundle)
        specs = _load_specs(args.config, args.scorers)
        loaded: dict[str, Any] = {}
        failures: dict[str, str] = {}
        for spec in specs:
            try:
                loaded[spec.scorer_id] = build_scorer(spec, device=args.device, batch_size=args.batch_size)
            except Exception as exc:
                failures[spec.scorer_id] = f"{type(exc).__name__}: {exc}"
                if args.strict_models:
                    raise BenchmarkError(f"cannot load scorer {spec.scorer_id}: {exc}") from exc
        if not loaded:
            raise BenchmarkError("no scorer could be loaded")

        select_passages = embedder = qore_error = None
        if not args.skip_qore:
            select_passages, embedder, qore_error = _load_qore(args)
        qore_ready = select_passages is not None and embedder is not None
        if not qore_ready and not args.skip_qore and not args.allow_qore_failure:
            raise BenchmarkError(f"matched QORE arm is unavailable: {qore_error}")
        method_rows: dict[str, list[dict[str, Any]]] = {}
        trace_cases: list[dict[str, Any]] = []
        retrieval_miss_cases = 0
        for case_number, case in enumerate(cases, start=1):
            question = str(case["question"])
            online = _online_candidates(case)
            identifiers = [str(item["id"]) for item in online]
            ranks = [int(item["retrieved_rank"]) for item in online]
            top50_broad = sum(int(_evidence_flags(item)[0]) for item in case["top_50"])
            retrieval_miss_cases += int(top50_broad == 0)

            # Online phase: every scorer and selector sees only the projected fields.
            online_outputs: dict[str, dict[str, Any]] = {}
            for scorer_id, scorer in loaded.items():
                started = time.perf_counter()
                scores = np.asarray(scorer.score(question, [str(item["text"]) for item in online]), dtype=np.float64)
                scoring_ms = (time.perf_counter() - started) * 1000.0
                direct_indices = rank_indices(scores, ranks, identifiers, K)
                online_outputs[f"{scorer_id}_direct_topk"] = {
                    "indices": direct_indices,
                    "scores": scores,
                    "selection_ms": scoring_ms,
                    "score_summary": score_summary(scores),
                }
                if qore_ready:
                    try:
                        qore_indices, qore_ms = _qore_select(select_passages, embedder, question, online, scores, args)
                        online_outputs[f"{scorer_id}_qore"] = {
                            "indices": qore_indices,
                            "scores": scores,
                            "selection_ms": scoring_ms + qore_ms,
                            "score_summary": score_summary(scores),
                        }
                    except Exception as exc:
                        failures[f"{scorer_id}_qore"] = f"{type(exc).__name__}: {exc}"

            # Post-hoc phase begins only after all online selections for this case.
            case_trace: dict[str, Any] = {"case_number": case_number, "methods": {}}
            for method, output in online_outputs.items():
                indices = [int(index) for index in output["indices"]]
                if len(indices) != K or len(set(indices)) != K:
                    raise BenchmarkError(f"{method} returned invalid selection in case {case_number}")
                selected = [online[index] for index in indices]
                selected_scores = [float(output["scores"][index]) for index in indices]
                row = _posthoc_metrics(case, selected, selected_scores)
                row.update({"case_number": case_number, "selection_time_ms": float(output["selection_ms"])})
                method_rows.setdefault(method, []).append(row)
                case_trace["methods"][method] = {
                    "selected_ids": row["selected_ids"],
                    "selected_ranks": row["selected_ranks"],
                    "selected_scores": row["selected_scores"],
                    "all_scores": [
                        {"id": identifier, "retrieved_rank": rank, "score": round(float(score), 8)}
                        for identifier, rank, score in zip(identifiers, ranks, output["scores"])
                    ],
                    "score_summary": output["score_summary"],
                }
            trace_cases.append(case_trace)
            if args.progress and (case_number % 10 == 0 or case_number == EXPECTED_CASES):
                print(f"  scorer benchmark: {case_number}/{EXPECTED_CASES}", flush=True)

        summaries = {
            method: _summarize(rows, method)
            for method, rows in method_rows.items()
        }
        paired: dict[str, Any] = {}
        for method in method_rows:
            paired[method] = {}
            for baseline, baseline_id in (("vs_qore", "qore_as"), ("vs_topk", "topk_as")):
                baseline_rows: list[dict[str, Any]] = []
                for case in cases:
                    selected_ids = _stored_ids(case, baseline_id)
                    selected = [next(item for item in case["top_50"] if str(item.get("id")) == identifier) for identifier in selected_ids]
                    baseline_rows.append(_posthoc_metrics(case, selected, [float(item.get("answer_scorer_score", 0.0)) for item in selected]))
                baseline_rows = [dict(row, case_number=index) for index, row in enumerate(baseline_rows, start=1)]
                paired[method][baseline] = _paired({method: method_rows[method], baseline_id: baseline_rows}, method, baseline_id)

        output_dir = _output_dir(args.output_root)
        report: dict[str, Any] = {
            "schema_version": "rag.classical_scorer_benchmark_100.v1",
            "artifact_type": "fixed_top50_classical_answer_scorer_benchmark",
            "diagnostic_only": True,
            "input": {"path_name": input_path.name, "bytes": input_path.stat().st_size, "sha256": _sha256(input_path)},
            "protocol": {
                "case_count": EXPECTED_CASES,
                "top50_count": EXPECTED_TOP50,
                "k": K,
                "online_fields": ["question", "id", "text", "retrieved_rank", "retrieval_score"],
                "silver_labels_used_online": False,
                "gold_answers_used_online": False,
                "generator_called": False,
                "evaluator_called": False,
                "qore_status": "available" if qore_ready else "unavailable",
                "qore_error": qore_error,
                "qore_embedding_model": args.embedding_model if not args.skip_qore else None,
                "qore_lambda": args.qore_lambda,
                "qore_prefilter_size": args.qore_prefilter_size,
                "seed": args.seed,
            },
            "scorers": {scorer_id: scorer.identity() for scorer_id, scorer in loaded.items()},
            "failures": failures,
            "summaries": summaries,
            "paired_comparisons": paired,
            "diagnostics": {"retrieval_miss_cases": retrieval_miss_cases, "case_count_validated": len(cases)},
            "provenance": {
                "script": str(SCRIPT_PATH.relative_to(ROOT)) if SCRIPT_PATH.is_relative_to(ROOT) else str(SCRIPT_PATH),
                "script_sha256": _sha256(SCRIPT_PATH),
                "git_revision": _git_revision(),
            },
            "next_step": "仅当某个 scorer 的直接 Top-5 或匹配 QORE arm 在同一固定输入上超过旧 Top-k，并且冗余/成本没有明显恶化时，才进入量子优化候选；否则先停止换 scorer。",
        }
        _write_json(output_dir / "summary.json", report)
        _write_json(output_dir / "selector_trace.json", {
            "schema_version": "rag.classical_scorer_benchmark_100.trace.v1",
            "diagnostic_only": True,
            "input": report["input"],
            "protocol": report["protocol"],
            "cases": trace_cases,
        })
        metadata = {
            "schema_version": "rag.classical_scorer_benchmark_100.run_metadata.v1",
            "run_timestamp_utc": output_dir.name,
            "input": report["input"],
            "scorers": report["scorers"],
            "failures": failures,
            "output_files": ["summary.json", "report.md", "run_metadata.json", "upload_manifest.json", "selector_trace.json"],
            "selector_trace_exchange_only": True,
        }
        _write_json(output_dir / "run_metadata.json", metadata)
        (output_dir / "report.md").write_text(_markdown(report) + "\n", encoding="utf-8")
        summary_bytes = (output_dir / "summary.json").stat().st_size
        if summary_bytes > MAX_GITHUB_BYTES:
            raise BenchmarkError(f"compact summary exceeds 1 MiB: {summary_bytes} bytes")
        upload_manifest = {
            "schema_version": "rag.classical_scorer_benchmark_100.upload_manifest.v1",
            "artifact_type": "classical_scorer_benchmark_100_upload_manifest",
            "status": "ready_for_authenticated_exchange_upload",
            "target_directory": f"{DEFAULT_EXCHANGE_TARGET}/{output_dir.name}",
            "exchange_files": [{"name": "selector_trace.json"}],
            "compact_files": [
                {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": _sha256(output_dir / name)}
                for name in ("summary.json", "report.md", "run_metadata.json")
            ],
            "privacy": {"raw_passage_text_in_trace": False, "gold_answers_in_output": False, "silver_labels_used_online": False},
            "provenance": report["provenance"],
        }
        _write_json(output_dir / "upload_manifest.json", upload_manifest)
        if args.upload:
            try:
                from scripts.collab.lib.exchange_upload import upload_manifest as upload_exchange_manifest
                receipts = upload_exchange_manifest(output_dir / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
            except Exception as exc:
                raise BenchmarkError(f"exchange upload failed: {type(exc).__name__}: {exc}") from exc
            _write_json(output_dir / "upload_receipts.json", {"receipts": receipts})
        print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "summary.json"), "upload": bool(args.upload)}, ensure_ascii=False))
        return output_dir
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


def _git_revision() -> str | None:
    import subprocess
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="optional JSON scorer allowlist")
    parser.add_argument("--scorers", default=None, help="comma-separated scorer IDs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exchange-url", default=None)
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--qore-prefilter-size", type=int, default=15)
    parser.add_argument("--qore-lambda", type=float, default=2.0)
    parser.add_argument("--qore-num-reads", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-qore", action="store_true")
    parser.add_argument("--allow-qore-failure", action="store_true")
    parser.add_argument("--strict-models", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.qore_prefilter_size < K or args.qore_num_reads < 1:
        parser.error("batch size, qore prefilter and qore reads must be positive")
    try:
        if args.validate_only:
            input_path, temporary_dir = _prepare_input(args.input, args.exchange_url, args.token_env)
            try:
                bundle = _load_json(input_path)
                cases = _validate_bundle(bundle)
                print(json.dumps({
                    "input": str(input_path),
                    "bytes": input_path.stat().st_size,
                    "sha256": _sha256(input_path),
                    "case_count": len(cases),
                    "top50_count": len(cases[0]["top_50"]),
                    "model_calls": 0,
                    "selector_called": False,
                }, ensure_ascii=False))
                return 0
            finally:
                if temporary_dir is not None:
                    temporary_dir.cleanup()
        run(args)
    except (BenchmarkError, ScorerBenchmarkError) as exc:
        print(f"classical scorer benchmark failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
