#!/usr/bin/env python3
"""Replay the DUET-inspired two-stage selector on the fixed 100-case Top-50.

This is an L0 selector-only screen. The registered Silver annotations are
read after selection for diagnostics and never enter the selector objective.
The runner rehydrates Wiki-DPR locally, validates exact Top-50 identity, and
uploads only the large selector trace to the authenticated exchange service.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collab.five_ideas import replay_selector_benchmark_100_historical as historical


DEFAULT_CONFIG = ROOT / "configs/experiments/q_duet_selector_replay_100.yaml"
DEFAULT_PLAN = ROOT / "configs/experiments/q_duet_selector_replay_100_plan.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exchange/five_ideas/q_duet_selector_replay_100"
EXPECTED_CASES = historical.EXPECTED_CASES
EXPECTED_TOP50 = historical.EXPECTED_TOP50
K = historical.K
SEED = 42


class ReplayError(RuntimeError):
    """Raised when the frozen Q-DUET replay contract is violated."""


def _load_plan(path: Path) -> dict[str, Any]:
    plan = historical._load_json(path)
    if plan.get("authorization") != "implemented":
        raise ReplayError("Q-DUET plugin plan is not implemented")
    if plan.get("reproducibility", {}).get("silver_labels_used_online") is not False:
        raise ReplayError("Q-DUET plan permits Silver leakage")
    return plan


def _validate_config(config_path: Path, plan_path: Path) -> dict[str, Any]:
    config = historical._load_yaml(config_path)
    phase = config.get("phase")
    if not isinstance(phase, Mapping):
        raise ReplayError("phase section is missing")
    expected = {
        "name": "q_duet_selector_replay_100",
        "schema_version": 1,
        "evidence_tier": "L0_diagnostic",
        "diagnostic_only": True,
        "selection_mutation": False,
    }
    for key, value in expected.items():
        if phase.get(key) != value:
            raise ReplayError(f"Q-DUET config mismatch for {key}")
    input_spec = phase.get("input")
    if not isinstance(input_spec, Mapping):
        raise ReplayError("100-case input contract is missing")
    for key in ("case_count", "top50_count", "exchange_path", "exchange_bytes", "exchange_sha256"):
        historical_value = historical._load_yaml(historical.DEFAULT_CONFIG)["phase"]["input"].get(key)
        if input_spec.get(key) != historical_value:
            raise ReplayError(f"Q-DUET input contract mismatch for {key}")
    if phase.get("retrieval") != historical._load_yaml(historical.DEFAULT_CONFIG)["phase"]["retrieval"]:
        raise ReplayError("Q-DUET retrieval contract is not frozen")
    selectors = phase.get("selectors")
    if not isinstance(selectors, list):
        raise ReplayError("selector allowlist is missing")
    ids = [str(item.get("id")) for item in selectors if isinstance(item, Mapping)]
    required = ["qore_as", "topk_as", "mmr_as", "submodular_as", "spectral_dpp_as", "q_duet_rag"]
    if ids != required:
        raise ReplayError(f"unexpected selector allowlist: {ids}")
    _load_plan(plan_path)
    return dict(phase)


def _select_q_duet(
    query_embedding: Any,
    embeddings: Any,
    candidates: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    import numpy as np
    from applications.rag.q_duet_selector import select_passages

    answer_scores = np.asarray([float(item["answer_scorer_score"]) for item in candidates], dtype=np.float64)
    retrieval_scores = np.asarray([float(item["retrieval_score"]) for item in candidates], dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    selected = select_passages(
        query_embedding,
        embeddings,
        K,
        relevance_scores=answer_scores,
        retrieval_scores=retrieval_scores,
        stage1_budget=int(spec["stage1_budget"]),
        dominant_budget=int(spec["dominant_budget"]),
        stage1_solver=str(spec["stage1_solver"]),
        stage2_solver=str(spec["stage2_solver"]),
        stage1_lam=float(spec["stage1_lam"]),
        stage1_gamma=float(spec["stage1_gamma"]),
        stage2_lam=float(spec["stage2_lam"]),
        stage2_gamma=float(spec["stage2_gamma"]),
        cross_complementarity=float(spec["cross_complementarity"]),
        num_reads=int(spec["num_reads"]),
        seed=int(spec["seed"]),
        diagnostics=diagnostics,
    )
    return [int(value) for value in np.asarray(selected).reshape(-1)], diagnostics


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Q-DUET-RAG 100题 Top-50 -> Top-5 replay",
        "",
        "这是固定 Top-50 上的 L0 selector-only screen。Q-DUET-RAG 借鉴 DUET-VLM 的 dominant/residual 双阶段结构：第一阶段用检索结构和候选中心性选 dominant passages，再从残余候选中按答案支持与覆盖性选 contextual representatives；第二阶段用问题条件信号完成最终 QUBO 选择。Silver labels 只在选择完成后用于诊断，未调用 Generator。",
        "",
        f"- 输入：`{report['input']['path_name']}`，SHA-256 `{report['input']['sha256']}`",
        f"- 规模：{report['protocol']['case_count']} 题 × {report['protocol']['top50_count']} 候选，K={report['protocol']['k']}",
        f"- 代码 revision：`{report['provenance'].get('git_revision') or 'unavailable'}`",
        "",
        "## 结果",
        "",
        "| 方法 | 状态 | Silver overlap 均值 | set-F1 均值 | broad 选中均值 | selector miss | embedding 冗余 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method, summary in report["summaries"].items():
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {method} | {summary['status']} | {fmt(summary['mean_silver_oracle_overlap'])} | "
            f"{fmt(summary['mean_silver_oracle_set_f1'])} | {fmt(summary['mean_selected_broad_positive_count'])} | "
            f"{summary['selector_miss_cases']} | {fmt(summary['mean_embedding_cosine_redundancy'])} |"
        )
    lines.extend([
        "",
        "## Q-DUET 与基线的逐题胜负",
        "",
        "| 对照 | Q-DUET 胜/平/负 |",
        "|---|---:|",
    ])
    for baseline in ("qore_as", "topk_as", "mmr_as", "submodular_as", "spectral_dpp_as"):
        values = report["paired_comparisons"]["q_duet_rag"][f"vs_{baseline}"]
        lines.append(f"| {baseline} | {values['left_wins']}/{values['ties']}/{values['right_wins']} |")
    lines.extend([
        "",
        "## 边界",
        "",
        f"- Wiki-DPR Top-50 逐题 ID、title、正文和检索分数通过精确重建；最大检索分数绝对误差 `{report['diagnostics']['max_retrieval_score_abs_error']:.6g}`。",
        f"- retrieval miss：{report['diagnostics']['retrieval_miss_cases']}/{report['protocol']['case_count']}；这些题不归因于 selector。",
        "- Q-DUET 的 anneal/brute 运行是同一 QUBO 的经典求解控制；代码同时保留 qaoa_qk/qaoa_pl/qaoa_tc 后端入口。不能仅凭本 replay 声称量子加速。",
        "- 完整 selector trace 只通过 18083 exchange 保存；GitHub 只保存 compact summary、report、metadata 和 manifest。",
        "",
    ])
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace, *, input_path: Path, input_source: Mapping[str, Any], phase: Mapping[str, Any]) -> Path:
    import numpy as np
    from applications.rag.data import make_corpus_manager
    from applications.rag.retrieval import make_encoder

    bundle = historical._load_json(input_path)
    cases = historical._validate_input(bundle)
    selectors = list(phase["selectors"])
    encoder = make_encoder("dpr")
    manager = make_corpus_manager("wiki_dpr", {
        "wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"],
        "nprobe": int(phase["retrieval"]["nprobe"]),
    })
    manager.build([])
    method_rows: dict[str, list[dict[str, Any]]] = {}
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    retrieval_miss_cases = 0
    max_score_error = 0.0

    for case_number, case in enumerate(cases, start=1):
        question = str(case["question"])
        query_embedding = encoder.encode_queries([question])[0]
        records, embeddings = historical._retrieve(manager, query_embedding)
        validation = historical._validate_retrieval(case, records)
        max_score_error = max(max_score_error, float(validation["max_retrieval_score_abs_error"]))
        candidates = [historical._online_candidate(item) for item in case["top_50"]]
        embeddings_by_id = {
            str(record["id"]): [float(value) for value in embeddings[index]]
            for index, record in enumerate(records)
        }
        oracle_ids = set(historical._stored_ids(case, "silver_oracle_common_order"))
        if not any(historical._evidence_flags(item)[0] for item in candidates):
            retrieval_miss_cases += 1
        trace_methods: dict[str, Any] = {}

        for selector in selectors:
            method_id = str(selector["id"])
            started_one = time.perf_counter()
            if method_id in {"qore_as", "topk_as"}:
                source_id = str(selector["source_selector_id"])
                selected_ids = historical._stored_ids(case, source_id)
                diagnostics = {"source": source_id}
                selected_indices = [next(index for index, item in enumerate(records) if str(item["id"]) == identifier) for identifier in selected_ids]
            elif method_id == "q_duet_rag":
                selected_indices, diagnostics = _select_q_duet(query_embedding, embeddings, candidates, selector)
                selected_ids = [str(records[index]["id"]) for index in selected_indices]
            else:
                spec = dict(selector)
                spec["question"] = question
                selected_indices, diagnostics = historical._select(spec, query_embedding, embeddings, candidates, None)
                selected_ids = [str(records[index]["id"]) for index in selected_indices]
            elapsed_ms = (time.perf_counter() - started_one) * 1000.0
            row = historical._build_row(
                case,
                candidates,
                selected_ids,
                oracle_ids,
                embeddings_by_id,
                elapsed_ms=elapsed_ms,
                diagnostics=diagnostics,
            )
            row["case_number"] = case_number
            method_rows.setdefault(method_id, []).append(row)
            trace_entry: dict[str, Any] = {
                "selected_ids": selected_ids,
                "selected_ranks": row["selected_ranks"],
                "selector_diagnostics": diagnostics,
            }
            if method_id == "q_duet_rag":
                for name in ("dominant_indices", "contextual_indices", "stage1_indices", "selected_indices"):
                    indices = diagnostics.get(name)
                    if isinstance(indices, list):
                        trace_entry[f"{name}_ids"] = [str(records[int(index)]["id"]) for index in indices]
                        trace_entry[f"{name}_ranks"] = [int(records[int(index)]["retrieved_rank"]) for index in indices]
            trace_methods[method_id] = trace_entry

        traces.append({
            "case_number": case_number,
            "source_id": str(case.get("source_id") or case.get("question_id") or ""),
            "question_id": str(case.get("question_id") or ""),
            "top50_candidate_id_sha256": hashlib.sha256("\n".join(str(item["id"]) for item in candidates).encode("utf-8")).hexdigest(),
            "retrieval_validation": validation,
            "methods": trace_methods,
        })
        if args.progress and (case_number == EXPECTED_CASES or case_number % 10 == 0):
            print(f"  Q-DUET replay: {case_number}/{EXPECTED_CASES}", flush=True)

    baseline_methods = {"qore_as", "topk_as"}
    summaries = {
        method: historical._summary(rows, method, "registered_baseline" if method in baseline_methods else "replayed")
        for method, rows in method_rows.items()
    }
    paired = {
        "q_duet_rag": {
            f"vs_{method}": historical._paired(method_rows, "q_duet_rag", method)
            for method in ("qore_as", "topk_as", "mmr_as", "submodular_as", "spectral_dpp_as")
        }
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root.resolve() if args.output_root else DEFAULT_OUTPUT_ROOT
    output_dir = historical._unique_output_dir(output_root, timestamp)
    target_directory = f"five_ideas/q_duet_selector_replay_100/{output_dir.name}"
    input_record = {"path_name": input_path.name, "sha256": historical._sha256(input_path), **dict(input_source)}
    report: dict[str, Any] = {
        "schema_version": "rag.q_duet_selector_replay_100.v1",
        "artifact_type": "q_duet_selector_only_replay",
        "diagnostic_only": True,
        "input": input_record,
        "protocol": {
            "case_count": EXPECTED_CASES,
            "top50_count": EXPECTED_TOP50,
            "k": K,
            "silver_oracle_used_for_selection": False,
            "generator_called": False,
            "retrieval": dict(phase["retrieval"]),
            "selector_allowlist": [str(item["id"]) for item in selectors],
        },
        "provenance": {
            "git_revision": historical._git("rev-parse", "HEAD"),
            "git_branch": historical._git("branch", "--show-current"),
            "script": str(SCRIPT_PATH.relative_to(ROOT)),
            "script_sha256": historical._sha256(SCRIPT_PATH),
            "config_path": str(args.config.relative_to(ROOT)),
            "config_sha256": historical._sha256(args.config),
            "plan_path": str(args.plan.relative_to(ROOT)),
            "plan_sha256": historical._sha256(args.plan),
        },
        "summaries": summaries,
        "paired_comparisons": paired,
        "diagnostics": {
            "retrieval_miss_cases": retrieval_miss_cases,
            "max_retrieval_score_abs_error": max_score_error,
            "exact_top50_identity_validated": True,
            "case_count_validated": EXPECTED_CASES,
        },
        "claim_ceiling": "L0 diagnostic; fixed Top-50 selector alignment only",
        "next_step": "Only a candidate that beats Top-k on the preregistered paired Silver alignment gate can proceed to a later frozen-Generator screen.",
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    _write_json(output_dir / "selector_trace.json", {
        "schema_version": "rag.q_duet_selector_replay_100.trace.v1",
        "diagnostic_only": True,
        "input": report["input"],
        "provenance": report["provenance"],
        "target_directory": target_directory,
        "cases": traces,
    })
    metadata = {
        "schema_version": "rag.q_duet_selector_replay_100.metadata.v1",
        "artifact_type": "q_duet_selector_replay_100_run_metadata",
        "status": "completed",
        "diagnostic_only": True,
        "selection_mutation": False,
        "generated_at_utc": timestamp,
        "input": report["input"],
        "provenance": report["provenance"],
        "protocol": report["protocol"],
        "outputs": {"root": str(output_root.relative_to(ROOT)), "target_directory": target_directory, "exchange_files": ["selector_trace.json"]},
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "validation": report["diagnostics"],
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    manifest = {
        "schema_version": "rag.q_duet_selector_replay_100.upload_manifest.v1",
        "artifact_type": "q_duet_selector_replay_100_upload_manifest",
        "status": "ready_for_authenticated_exchange_upload",
        "target_directory": target_directory,
        "generated_at_utc": timestamp,
        "exchange_files": [{"name": "selector_trace.json", "exchange_path": f"{target_directory}/selector_trace.json"}],
        "compact_files": [
            {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": historical._sha256(output_dir / name)}
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
            {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": historical._sha256(output_dir / name)}
            for name in ("summary.json", "report.md", "run_metadata.json")
        ]
        _write_json(output_dir / "upload_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "target_directory": target_directory, "uploaded": not args.no_upload}, ensure_ascii=False))
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--exchange-url", default=None)
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    for field in ("config", "plan"):
        path = getattr(args, field)
        if not path.is_absolute():
            setattr(args, field, ROOT / path)
    try:
        phase = _validate_config(args.config.resolve(), args.plan.resolve())
        input_args = argparse.Namespace(
            input=args.input,
            exchange_url=args.exchange_url,
            token_env=args.token_env,
        )
        input_path, temporary_dir, input_source = historical._prepare_input(input_args)
        try:
            if args.validate_only:
                bundle = historical._load_json(input_path)
                cases = historical._validate_input(bundle)
                print(json.dumps({"status": "valid", "phase": phase["name"], "case_count": len(cases), "top50_count": EXPECTED_TOP50, "input": {"path_name": input_path.name, **input_source}}, ensure_ascii=False))
                return 0
            run(args, input_path=input_path, input_source=input_source, phase=phase)
        finally:
            if temporary_dir is not None:
                temporary_dir.cleanup()
    except (ReplayError, historical.ReplayError, ValueError, OSError) as exc:
        print(f"Q-DUET replay error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
