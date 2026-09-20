#!/usr/bin/env python3
"""Run the registered F50-QWI mechanism audit and 100-case L0 replay.

The runner first rehydrates and verifies every frozen Wiki-DPR Top-50.  For
each case it runs the numerical/identity/no-leakage F50-QWI audit using only
ordinary online fields.  It reads Silver evidence only after all four F50
arms have selected their Top-5 sets.  No Generator, evaluator, labels, or
answer feedback enter the selector.
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
import tempfile
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
DEFAULT_CONFIG = ROOT / "configs/experiments/f50_qwi_selector_replay_100.yaml"
DEFAULT_PLAN = ROOT / "configs/experiments/f50_qwi_selector_replay_100_plan.json"
DEFAULT_OUTPUT_ROOT = ROOT / "exchange/five_ideas/f50_qwi_selector_replay_100"
DEFAULT_EXCHANGE_INPUT_PATH = (
    "five_ideas/selector_replay_100_historical_input/"
    "silver-oracle-top5-100-20260915T120525Z-detail.json"
)
DEFAULT_EXCHANGE_INPUT_BYTES = 8046318
DEFAULT_EXCHANGE_INPUT_SHA256 = "669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731"
EXPECTED_CASES = 100
EXPECTED_TOP50 = 50
K = 5
SCORE_ABS_TOLERANCE = 1.0e-3


class ReplayError(RuntimeError):
    """Raised when the registered F50-QWI protocol cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        "name": "f50_qwi_selector_replay_100",
        "schema_version": 1,
        "evidence_tier": "L0_diagnostic",
        "diagnostic_only": True,
        "selection_mutation": False,
        "generator_called": False,
        "evaluator_called": False,
        "silver_labels_used_online": False,
    }
    for key, value in expected.items():
        if phase.get(key) != value:
            raise ReplayError(f"F50-QWI config mismatch for {key}")
    input_spec = phase.get("input")
    if not isinstance(input_spec, Mapping) or input_spec != {
        "exchange_path": DEFAULT_EXCHANGE_INPUT_PATH,
        "exchange_bytes": DEFAULT_EXCHANGE_INPUT_BYTES,
        "exchange_sha256": DEFAULT_EXCHANGE_INPUT_SHA256,
        "case_count": EXPECTED_CASES,
        "top50_count": EXPECTED_TOP50,
    }:
        raise ReplayError("registered 100-case input contract is not frozen")
    retrieval = phase.get("retrieval")
    if retrieval != {
        "corpus_mode": "wiki_dpr",
        "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64,
        "top_k": 50,
        "exact_identity_required": True,
    }:
        raise ReplayError("Wiki-DPR identity contract is not frozen")
    mechanism = phase.get("mechanism")
    expected_mechanism = {
        "register_qubits": 6,
        "register_dimension": 64,
        "valid_basis_states": "|0> through |49>",
        "invalid_basis_states": "|50> through |63>, initialized at zero and block-isolated",
        "score_normalization": "independent_top50_minmax(retrieval_score,answer_scorer_score)",
        "retrieval_weight": 0.5,
        "answer_scorer_weight": 0.5,
        "initial_distribution": "softmax((0.5*r_norm+0.5*a_norm)/1.0)",
        "diagonal_strength": 1.0,
        "interaction_strength_kappa": 0.85,
        "coupling": "off_diagonal_centered_l2_cosine_similarity_divided_by_spectral_norm",
        "evolution_time_tau": 1.0,
        "phase_scramble": "sha256(phase_seed:passage_id) mapped uniformly to [0,2pi)",
        "phase_seed": 42,
        "born_sampling": "none_exact_probabilities_only",
        "tie_break": "retrieved_rank_then_passage_id",
        "single_particle_marginal_ranking": True,
        "classically_exactly_simulated": True,
    }
    if mechanism != expected_mechanism:
        raise ReplayError("F50-QWI mechanism parameters or formulas changed")
    controls = phase.get("controls")
    if controls != ["born_exact", "real_diffusion_control", "dephased_control", "phase_scramble_control"]:
        raise ReplayError("F50-QWI control allowlist changed")
    gate = phase.get("gates")
    if not isinstance(gate, Mapping) or gate.get("max_total_cpu_seconds") != 600 or gate.get("strict_born_mean_silver_overlap_gt") != 3.55:
        raise ReplayError("F50-QWI replay gates changed")
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v1" or plan.get("authorization") != "implemented":
        raise ReplayError("F50-QWI plugin plan is not implemented")
    if plan.get("reproducibility", {}).get("silver_labels_used_online") is not False:
        raise ReplayError("plugin plan permits Silver leakage")
    return dict(phase)


def _prepare_input(args: argparse.Namespace) -> tuple[Path, tempfile.TemporaryDirectory | None, dict[str, Any]]:
    if args.input is not None:
        path = args.input if args.input.is_absolute() else ROOT / args.input
        return path.resolve(), None, {"source": "explicit_local"}
    try:
        from scripts.collab.lib.exchange_upload import ExchangeUploadError, download_exchange_file
    except ImportError as exc:
        raise ReplayError("exchange download helper is unavailable; pass --input explicitly") from exc
    temporary_dir = tempfile.TemporaryDirectory(prefix="qore-f50-qwi-input-")
    input_path = Path(temporary_dir.name) / Path(DEFAULT_EXCHANGE_INPUT_PATH).name
    try:
        receipt = download_exchange_file(
            DEFAULT_EXCHANGE_INPUT_PATH,
            input_path,
            base_url=args.exchange_url,
            token_env=args.token_env,
            expected_size=DEFAULT_EXCHANGE_INPUT_BYTES,
            expected_sha256=DEFAULT_EXCHANGE_INPUT_SHA256,
        )
    except (ExchangeUploadError, OSError) as exc:
        temporary_dir.cleanup()
        raise ReplayError("cannot download registered 100-case detail JSON; set QORE_EXCHANGE_TOKEN or pass --input") from exc
    return input_path, temporary_dir, {"source": "authenticated_exchange", "exchange_path": receipt["path"], "exchange_sha256": receipt["sha256"]}


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
        identifiers = [str(item.get("id", "")) for item in selected if isinstance(item, Mapping)]
        if len(identifiers) != K or len(set(identifiers)) != K or any(not item for item in identifiers):
            raise ReplayError(f"stored selector {selector_id} has invalid IDs")
        return identifiers
    raise ReplayError(f"stored selector {selector_id} is missing")


def _raw_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
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
        "id": str(identifier), "title": title, "text": text, "retrieved_rank": rank,
        "retrieval_score": _finite(candidate.get("retrieval_score"), "retrieval_score"),
        "answer_scorer_score": _finite(candidate.get("answer_scorer_score"), "answer_scorer_score"),
        "evidence": candidate.get("evidence"),
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
        candidates = [_raw_candidate(item) for item in top50 if isinstance(item, Mapping)]
        if len(candidates) != EXPECTED_TOP50:
            raise ReplayError(f"case {number} contains an invalid candidate")
        if [item["retrieved_rank"] for item in candidates] != list(range(1, EXPECTED_TOP50 + 1)):
            raise ReplayError(f"case {number} ranks are not the registered 1..50 order")
        if len({item["id"] for item in candidates}) != EXPECTED_TOP50:
            raise ReplayError(f"case {number} IDs are not unique")
        for selector_id in ("qore_common_order", "topk_common_order", "silver_oracle_common_order"):
            _stored_ids(case, selector_id)
    return cases


def _retrieve(manager: Any, query_embedding: Any) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np
    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise ReplayError("Wiki-DPR manager does not expose its indexed dataset")
    scores, retrieved = dataset.get_nearest_examples("embeddings", np.asarray(query_embedding, dtype=np.float32), k=EXPECTED_TOP50)
    if any(key not in retrieved for key in ("id", "title", "text", "embeddings")) or len(scores) != EXPECTED_TOP50:
        raise ReplayError("Wiki-DPR retrieval result lacks exact Top-50 fields")
    records = [{
        "id": str(retrieved["id"][index]), "title": str(retrieved["title"][index] or ""),
        "text": str(retrieved["text"][index] or ""), "retrieved_rank": index + 1,
        "retrieval_score": float(scores[index]),
    } for index in range(EXPECTED_TOP50)]
    return records, np.asarray(retrieved["embeddings"], dtype=np.float64)


def _validate_retrieval(case: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [_raw_candidate(item) for item in case["top_50"]]
    if len(records) != EXPECTED_TOP50:
        raise ReplayError("rehydrated Top-50 length mismatch")
    id_mismatches = 0
    text_mismatches = 0
    max_score_error = 0.0
    for stored, observed in zip(expected, records):
        id_mismatches += int(stored["id"] != observed["id"])
        text_mismatches += int(stored["title"] != observed["title"] or stored["text"] != observed["text"])
        max_score_error = max(max_score_error, abs(float(stored["retrieval_score"]) - float(observed["retrieval_score"])))
    if id_mismatches or text_mismatches or max_score_error > SCORE_ABS_TOLERANCE:
        raise ReplayError(
            "rehydrated Top-50 differs from registered artifact: "
            f"id_mismatches={id_mismatches}, text_mismatches={text_mismatches}, max_score_error={max_score_error:.6g}"
        )
    return {"ids_match": True, "texts_match": True, "max_retrieval_score_abs_error": max_score_error, "candidate_count": EXPECTED_TOP50}


def _online_candidates(raw: Sequence[Mapping[str, Any]], embeddings: Any) -> list[dict[str, Any]]:
    if len(raw) != EXPECTED_TOP50 or len(embeddings) != EXPECTED_TOP50:
        raise ReplayError("cannot form F50-QWI online candidate contract")
    return [{
        "id": str(candidate["id"]), "retrieved_rank": int(candidate["retrieved_rank"]),
        "retrieval_score": float(candidate["retrieval_score"]),
        "answer_scorer_score": float(candidate["answer_scorer_score"]),
        "dpr_passage_embedding": [float(value) for value in embeddings[index]],
    } for index, candidate in enumerate(raw)]


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


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    import numpy as np
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denominator if denominator else 0.0


def _redundancy(selected: Sequence[Mapping[str, Any]], embeddings_by_id: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    import re
    token_sets = [set(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", f"{item['title']}. {item['text']}".lower())) for item in selected]
    pair_count = 0
    token_sum = 0.0
    cosine_sum = 0.0
    same_title = 0
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            pair_count += 1
            union = token_sets[left] | token_sets[right]
            token_sum += len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0
            cosine_sum += _cosine(embeddings_by_id[str(selected[left]["id"])], embeddings_by_id[str(selected[right]["id"])])
            same_title += int(selected[left]["title"] == selected[right]["title"])
    return {
        "token_jaccard_mean": token_sum / pair_count if pair_count else None,
        "embedding_cosine_mean": cosine_sum / pair_count if pair_count else None,
        "same_title_pair_fraction": same_title / pair_count if pair_count else None,
        "unique_title_count": len({str(item["title"]) for item in selected}),
    }


def _build_row(raw: Sequence[Mapping[str, Any]], selected_ids: Sequence[str], oracle_ids: set[str], embeddings_by_id: Mapping[str, Sequence[float]], *, elapsed_ms: float, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    if len(selected_ids) != K or len(set(selected_ids)) != K:
        raise ReplayError("selector did not return five unique passage IDs")
    by_id = {str(item["id"]): item for item in raw}
    if any(identifier not in by_id for identifier in selected_ids):
        raise ReplayError("selection includes an unknown passage ID")
    selected = [by_id[identifier] for identifier in selected_ids]
    broad = {str(item["id"]): _evidence_flags(item)[0] for item in raw}
    direct = {str(item["id"]): _evidence_flags(item)[1] for item in raw}
    overlap = len(set(selected_ids) & oracle_ids)
    precision = overlap / K
    recall = overlap / len(oracle_ids) if oracle_ids else None
    f1 = 2.0 * precision * recall / (precision + recall) if recall is not None and precision + recall else (0.0 if oracle_ids else None)
    return {
        "selected_set_sha256": hashlib.sha256("\n".join(sorted(selected_ids)).encode("utf-8")).hexdigest(),
        "selected_ranks": [int(item["retrieved_rank"]) for item in selected],
        "selected_answer_scorer_scores": [round(float(item["answer_scorer_score"]), 8) for item in selected],
        "selected_broad_positive_count": sum(int(broad[identifier]) for identifier in selected_ids),
        "selected_direct_positive_count": sum(int(direct[identifier]) for identifier in selected_ids),
        "top50_broad_positive_count": sum(broad.values()),
        "top50_direct_positive_count": sum(direct.values()),
        "retrieval_miss": not any(broad.values()),
        "selector_miss": any(broad.values()) and not any(broad[identifier] for identifier in selected_ids),
        "silver_oracle_overlap_count": overlap,
        "silver_oracle_precision": precision,
        "silver_oracle_recall": recall,
        "silver_oracle_set_f1": f1,
        "redundancy": _redundancy(selected, embeddings_by_id),
        "selection_time_ms": elapsed_ms,
        "selector_diagnostics": dict(diagnostics),
    }


def _mean(rows: Sequence[Mapping[str, Any]], path: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for part in path.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _summary(rows: Sequence[Mapping[str, Any]], method: str, status: str) -> dict[str, Any]:
    overlap = [int(row["silver_oracle_overlap_count"]) for row in rows]
    return {
        "method": method, "status": status, "case_count": len(rows),
        "mean_silver_oracle_overlap": _mean(rows, "silver_oracle_overlap_count"),
        "median_silver_oracle_overlap": statistics.median(overlap) if overlap else None,
        "mean_silver_oracle_set_f1": _mean(rows, "silver_oracle_set_f1"),
        "mean_selected_broad_positive_count": _mean(rows, "selected_broad_positive_count"),
        "mean_selected_direct_positive_count": _mean(rows, "selected_direct_positive_count"),
        "retrieval_miss_cases": sum(bool(row["retrieval_miss"]) for row in rows),
        "selector_miss_cases": sum(bool(row["selector_miss"]) for row in rows),
        "mean_embedding_cosine_redundancy": _mean(rows, "redundancy.embedding_cosine_mean"),
        "mean_selection_time_ms": _mean(rows, "selection_time_ms"),
        "median_selection_time_ms": statistics.median([float(row["selection_time_ms"]) for row in rows]) if rows else None,
        "overlap_distribution": dict(sorted(Counter(overlap).items())),
    }


def _paired(rows: Mapping[str, Sequence[Mapping[str, Any]]], left: str, right: str) -> dict[str, int]:
    left_map = {int(row["case_number"]): row for row in rows[left]}
    right_map = {int(row["case_number"]): row for row in rows[right]}
    pairs = [(left_map[index]["silver_oracle_overlap_count"], right_map[index]["silver_oracle_overlap_count"]) for index in sorted(set(left_map) & set(right_map))]
    return {"left_wins": sum(int(a > b) for a, b in pairs), "ties": sum(int(a == b) for a, b in pairs), "right_wins": sum(int(a < b) for a, b in pairs)}


def _unique_output_dir(root: Path, timestamp: str) -> Path:
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# F50-QWI 100题 selector replay",
        "",
        "该运行是固定 Top-50 -> Top-5 的 L0 诊断。F50-QWI 是六量子比特寄存器上的单粒子边缘 Born 排序，并由经典矩阵特征分解精确计算；它不是联合五段量子状态，也不构成量子优势或 L1/L2 结论。",
        "",
        f"- 输入 SHA-256：`{report['input']['sha256']}`",
        f"- Top-50 identity：`{report['diagnostics']['exact_top50_identity_validated']}`",
        f"- 数值/泄漏机制门：`{report['gates']['mechanism_audit_pass']}`",
        f"- 四臂总选择耗时：`{report['gates']['total_selector_seconds']:.3f}s` / 600s",
        "",
        "## Silver 后验诊断",
        "",
        "| 方法 | Silver overlap 均值 | broad 选中均值 | selector miss | 归一化 embedding 冗余 | 中位选择 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, summary in report["summaries"].items():
        fmt = lambda value: "NA" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {method} | {fmt(summary['mean_silver_oracle_overlap'])} | {fmt(summary['mean_selected_broad_positive_count'])} | {summary['selector_miss_cases']} | {fmt(summary['mean_embedding_cosine_redundancy'])} | {fmt(summary['median_selection_time_ms'])} |"
        )
    lines.extend([
        "", "## 判定", "",
        f"- Born 严格超过 Top-k 3.55/5：`{report['gates']['born_strictly_beats_topk']}`。",
        f"- 下一层完整 selector replay 是否获准：`{report['gates']['next_complete_selector_replay_authorized']}`。",
        "- Silver labels 只在所有选择完成后读取；13 个或更多 retrieval miss 不能归因于 selector。",
        "- 即使门通过，结果仍是 L0 诊断，不能声明真实数据集 utility、量子优势、L1 或 L2。",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace, *, input_path: Path, input_source: Mapping[str, Any]) -> Path:
    phase = _validate_config(args.config.resolve(), args.plan.resolve())
    bundle = _load_json(input_path)
    cases = _validate_input(bundle)
    from applications.rag.f50_qwi_selector import F50QWIConfig, mechanism_audit
    from applications.rag.data import make_corpus_manager
    from applications.rag.retrieval import make_encoder

    mechanism = phase["mechanism"]
    config = F50QWIConfig(
        diagonal_strength=float(mechanism["diagonal_strength"]),
        interaction_strength=float(mechanism["interaction_strength_kappa"]),
        evolution_time=float(mechanism["evolution_time_tau"]),
        initial_temperature=1.0,
        retrieval_weight=float(mechanism["retrieval_weight"]),
        answer_scorer_weight=float(mechanism["answer_scorer_weight"]),
        phase_seed=int(mechanism["phase_seed"]),
    )
    encoder = make_encoder("dpr")
    manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    manager.build([])

    method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in ("qore_as", "topk_as", *phase["controls"])}
    traces: list[dict[str, Any]] = []
    audit_failures: list[dict[str, Any]] = []
    control_difference_counts = {key: 0 for key in phase["controls"] if key != "born_exact"}
    retrieval_miss_cases = 0
    max_score_error = 0.0
    total_selector_seconds = 0.0
    started = time.perf_counter()
    for case_number, case in enumerate(cases, start=1):
        question = str(case["question"])
        query_embedding = encoder.encode_queries([question])[0]
        records, embeddings = _retrieve(manager, query_embedding)
        validation = _validate_retrieval(case, records)
        max_score_error = max(max_score_error, float(validation["max_retrieval_score_abs_error"]))
        raw = [_raw_candidate(item) for item in case["top_50"]]
        online = _online_candidates(raw, embeddings)
        selection_started = time.perf_counter()
        audit = mechanism_audit(online, config=config)
        elapsed_ms = (time.perf_counter() - selection_started) * 1000.0
        total_selector_seconds += elapsed_ms / 1000.0
        if not audit.passed:
            audit_failures.append({"case_number": case_number, "checks": audit.checks})
            raise ReplayError(f"F50-QWI mechanism audit failed for case {case_number}; Silver was not read")
        born_indices = audit.variants["born_exact"].selected_indices
        for control in control_difference_counts:
            control_difference_counts[control] += int(born_indices != audit.variants[control].selected_indices)

        # Only after the label-free audit and all four selections have completed.
        oracle_ids = set(_stored_ids(case, "silver_oracle_common_order"))
        if not any(_evidence_flags(item)[0] for item in raw):
            retrieval_miss_cases += 1
        embeddings_by_id = {str(record["id"]): [float(value) for value in embeddings[index]] for index, record in enumerate(records)}
        baseline_ids = {
            "qore_as": _stored_ids(case, "qore_common_order"),
            "topk_as": _stored_ids(case, "topk_common_order"),
        }
        for method, selected_ids in baseline_ids.items():
            row = _build_row(raw, selected_ids, oracle_ids, embeddings_by_id, elapsed_ms=0.0, diagnostics={"source": f"stored_{method}"})
            row["case_number"] = case_number
            method_rows[method].append(row)
        trace_methods: dict[str, Any] = {method: {"selected_ids": ids} for method, ids in baseline_ids.items()}
        for variant, result in audit.variants.items():
            selected_ids = [str(raw[index]["id"]) for index in result.selected_indices]
            row = _build_row(raw, selected_ids, oracle_ids, embeddings_by_id, elapsed_ms=elapsed_ms / len(audit.variants), diagnostics=result.diagnostics)
            row["case_number"] = case_number
            method_rows[variant].append(row)
            trace_methods[variant] = {
                "selected_ids": selected_ids,
                "selected_ranks": row["selected_ranks"],
                "selected_probability_sha256": hashlib.sha256("\n".join(f"{value:.15g}" for value in result.probabilities).encode("utf-8")).hexdigest(),
                "numerical": result.diagnostics,
            }
        traces.append({
            "case_number": case_number,
            "source_id": str(case.get("source_id") or case.get("question_id") or ""),
            "question_id": str(case.get("question_id") or ""),
            "top50_candidate_id_sha256": hashlib.sha256("\n".join(str(item["id"]) for item in raw).encode("utf-8")).hexdigest(),
            "retrieval_validation": validation,
            "mechanism_checks": audit.checks,
            "methods": trace_methods,
        })
        if args.progress and (case_number == EXPECTED_CASES or case_number % 10 == 0):
            print(f"  F50-QWI replay: {case_number}/{EXPECTED_CASES}", flush=True)

    summaries = {method: _summary(rows, method, "registered_baseline" if method in {"qore_as", "topk_as"} else "f50_control") for method, rows in method_rows.items()}
    paired = {method: {"vs_qore": _paired(method_rows, method, "qore_as"), "vs_topk": _paired(method_rows, method, "topk_as")} for method in phase["controls"]}
    born_mean = float(summaries["born_exact"]["mean_silver_oracle_overlap"])
    mechanism_pass = not audit_failures
    cpu_pass = total_selector_seconds < float(phase["gates"]["max_total_cpu_seconds"])
    born_beats_topk = born_mean > float(phase["gates"]["strict_born_mean_silver_overlap_gt"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root.resolve() if args.output_root else DEFAULT_OUTPUT_ROOT
    output_dir = _unique_output_dir(output_root, timestamp)
    target_directory = f"five_ideas/f50_qwi_selector_replay_100/{output_dir.name}"
    report = {
        "schema_version": "rag.f50_qwi_selector_replay_100.v1", "artifact_type": "f50_qwi_selector_l0_replay", "diagnostic_only": True,
        "input": {"path_name": input_path.name, "sha256": _sha256(input_path), **dict(input_source)},
        "protocol": {"case_count": EXPECTED_CASES, "top50_count": EXPECTED_TOP50, "k": K, "online_fields": ["id", "retrieved_rank", "retrieval_score", "answer_scorer_score", "dpr_passage_embedding"], "silver_oracle_used_for_selection": False, "generator_called": False, "evaluator_called": False, "retrieval": dict(phase["retrieval"]), "mechanism": dict(mechanism), "control_allowlist": list(phase["controls"])},
        "provenance": {"git_revision": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"), "script": str(SCRIPT_PATH.relative_to(ROOT)), "script_sha256": _sha256(SCRIPT_PATH), "selector_module": "applications/rag/f50_qwi_selector.py", "selector_module_sha256": _sha256(ROOT / "applications/rag/f50_qwi_selector.py"), "config_path": str(args.config.resolve().relative_to(ROOT)), "config_sha256": _sha256(args.config.resolve()), "plan_path": str(args.plan.resolve().relative_to(ROOT)), "plan_sha256": _sha256(args.plan.resolve())},
        "summaries": summaries, "paired_comparisons": paired,
        "diagnostics": {"exact_top50_identity_validated": True, "max_retrieval_score_abs_error": max_score_error, "retrieval_miss_cases": retrieval_miss_cases, "control_selection_difference_cases": control_difference_counts, "audit_failures": audit_failures},
        "gates": {"mechanism_audit_pass": mechanism_pass, "cpu_budget_pass": cpu_pass, "total_selector_seconds": total_selector_seconds, "max_total_cpu_seconds": phase["gates"]["max_total_cpu_seconds"], "born_mean_silver_overlap": born_mean, "strict_topk_threshold": phase["gates"]["strict_born_mean_silver_overlap_gt"], "born_strictly_beats_topk": born_beats_topk, "next_complete_selector_replay_authorized": bool(mechanism_pass and cpu_pass and born_beats_topk)},
        "claim_ceiling": "L0 diagnostic only: fixed 100-case Silver alignment, no utility or quantum-advantage claim.",
        "next_step": "If and only if all listed gates pass, the user-authorized next complete selector replay may be prepared; do not hand off a full-data run or claim L1/L2 from this result.",
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    _write_json(output_dir / "selector_trace.json", {"schema_version": "rag.f50_qwi_selector_replay_100.trace.v1", "diagnostic_only": True, "input": report["input"], "provenance": report["provenance"], "target_directory": target_directory, "cases": traces})
    metadata = {"schema_version": "rag.f50_qwi_selector_replay_100.metadata.v1", "artifact_type": "f50_qwi_selector_run_metadata", "status": "completed", "diagnostic_only": True, "generated_at_utc": timestamp, "input": report["input"], "provenance": report["provenance"], "protocol": report["protocol"], "gates": report["gates"], "outputs": {"root": str(output_root.relative_to(ROOT)), "target_directory": target_directory, "exchange_files": ["selector_trace.json"]}, "timing_ms": {"total": (time.perf_counter() - started) * 1000.0}}
    _write_json(output_dir / "run_metadata.json", metadata)
    manifest = {"schema_version": "rag.f50_qwi_selector_replay_100.upload_manifest.v1", "artifact_type": "f50_qwi_selector_upload_manifest", "status": "ready_for_authenticated_exchange_upload", "target_directory": target_directory, "generated_at_utc": timestamp, "exchange_files": [{"name": "selector_trace.json", "exchange_path": f"{target_directory}/selector_trace.json"}], "compact_files": [{"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": _sha256(output_dir / name)} for name in ("summary.json", "report.md", "run_metadata.json")], "privacy": {"raw_passage_text_in_compact": False, "gold_answers_in_compact": False, "silver_labels_used_online": False, "generator_outputs": False}, "github_policy": "Commit compact files only when <=1 MiB and privacy-checked; selector_trace.json is exchange-only."}
    _write_json(output_dir / "upload_manifest.json", manifest)
    if not args.no_upload:
        from scripts.collab.lib.exchange_upload import upload_manifest
        metadata["exchange_upload"] = {"status": "uploaded", "files": upload_manifest(output_dir / "upload_manifest.json")}
        _write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps({"output_dir": str(output_dir), "target_directory": target_directory, "next_complete_selector_replay_authorized": report["gates"]["next_complete_selector_replay_authorized"]}, ensure_ascii=False))
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="optional registered 100-case detail JSON; default downloads it from exchange")
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
        value = getattr(args, field)
        if not value.is_absolute():
            setattr(args, field, ROOT / value)
    try:
        input_path, temporary_dir, input_source = _prepare_input(args)
        try:
            if args.validate_only:
                phase = _validate_config(args.config.resolve(), args.plan.resolve())
                cases = _validate_input(_load_json(input_path))
                print(json.dumps({"status": "valid", "phase": phase["name"], "case_count": len(cases), "top50_count": EXPECTED_TOP50, "input": {"path_name": input_path.name, **input_source}}, ensure_ascii=False))
                return 0
            run(args, input_path=input_path, input_source=input_source)
        finally:
            if temporary_dir is not None:
                temporary_dir.cleanup()
    except (ReplayError, ValueError, OSError) as exc:
        print(f"F50-QWI replay error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
