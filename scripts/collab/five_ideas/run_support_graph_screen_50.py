#!/usr/bin/env python3
"""Run a private, content-complete fresh-50 support-graph selector screen.

The runner is observation-only.  It retrieves and scores each fixed question
once, fans the same Top-50 through an explicit selector allowlist, and sends
each returned Top-5 to the same frozen Generator.  Complete content is kept
in the exchange artifact; compact provenance is safe for GitHub.
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


class CaseStudyError(RuntimeError):
    """Raised when the frozen case-study contract cannot be satisfied."""


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
CASE_COUNT = 50
EXPECTED_SAMPLE_OFFSET = 3150
EXPECTED_POSITIONS = tuple(range(CASE_COUNT))
EXPECTED_GLOBAL_INDICES = tuple(range(EXPECTED_SAMPLE_OFFSET, EXPECTED_SAMPLE_OFFSET + CASE_COUNT))
EXPECTED_SELECTOR_IDS = (
    "qore_as",
    "topk_as",
    "anchor_conditioned_support_graph",
    "support_disabled_control",
)
EXPECTED_SELECTOR_SPECS = (
    {
        "id": "qore_as", "method": "qore", "K": 5, "num_reads": 100,
        "lam": 2.0, "seed": 42, "gamma": 1.0, "delta": 0.0,
        "complementarity_method": None, "direct_solve_max_n": 20,
        "qore_prefilter_size": None,
    },
    {"id": "topk_as", "method": "topk", "K": 5},
    {
        "id": "anchor_conditioned_support_graph", "method": "support_graph", "K": 5,
        "anchor_count": 2, "support_passage_weight": 0.70,
        "support_question_weight": 0.30,
    },
    {"id": "support_disabled_control", "method": "support_graph", "K": 5,
     "anchor_count": 2, "support_passage_weight": 0.0,
     "support_question_weight": 0.0,
     "control_of": "anchor_conditioned_support_graph",
     "support_objective_enabled": False},
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


def _root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise CaseStudyError("cannot locate project root")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CaseStudyError(f"cannot read config: {exc}") from exc
    if not isinstance(document, dict):
        raise CaseStudyError("config root must be a mapping")
    return document


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseStudyError(f"cannot read plugin plan: {exc}") from exc
    if not isinstance(document, dict):
        raise CaseStudyError("plugin plan root must be a mapping")
    return document


def _selector_contracts(phase: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    selectors = phase.get("selectors")
    if not isinstance(selectors, list) or len(selectors) != len(EXPECTED_SELECTOR_IDS):
        raise CaseStudyError("selector allowlist must contain exactly four selectors")
    if tuple(item.get("id") for item in selectors if isinstance(item, dict)) != EXPECTED_SELECTOR_IDS:
        raise CaseStudyError("selector order or ids are not frozen")
    normalized = tuple(dict(item) for item in selectors if isinstance(item, dict))
    if len(normalized) != len(EXPECTED_SELECTOR_SPECS):
        raise CaseStudyError("each selector entry must be a mapping")
    for observed, expected in zip(normalized, EXPECTED_SELECTOR_SPECS):
        if observed != expected:
            raise CaseStudyError(f"selector specification is not frozen for {expected['id']}")
    return normalized


def validate_contract(config_path: Path, plan_path: Path) -> dict[str, Any]:
    """Validate configuration and plan without datasets, models, or GPU use."""

    document = _load_yaml(config_path)
    phase = document.get("phase")
    if not isinstance(phase, dict):
        raise CaseStudyError("phase section is missing")
    if phase.get("name") != "support_graph_screen_50" or int(phase.get("schema_version", -1)) != 1:
        raise CaseStudyError("unexpected case-study config")
    if phase.get("authorization") != "implemented" or phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise CaseStudyError("case study must remain observation-only")
    dataset = phase.get("dataset")
    if not isinstance(dataset, dict) or (dataset.get("name"), dataset.get("split")) != ("nq_open", "validation"):
        raise CaseStudyError("dataset identity is not frozen")
    parent = dataset.get("parent_slice")
    if not isinstance(parent, dict) or (int(parent.get("sample_offset", -1)), int(parent.get("max_samples", -1))) != (EXPECTED_SAMPLE_OFFSET, CASE_COUNT):
        raise CaseStudyError("parent slice must be validation[3150:3200]")
    positions = tuple(int(value) for value in dataset.get("selected_relative_positions", ()))
    globals_ = tuple(int(value) for value in dataset.get("selected_global_indices", ()))
    if positions != EXPECTED_POSITIONS or globals_ != EXPECTED_GLOBAL_INDICES:
        raise CaseStudyError("fixed case-study question positions changed")
    retrieval = phase.get("retrieval")
    if retrieval != {
        "corpus_mode": "wiki_dpr",
        "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64,
        "initial_top_k": 50,
        "one_retrieval_per_question": True,
    }:
        raise CaseStudyError("retrieval contract is not frozen")
    scorer = phase.get("answer_scorer")
    if scorer != {"backend": "dpr", "one_scoring_pass_per_question": True}:
        raise CaseStudyError("answer scorer contract is not frozen")
    selectors = _selector_contracts(phase)
    generator = phase.get("generator")
    if not isinstance(generator, dict) or generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION:
        raise CaseStudyError("generator identity is not frozen")
    if int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy" or generator.get("use_chat_template") is not True:
        raise CaseStudyError("generator decoding contract is not frozen")
    outputs = phase.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("root") != "exchange/five_ideas/support_graph_screen_50":
        raise CaseStudyError("output root is not frozen")
    if outputs.get("complete_files_exchange_only") is not True:
        raise CaseStudyError("complete files must be exchange-only")
    if outputs.get("complete_files") != ["case_study.md", "case_study.json"] or outputs.get("compact_files") != ["summary.json", "run_metadata.json", "upload_manifest.json"]:
        raise CaseStudyError("output file contract is not frozen")
    if set(outputs.get("forbidden_compact_fields", ())) != FORBIDDEN_COMPACT_FIELDS:
        raise CaseStudyError("compact privacy fields are not frozen")
    plan = _load_plan(plan_path)
    if plan.get("schema_version") not in {"research-plugin-architecture.plugin-plan.v1", "research-plugin-architecture.plugin-plan.v2"} or plan.get("project") != "Q-DUET-VLM" or plan.get("authorization") != "implemented":
        raise CaseStudyError("plugin plan identity or authorization is invalid")
    discovery = plan.get("discovery")
    composition = plan.get("composition")
    expected_plugins = (
        "frozen_retrieval_observer",
        "allowlisted_selector_comparator",
        "frozen_generator_case_observer",
        "case_study_renderer",
    )
    if not isinstance(discovery, dict) or discovery.get("mode") != "explicit_allowlist" or tuple(discovery.get("allowlist", ())) != expected_plugins:
        raise CaseStudyError("plugin discovery must be explicit and allowlisted")
    if not isinstance(composition, dict) or composition.get("mode") != "sequential" or tuple(composition.get("order", ())) != expected_plugins:
        raise CaseStudyError("plugin composition order is not frozen")
    if plan.get("reproducibility", {}).get("config_path") != "configs/experiments/support_graph_screen_50.yaml":
        raise CaseStudyError("plugin plan config path is not frozen")
    return {
        "status": "valid",
        "phase": phase["name"],
        "dataset": "nq_open validation[3150:3200] all 50 positions",
        "selected_global_indices": list(EXPECTED_GLOBAL_INDICES),
        "selectors": [item["id"] for item in selectors],
        "selection_mutation": False,
        "model_loaded": False,
        "wiki_dpr_started": False,
    }


def _resolve_generator(root: Path, specification: Mapping[str, Any], override: str | None) -> dict[str, str]:
    candidates: list[tuple[Path, str]] = []
    if override:
        candidates.append((Path(os.path.expanduser(os.path.expandvars(override))), "cli_override"))
    for raw in specification.get("model_path_candidates", []):
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append((candidate if candidate.is_absolute() else root / candidate, "project_candidate"))
    candidates.append((Path.home() / ".cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct", "hf_cache"))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append((Path(hf_home) / "hub/models--NousResearch--Meta-Llama-3-8B-Instruct", "hf_cache"))
    attempted: list[str] = []
    for candidate, resolution in candidates:
        attempted.append(str(candidate))
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate if (candidate / "config.json").is_file() else snapshot
        if (resolved / "config.json").is_file():
            return {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_path": str(resolved.resolve()),
                "config_sha256": _sha256(resolved / "config.json"),
                "resolution": resolution,
            }
    raise CaseStudyError("reference generator not found; attempted: " + ", ".join(attempted))


def _retrieve_with_metadata(manager: Any, query_embedding: np.ndarray, top_k: int) -> dict[str, Any]:
    """Query Wiki-DPR once and retain its global metadata without changing API."""

    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise CaseStudyError("wiki_dpr manager does not expose its retrieval dataset")
    scores, retrieved = dataset.get_nearest_examples("embeddings", np.asarray(query_embedding, dtype=np.float32), k=top_k)
    required = ("id", "title", "text", "embeddings")
    if any(key not in retrieved for key in required):
        raise CaseStudyError("Wiki-DPR retrieval result lacks id/title/text/embeddings")
    count = len(retrieved["id"])
    if count != top_k or len(scores) != top_k:
        raise CaseStudyError(f"expected {top_k} retrieved passages, got {count}")
    records: list[dict[str, Any]] = []
    embeddings = np.asarray(retrieved["embeddings"], dtype=np.float32)
    if embeddings.shape[0] != top_k:
        raise CaseStudyError("retrieved embedding count mismatch")
    for rank in range(top_k):
        title = str(retrieved["title"][rank] or "")
        body = str(retrieved["text"][rank] or "")
        records.append({
            "id": str(retrieved["id"][rank]),
            "title": title,
            "text": body,
            "passage": f"{title}. {body}" if title else body,
            "retrieved_rank": rank + 1,
            "retrieval_score": float(scores[rank]),
        })
    return {"records": records, "embeddings": embeddings, "retrieval_scores": np.asarray(scores, dtype=np.float32)}


def _selector_kwargs(spec: Mapping[str, Any], query_embedding: np.ndarray, embeddings: np.ndarray, texts: list[str], question: str, scorer: Any, scores: np.ndarray) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "query_embedding": query_embedding,
        "passage_embeddings": embeddings,
        "K": int(spec["K"]),
        "method": str(spec["method"]),
        "relevance_scores": scores,
        "seed": 42,
    }
    method = str(spec["method"])
    if method == "qore":
        kwargs.update({
            "num_reads": int(spec["num_reads"]),
            "lam": float(spec["lam"]),
            "gamma": float(spec["gamma"]),
            "delta": float(spec["delta"]),
            "complementarity_method": spec.get("complementarity_method"),
            "direct_solve_max_n": int(spec["direct_solve_max_n"]),
            "qore_prefilter_size": spec.get("qore_prefilter_size"),
            "answer_scorer": scorer,
            "passage_texts": texts,
            "question": question,
        })
    elif method == "mmr":
        kwargs["lambda_mmr"] = float(spec["lambda_mmr"])
    elif method == "submodular":
        kwargs["saturation_alpha"] = float(spec["saturation_alpha"])
        kwargs["lambda_submodular"] = float(spec["lambda_submodular"])
    elif method == "spectral_dpp":
        kwargs["dpp_quality_scale"] = float(spec["dpp_quality_scale"])
        kwargs["dpp_jitter"] = float(spec["dpp_jitter"])
    return kwargs


def _select_one(
    spec: Mapping[str, Any],
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    question: str,
    scorer: Any,
    scores: np.ndarray,
) -> tuple[list[int], list[dict[str, Any]]]:
    if str(spec["method"]) == "support_graph":
        from applications.rag.support_graph_selector import select

        online = [
            {
                "id": str(record["id"]),
                "text": str(record["passage"]),
                "retrieved_rank": int(record["retrieved_rank"]),
                "retrieval_score": float(record["retrieval_score"]),
                "answer_scorer_score": float(scores[index]),
            }
            for index, record in enumerate(records)
        ]
        selection = select(
            question,
            online,
            k=int(spec["K"]),
            anchor_count=int(spec["anchor_count"]),
            support_passage_weight=float(spec["support_passage_weight"]),
            support_question_weight=float(spec["support_question_weight"]),
        )
        indices = [int(value) for value in selection.selected_indices]
        trace = [dict(item) for item in selection.support_trace]
    else:
        from applications.rag.selector import select_passages

        texts = [str(record["passage"]) for record in records]
        selected = np.asarray(select_passages(**_selector_kwargs(spec, query_embedding, embeddings, texts, question, scorer, scores))).reshape(-1)
        indices = [int(value) for value in selected]
        trace = []
    if len(indices) != 5 or len(set(indices)) != 5 or any(index < 0 or index >= len(records) for index in indices):
        raise CaseStudyError(f"selector {spec['id']} did not return five unique valid candidates")
    return indices, trace


def _answer_proxy(answers: Sequence[str], records: Sequence[Mapping[str, Any]]) -> bool:
    from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text

    return any(answer_has_match_in_text(str(answer), str(record["passage"])) for answer in answers for record in records)


def _selector_record(
    spec: Mapping[str, Any],
    indices: Sequence[int],
    records: Sequence[Mapping[str, Any]],
    answer_scores: np.ndarray,
    prediction: str,
    answers: Sequence[str],
    support_trace: Sequence[Mapping[str, Any]] = (),
    elapsed_ms: float = 0.0,
) -> dict[str, Any]:
    from applications.rag.evaluation import evaluate_answer

    selected = []
    for index in indices:
        record = dict(records[index])
        record["answer_scorer_score"] = float(answer_scores[index])
        selected.append(record)
    metrics = evaluate_answer(prediction, list(answers))
    return {
        "selector_id": str(spec["id"]),
        "method": str(spec["method"]),
        "selection_config": {key: value for key, value in spec.items() if key not in {"id", "method", "K"}},
        "selected_top_5": selected,
        "prediction": prediction,
        "metrics": {"em": float(metrics["em"]), "f1": float(metrics["f1"])},
        "timing_ms": {"selection_and_generation": float(elapsed_ms)},
        "diagnostics": {
            "answer_has_match_in_selected_text": _answer_proxy(answers, selected),
            "answer_string_match_is_proxy_not_strict_gold": True,
            "selected_retrieved_ranks": [int(item["retrieved_rank"]) for item in selected],
            "support_graph": {
                "enabled": str(spec["method"]) == "support_graph" and (
                    float(spec.get("support_passage_weight", 0.0)) > 0.0
                    or float(spec.get("support_question_weight", 0.0)) > 0.0
                ),
                "anchor_count": int(spec.get("anchor_count", 0)),
                "anchor_ids": [str(records[index]["id"]) for index in indices[: int(spec.get("anchor_count", 0))]]
                if str(spec["method"]) == "support_graph" else [],
                "support_trace": [dict(item) for item in support_trace],
                "control_of": spec.get("control_of"),
                "support_objective_enabled": spec.get("support_objective_enabled"),
            },
        },
    }


def _intersection_diagnostics(selector_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sets = {name: {str(item["id"]) for item in record["selected_top_5"]} for name, record in selector_records.items()}
    names = list(sets)
    pairwise: dict[str, dict[str, int]] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "intersection": len(sets[left] & sets[right]),
                "symmetric_difference": len(sets[left] ^ sets[right]),
            }
    return {"pairwise": pairwise, "unique_ids_across_selectors": len(set().union(*sets.values())) if sets else 0}


def _build_markdown(case_study: Mapping[str, Any]) -> str:
    lines = [
        "# RAG Selector Case Study",
        "",
        "> Private, observation-only diagnostic. Fifty fixed NQ-Open validation questions; this is not a full-population comparison.",
        "",
        f"- Question slice: `{case_study['dataset_slice']}`",
        f"- Retriever: `{case_study['retrieval']['corpus_mode']}` / `{case_study['retrieval']['wiki_dpr_config']}`",
        f"- Generator: `{case_study['generator']['model_id']}` revision `{case_study['generator']['revision']}`",
        "- Answer-string-in-text fields are proxies, not strict gold-passage labels.",
        "",
    ]
    for case in case_study["cases"]:
        lines.extend([f"## Case {case['case_number']}: {case['question_id']}", "", f"**Question:** {case['question']}", "", "**Gold answers:** " + "; ".join(case["gold_answers"]), "", "### Shared Top-50"])
        for passage in case["top_50"]:
            lines.extend([
                f"{passage['retrieved_rank']}. **{passage['title'] or '(untitled)'}** (id `{passage['id']}`, retrieval score `{passage['retrieval_score']:.6f}`, answer-score `{passage['answer_scorer_score']:.6f}`)",
                passage["text"],
                "",
            ])
        lines.append("### Selector Outputs")
        for selector in case["selectors"]:
            metrics = selector["metrics"]
            lines.extend([
                "",
                f"#### {selector['selector_id']} (`{selector['method']}`)",
                f"Prediction: `{selector['prediction']}`",
                f"EM: `{metrics['em']:.3f}`; F1: `{metrics['f1']:.3f}`; answer-string proxy hit: `{selector['diagnostics']['answer_has_match_in_selected_text']}`",
                "",
            ])
            for rank, passage in enumerate(selector["selected_top_5"], start=1):
                lines.extend([
                    f"{rank}. **{passage['title'] or '(untitled)'}** (retrieved rank `{passage['retrieved_rank']}`, id `{passage['id']}`, answer-score `{passage['answer_scorer_score']:.6f}`)",
                    passage["text"],
                    "",
                ])
            support_graph = selector.get("diagnostics", {}).get("support_graph", {})
            if support_graph.get("anchor_count", 0) or support_graph.get("support_trace"):
                lines.extend([
                    "**Support-graph trace**",
                    f"- enabled: `{support_graph.get('enabled')}`; anchors: "
                    + ", ".join(f"`{value}`" for value in support_graph.get("anchor_ids", [])),
                    "",
                ])
                for step in support_graph.get("support_trace", []):
                    lines.append(
                        f"- step {step['step']}: `{step['candidate_id']}`; total `{step['support_score']:.6f}`; "
                        f"passage overlap `{step['passage_overlap_mean']:.6f}`; "
                        f"question overlap `{step['question_overlap_mean']:.6f}`"
                    )
                lines.append("")
        lines.extend(["### Selector Set Diagnostics", "", "```json", json.dumps(case["selector_set_diagnostics"], indent=2, ensure_ascii=False), "```", ""])
    return "\n".join(lines)


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


def _case_study_fingerprints(run_dir: Path, target_dir: str) -> dict[str, Any]:
    files = []
    for name in ("case_study.md", "case_study.json"):
        path = run_dir / name
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path), "exchange_path": f"{target_dir}/{name}"})
    return {"files": files}


def _load_runtime_components(root: Path, phase: Mapping[str, Any], model_path_override: str | None):
    from applications.rag.answer_scorer import make_answer_scorer
    from applications.rag.data import load_dataset_for_rag, make_corpus_manager
    from applications.rag.generation import Generator
    from applications.rag.retrieval import make_encoder

    dataset = phase["dataset"]
    parent = dataset["parent_slice"]
    end = int(parent["sample_offset"]) + int(parent["max_samples"])
    all_questions = load_dataset_for_rag(dataset["name"], dataset["split"], end)
    if len(all_questions) != end:
        raise CaseStudyError(f"expected at least {end} dataset questions, got {len(all_questions)}")
    positions = tuple(int(value) for value in dataset["selected_relative_positions"])
    questions = [all_questions[int(parent["sample_offset"]) + position] for position in positions]
    encoder = make_encoder("dpr")
    manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    manager.build(questions)
    scorer = make_answer_scorer(backend="dpr")
    identity = _resolve_generator(root, phase["generator"], model_path_override)
    generator = Generator(identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    return questions, encoder, manager, scorer, generator, identity


def _bootstrap_lower(values: Sequence[float], *, seed: int, reps: int = 2000) -> float:
    """Return a deterministic percentile lower bound for paired differences."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(reps, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025))


def _screen_summary(cases: Sequence[Mapping[str, Any]], *, contract: Mapping[str, Any]) -> dict[str, Any]:
    selector_ids = list(EXPECTED_SELECTOR_IDS)
    by_selector: dict[str, list[Mapping[str, Any]]] = {selector_id: [] for selector_id in selector_ids}
    for case in cases:
        by_selector.update({
            str(item["selector_id"]): by_selector[str(item["selector_id"])] + [item]
            for item in case["selectors"]
        })
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "support_graph_screen_50_summary",
        "diagnostic_only": True,
        "case_count": len(cases),
        "selector_summaries": {},
        "paired_deltas": {},
        "gates": {},
        "limitations": [
            "This fresh 50-question screen is an L0 diagnostic screen, not population utility evidence.",
            "answer-string-in-text is a proxy and not a strict gold passage label.",
            "No silver panel, gold passage, generation output, or evaluator field is used online by selection.",
        ],
    }
    for selector_id in selector_ids:
        rows = by_selector[selector_id]
        em = [float(row["metrics"]["em"]) for row in rows]
        f1 = [float(row["metrics"]["f1"]) for row in rows]
        proxy_selected = [bool(row["diagnostics"]["answer_has_match_in_selected_text"]) for row in rows]
        proxy_top50 = [bool(case["top_50_diagnostics"]["answer_has_match_in_text"]) for case in cases]
        summary["selector_summaries"][selector_id] = {
            "case_count": len(rows),
            "mean_em": round(float(np.mean(em)), 6),
            "mean_f1": round(float(np.mean(f1)), 6),
            "answer_string_proxy_top50_hit_cases": int(sum(proxy_top50)),
            "answer_string_proxy_selected_hit_cases": int(sum(proxy_selected)),
            "answer_string_proxy_selector_miss_cases": int(sum(top and not selected for top, selected in zip(proxy_top50, proxy_selected))),
            "mean_selection_and_generation_ms": round(float(np.mean([float(row["timing_ms"]["selection_and_generation"]) for row in rows])), 3),
            "selected_id_order_valid": all(len(row["selected_top_5"]) == 5 and len({str(item["id"]) for item in row["selected_top_5"]}) == 5 for row in rows),
        }
    reference = by_selector["qore_as"]
    support = by_selector["anchor_conditioned_support_graph"]
    control = by_selector["support_disabled_control"]
    topk = by_selector["topk_as"]
    reference_f1 = np.asarray([float(row["metrics"]["f1"]) for row in reference])
    support_f1 = np.asarray([float(row["metrics"]["f1"]) for row in support])
    control_f1 = np.asarray([float(row["metrics"]["f1"]) for row in control])
    reference_em = np.asarray([float(row["metrics"]["em"]) for row in reference])
    support_em = np.asarray([float(row["metrics"]["em"]) for row in support])
    summary["paired_deltas"] = {
        "support_graph_minus_qore": {
            "em": round(float(np.mean(support_em - reference_em)), 6),
            "f1": round(float(np.mean(support_f1 - reference_f1)), 6),
            "f1_ci95_lower": round(_bootstrap_lower(support_f1 - reference_f1, seed=20260908), 6),
        },
        "support_graph_minus_support_disabled": {
            "f1": round(float(np.mean(support_f1 - control_f1)), 6),
            "f1_ci95_lower": round(_bootstrap_lower(support_f1 - control_f1, seed=20260909), 6),
        },
    }
    summary["control_parity"] = {
        "support_disabled_matches_topk_order": all(
            [str(item["id"]) for item in control_row["selected_top_5"]]
            == [str(item["id"]) for item in topk_row["selected_top_5"]]
            for control_row, topk_row in zip(control, topk)
        ),
        "support_disabled_matches_topk_set": all(
            {str(item["id"]) for item in control_row["selected_top_5"]}
            == {str(item["id"]) for item in topk_row["selected_top_5"]}
            for control_row, topk_row in zip(control, topk)
        ),
    }
    support_mean_ms = summary["selector_summaries"]["anchor_conditioned_support_graph"]["mean_selection_and_generation_ms"]
    qore_mean_ms = summary["selector_summaries"]["qore_as"]["mean_selection_and_generation_ms"]
    cost_ratio = support_mean_ms / qore_mean_ms if qore_mean_ms > 0 else float("inf")
    summary["cost"] = {"support_graph_to_qore_mean_time_ratio": round(float(cost_ratio), 6)}
    support_proxy = summary["selector_summaries"]["anchor_conditioned_support_graph"]
    qore_proxy = summary["selector_summaries"]["qore_as"]
    deltas = summary["paired_deltas"]
    summary["gates"] = {
        "contract_gate": {"pass": bool(contract.get("status") == "valid")},
        "selected_id_order_gate": {"pass": all(value["selected_id_order_valid"] for value in summary["selector_summaries"].values())},
        "support_graph_vs_qore_f1_gate": {
            "pass": deltas["support_graph_minus_qore"]["f1"] >= 0.01 and deltas["support_graph_minus_qore"]["f1_ci95_lower"] >= 0.0,
            "delta": deltas["support_graph_minus_qore"]["f1"],
            "ci95_lower": deltas["support_graph_minus_qore"]["f1_ci95_lower"],
            "required_delta": 0.01,
        },
        "no_em_drop_gate": {
            "pass": deltas["support_graph_minus_qore"]["em"] >= 0.0,
            "delta": deltas["support_graph_minus_qore"]["em"],
        },
        "support_vs_disabled_gate": {
            "pass": deltas["support_graph_minus_support_disabled"]["f1_ci95_lower"] > 0.0,
            "ci95_lower": deltas["support_graph_minus_support_disabled"]["f1_ci95_lower"],
        },
        "proxy_retention_direction_gate": {
            "pass": support_proxy["answer_string_proxy_selected_hit_cases"] >= qore_proxy["answer_string_proxy_selected_hit_cases"],
            "support_graph_selected_hit_cases": support_proxy["answer_string_proxy_selected_hit_cases"],
            "qore_selected_hit_cases": qore_proxy["answer_string_proxy_selected_hit_cases"],
        },
        "cost_gate": {"pass": cost_ratio <= 1.5, "ratio": round(float(cost_ratio), 6), "max_ratio": 1.5},
    }
    summary["decision"] = "pass_fresh_screen_gate" if all(gate["pass"] for gate in summary["gates"].values()) else "stop_candidate_after_fresh_screen"
    return summary


def run(args: argparse.Namespace) -> Path | None:
    root = _root()
    config_path = (args.config if args.config.is_absolute() else root / args.config).resolve()
    plan_path = (args.plan if args.plan.is_absolute() else root / args.plan).resolve()
    contract = validate_contract(config_path, plan_path)
    if args.validate_only:
        print(json.dumps(contract, sort_keys=True))
        return None
    phase = _load_yaml(config_path)["phase"]
    selectors = _selector_contracts(phase)
    questions, encoder, manager, scorer, generator, identity = _load_runtime_components(root, phase, args.model_path)
    if len(questions) != CASE_COUNT:
        raise CaseStudyError("fixed case-study question count mismatch")
    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    target_dir = f"five_ideas/support_graph_screen_50/{run_dir.name}"
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for case_number, question_item in enumerate(questions, start=1):
        question_id = str(question_item["id"])
        question = str(question_item["question"])
        answers = [str(answer) for answer in question_item.get("answers", [])]
        query_embedding = encoder.encode_queries([question])[0]
        retrieval = _retrieve_with_metadata(manager, query_embedding, 50)
        records = retrieval["records"]
        texts = [str(record["passage"]) for record in records]
        answer_scores = np.asarray(scorer.score_passages(question, texts), dtype=np.float32)
        if answer_scores.shape != (50,):
            raise CaseStudyError("DPR Answer Scorer did not return 50 scores")
        top_50 = []
        for index, record in enumerate(records):
            item = dict(record)
            item["answer_scorer_score"] = float(answer_scores[index])
            top_50.append(item)
        selector_records: dict[str, dict[str, Any]] = {}
        for spec in selectors:
            arm_started = time.perf_counter()
            indices, support_trace = _select_one(spec, query_embedding, retrieval["embeddings"], records, question, scorer, answer_scores)
            selected_texts = [texts[index] for index in indices]
            prediction = str(generator.generate(question, selected_texts))
            selector_records[str(spec["id"])] = _selector_record(
                spec, indices, records, answer_scores, prediction, answers,
                support_trace=support_trace,
                elapsed_ms=(time.perf_counter() - arm_started) * 1000.0,
            )
        topk_ids = [str(item["id"]) for item in selector_records["topk_as"]["selected_top_5"]]
        control_ids = [str(item["id"]) for item in selector_records["support_disabled_control"]["selected_top_5"]]
        control_matches_topk = control_ids == topk_ids
        cases.append({
            "case_number": case_number,
            "question_id": question_id,
            "question": question,
            "gold_answers": answers,
            "top_50": top_50,
            "top_50_diagnostics": {
                "answer_has_match_in_text": _answer_proxy(answers, top_50),
                "answer_string_match_is_proxy_not_strict_gold": True,
            },
            "selectors": list(selector_records.values()),
            "selector_set_diagnostics": _intersection_diagnostics(selector_records),
            "control_diagnostics": {
                "support_disabled_matches_topk_order": control_matches_topk,
                "support_disabled_matches_topk_set": set(control_ids) == set(topk_ids),
            },
        })
        print(f"  Support-graph fresh screen: {case_number}/{CASE_COUNT}")
    case_study = {
        "schema_version": 1,
        "artifact_type": "private_observation_only_support_graph_screen_50",
        "dataset_slice": "nq_open validation[3150:3200], all 50 relative positions",
        "question_global_indices": list(EXPECTED_GLOBAL_INDICES),
        "retrieval": dict(phase["retrieval"]),
        "answer_scorer": dict(phase["answer_scorer"]),
        "generator": {key: identity[key] for key in ("model_id", "revision", "model_path", "config_sha256")},
        "selectors": [dict(spec) for spec in selectors],
        "limitations": [
            "This fixed 50-question slice is a mechanism case study, not a population-level comparison.",
            "answer-string-in-text is a proxy and not a strict gold passage label.",
            "All selectors shared one retrieval and one Answer Scorer result per question.",
            "support_disabled_control is the same two-anchor selector with support weights set to zero; Top-k is retained as a separate rank-only baseline.",
        ],
        "cases": cases,
    }
    if not _forbidden_fields({"question": None}):
        raise AssertionError("forbidden-field self-test failed")
    _write_json(run_dir / "case_study.json", case_study)
    (run_dir / "case_study.md").write_text(_build_markdown(case_study), encoding="utf-8")
    complete_files = _case_study_fingerprints(run_dir, target_dir)
    summary = _screen_summary(cases, contract=contract)
    _write_json(run_dir / "summary.json", summary)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "support_graph_screen_50_metadata",
        "status": "completed",
        "diagnostic_only": True,
        "selection_mutation": False,
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current")},
        "environment": {"python_executable": sys.executable, "python_version": sys.version.split()[0]},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "plugin_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "dataset": {"name": "nq_open", "split": "validation", "parent_slice": "[3150:3200]", "relative_positions": list(EXPECTED_POSITIONS), "global_indices": list(EXPECTED_GLOBAL_INDICES), "question_count": CASE_COUNT},
        "retrieval": {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "top_k": 50, "retrieval_calls": CASE_COUNT},
        "answer_scorer": {"backend": "dpr", "scoring_calls": CASE_COUNT},
        "selectors": {"allowlist": list(EXPECTED_SELECTOR_IDS), "calls": CASE_COUNT * len(EXPECTED_SELECTOR_IDS), "K": 5, "support_disabled_control_matches_topk": summary["control_parity"]["support_disabled_matches_topk_order"]},
        "generator": identity,
        "outputs": {"target_directory": target_dir, "complete_files_exchange_only": True, "complete_files": complete_files["files"], "summary": {"name": "summary.json", "bytes": (run_dir / "summary.json").stat().st_size, "sha256": _sha256(run_dir / "summary.json"), "exchange_path": f"{target_dir}/summary.json"}},
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "validation": {"contract": contract, "answer_string_match_is_proxy_not_strict_gold": True, "decision": summary["decision"]},
    }
    if _forbidden_fields(metadata):
        raise CaseStudyError("compact run metadata contains forbidden raw-content fields")
    _write_json(run_dir / "run_metadata.json", metadata)
    upload_manifest = {
        "schema_version": 1,
        "artifact_type": "rag_selector_case_study_upload_manifest",
        "status": "ready_for_authenticated_exchange_upload",
        "target_directory": target_dir,
        "generated_at_utc": timestamp,
        "git_commit": metadata["git"]["commit"],
        "required_upload_files": ["case_study.md", "case_study.json", "summary.json", "run_metadata.json", "upload_manifest.json"],
        "files_with_precomputed_hashes": complete_files["files"] + [
            {"name": "summary.json", "bytes": (run_dir / "summary.json").stat().st_size, "sha256": _sha256(run_dir / "summary.json"), "exchange_path": f"{target_dir}/summary.json"},
            {"name": "run_metadata.json", "bytes": (run_dir / "run_metadata.json").stat().st_size, "sha256": _sha256(run_dir / "run_metadata.json"), "exchange_path": f"{target_dir}/run_metadata.json"},
        ],
        "github_policy": "Only this manifest and compact run_metadata may be committed; complete case-study files contain raw questions/passages/answers/predictions and remain exchange-only.",
    }
    if _forbidden_fields(upload_manifest):
        raise CaseStudyError("upload manifest contains forbidden raw-content fields")
    _write_json(run_dir / "upload_manifest.json", upload_manifest)
    print(f"Completed case study: {run_dir}")
    print(f"Upload target: {target_dir}")
    print("Upload files: case_study.md case_study.json summary.json run_metadata.json upload_manifest.json")
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/support_graph_screen_50.yaml")
    parser.add_argument("--plan", type=Path, default=root / "configs/experiments/support_graph_screen_50_plan.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (CaseStudyError, OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
