#!/usr/bin/env python3
"""Generate a private, content-complete RAG selector case study.

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
EXPECTED_POSITIONS = (0, 10, 20, 30, 40)
EXPECTED_GLOBAL_INDICES = (1950, 1960, 1970, 1980, 1990)
EXPECTED_SELECTOR_IDS = (
    "qore_as",
    "topk_as",
    "mmr_as",
    "submodular_as",
    "spectral_dpp_as",
)
EXPECTED_SELECTOR_SPECS = (
    {
        "id": "qore_as", "method": "qore", "K": 5, "num_reads": 100,
        "lam": 2.0, "seed": 42, "gamma": 1.0, "delta": 0.0,
        "complementarity_method": None, "direct_solve_max_n": 20,
        "qore_prefilter_size": None,
    },
    {"id": "topk_as", "method": "topk", "K": 5},
    {"id": "mmr_as", "method": "mmr", "K": 5, "lambda_mmr": 0.7},
    {
        "id": "submodular_as", "method": "submodular", "K": 5,
        "saturation_alpha": 1.0, "lambda_submodular": 0.5,
    },
    {
        "id": "spectral_dpp_as", "method": "spectral_dpp", "K": 5,
        "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8,
    },
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
        raise CaseStudyError("selector allowlist must contain exactly five selectors")
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
    if phase.get("name") != "rag_selector_case_study" or int(phase.get("schema_version", -1)) != 1:
        raise CaseStudyError("unexpected case-study config")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise CaseStudyError("case study must remain observation-only")
    dataset = phase.get("dataset")
    if not isinstance(dataset, dict) or (dataset.get("name"), dataset.get("split")) != ("nq_open", "validation"):
        raise CaseStudyError("dataset identity is not frozen")
    parent = dataset.get("parent_slice")
    if not isinstance(parent, dict) or (int(parent.get("sample_offset", -1)), int(parent.get("max_samples", -1))) != (1950, 50):
        raise CaseStudyError("parent slice must be validation[1950:2000]")
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
    if not isinstance(outputs, dict) or outputs.get("root") != "exchange/five_ideas/rag_selector_case_study":
        raise CaseStudyError("output root is not frozen")
    if outputs.get("complete_files_exchange_only") is not True:
        raise CaseStudyError("complete files must be exchange-only")
    if outputs.get("complete_files") != ["case_study.md", "case_study.json"] or outputs.get("compact_files") != ["run_metadata.json", "upload_manifest.json"]:
        raise CaseStudyError("output file contract is not frozen")
    if set(outputs.get("forbidden_compact_fields", ())) != FORBIDDEN_COMPACT_FIELDS:
        raise CaseStudyError("compact privacy fields are not frozen")
    plan = _load_plan(plan_path)
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v1" or plan.get("project") != "Q-DUET-VLM" or plan.get("authorization") != "implemented":
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
    if plan.get("reproducibility", {}).get("config_path") != "configs/experiments/rag_selector_case_study.yaml":
        raise CaseStudyError("plugin plan config path is not frozen")
    return {
        "status": "valid",
        "phase": phase["name"],
        "dataset": "nq_open validation[1950:2000] positions 0,10,20,30,40",
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


def _select_one(spec: Mapping[str, Any], query_embedding: np.ndarray, embeddings: np.ndarray, records: Sequence[Mapping[str, Any]], question: str, scorer: Any, scores: np.ndarray) -> list[int]:
    from applications.rag.selector import select_passages

    texts = [str(record["passage"]) for record in records]
    selected = np.asarray(select_passages(**_selector_kwargs(spec, query_embedding, embeddings, texts, question, scorer, scores))).reshape(-1)
    indices = [int(value) for value in selected]
    if len(indices) != 5 or len(set(indices)) != 5 or any(index < 0 or index >= len(records) for index in indices):
        raise CaseStudyError(f"selector {spec['id']} did not return five unique valid candidates")
    return indices


def _answer_proxy(answers: Sequence[str], records: Sequence[Mapping[str, Any]]) -> bool:
    from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text

    return any(answer_has_match_in_text(str(answer), str(record["passage"])) for answer in answers for record in records)


def _selector_record(spec: Mapping[str, Any], indices: Sequence[int], records: Sequence[Mapping[str, Any]], answer_scores: np.ndarray, prediction: str, answers: Sequence[str]) -> dict[str, Any]:
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
        "diagnostics": {
            "answer_has_match_in_selected_text": _answer_proxy(answers, selected),
            "answer_string_match_is_proxy_not_strict_gold": True,
            "selected_retrieved_ranks": [int(item["retrieved_rank"]) for item in selected],
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
        "> Private, observation-only diagnostic. Five fixed NQ-Open validation questions; this is not a population-level comparison.",
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
    if len(questions) != 5:
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
    target_dir = f"five_ideas/rag_selector_case_study/{run_dir.name}"
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
            indices = _select_one(spec, query_embedding, retrieval["embeddings"], records, question, scorer, answer_scores)
            selected_texts = [texts[index] for index in indices]
            prediction = str(generator.generate(question, selected_texts))
            selector_records[str(spec["id"])] = _selector_record(spec, indices, records, answer_scores, prediction, answers)
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
        })
        print(f"  RAG selector case study: {case_number}/5")
    case_study = {
        "schema_version": 1,
        "artifact_type": "private_observation_only_rag_selector_case_study",
        "dataset_slice": "nq_open validation[1950:2000], relative positions 0,10,20,30,40",
        "question_global_indices": list(EXPECTED_GLOBAL_INDICES),
        "retrieval": dict(phase["retrieval"]),
        "answer_scorer": dict(phase["answer_scorer"]),
        "generator": {key: identity[key] for key in ("model_id", "revision", "model_path", "config_sha256")},
        "selectors": [dict(spec) for spec in selectors],
        "limitations": [
            "Five fixed questions are a mechanism case study, not a population-level comparison.",
            "answer-string-in-text is a proxy and not a strict gold passage label.",
            "All selectors shared one retrieval and one Answer Scorer result per question.",
        ],
        "cases": cases,
    }
    if not _forbidden_fields({"question": None}):
        raise AssertionError("forbidden-field self-test failed")
    _write_json(run_dir / "case_study.json", case_study)
    (run_dir / "case_study.md").write_text(_build_markdown(case_study), encoding="utf-8")
    complete_files = _case_study_fingerprints(run_dir, target_dir)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "rag_selector_case_study_metadata",
        "status": "completed",
        "diagnostic_only": True,
        "selection_mutation": False,
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current")},
        "environment": {"python_executable": sys.executable, "python_version": sys.version.split()[0]},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "plugin_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "dataset": {"name": "nq_open", "split": "validation", "parent_slice": "[1950:2000]", "relative_positions": list(EXPECTED_POSITIONS), "global_indices": list(EXPECTED_GLOBAL_INDICES), "question_count": 5},
        "retrieval": {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "top_k": 50, "retrieval_calls": 5},
        "answer_scorer": {"backend": "dpr", "scoring_calls": 5},
        "selectors": {"allowlist": list(EXPECTED_SELECTOR_IDS), "calls": 25, "K": 5},
        "generator": identity,
        "outputs": {"target_directory": target_dir, "complete_files_exchange_only": True, "complete_files": complete_files["files"]},
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "validation": {"contract": contract, "answer_string_match_is_proxy_not_strict_gold": True},
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
        "required_upload_files": ["case_study.md", "case_study.json", "run_metadata.json", "upload_manifest.json"],
        "files_with_precomputed_hashes": complete_files["files"] + [
            {"name": "run_metadata.json", "bytes": (run_dir / "run_metadata.json").stat().st_size, "sha256": _sha256(run_dir / "run_metadata.json"), "exchange_path": f"{target_dir}/run_metadata.json"},
        ],
        "github_policy": "Only this manifest and compact run_metadata may be committed; complete case-study files contain raw questions/passages/answers/predictions and remain exchange-only.",
    }
    if _forbidden_fields(upload_manifest):
        raise CaseStudyError("upload manifest contains forbidden raw-content fields")
    _write_json(run_dir / "upload_manifest.json", upload_manifest)
    print(f"Completed case study: {run_dir}")
    print(f"Upload target: {target_dir}")
    print("Upload files: case_study.md case_study.json run_metadata.json upload_manifest.json")
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/rag_selector_case_study.yaml")
    parser.add_argument("--plan", type=Path, default=root / "configs/experiments/rag_selector_case_study_plan.json")
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
