#!/usr/bin/env python3
"""Run the report-only Phase 10B answer-hypothesis bridge screen."""

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

from applications.rag.answer_hypothesis_bridge import consensus_hypothesis, build_bridge_query
from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation import evaluate_answer
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from scripts.collab.five_ideas.phase10b_metrics import FORBIDDEN_FIELDS, Phase10BError, summarize_phase10b
from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_ARMS = ("baseline_frozen_query", "always_bridge", "consensus_gated_bridge")
EXPECTED_SLICES = {"screen": (2700, 50), "formal": (2750, 200), "replication": (2950, 200)}
EXPECTED_PLUGIN_ORDER = (
    "frozen_initial_retrieval_observer",
    "consensus_hypothesis_builder",
    "auxiliary_bridge_retrieval",
    "frozen_union_qore_selector",
    "bridge_order_and_privacy_audit",
)


class Phase10BConfigError(RuntimeError):
    pass


def _root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase10BConfigError("cannot locate project root")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _forbidden(value: Any, path: str = "$root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_FIELDS:
                found.append(child_path)
            found.extend(_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


def _load(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Phase10BConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != "phase10b_answer_hypothesis_bridge" or int(phase.get("schema_version", -1)) != 1:
        raise Phase10BConfigError("unexpected Phase 10B config")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False or phase.get("authorization") != "implemented_observation_only":
        raise Phase10BConfigError("Phase 10B must be implemented observation-only")
    dataset = phase.get("dataset", {})
    if (dataset.get("name"), dataset.get("split")) != ("nq_open", "validation"):
        raise Phase10BConfigError("dataset identity is not frozen")
    stage_spec = dataset.get(stage)
    if not isinstance(stage_spec, dict):
        raise Phase10BConfigError(f"dataset.{stage} is missing")
    expected_offset, expected_count = EXPECTED_SLICES[stage]
    if (int(stage_spec.get("sample_offset", -1)), int(stage_spec.get("max_samples", -1)), bool(stage_spec.get("fresh_slice"))) != (expected_offset, expected_count, True):
        raise Phase10BConfigError(f"dataset.{stage} slice is not frozen")
    if phase.get("retrieval") != {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "initial_top_k": 50, "bridge_top_k": 50, "union_max_candidates": 100}:
        raise Phase10BConfigError("retrieval contract is not frozen")
    expected_selection = {"method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42, "gamma": 1.0, "delta": 0.0, "complementarity_method": None, "qore_prefilter_size": None, "direct_solve_max_n": 20, "use_answer_scorer": True, "answer_scorer_backend": "dpr"}
    if phase.get("selection") != expected_selection:
        raise Phase10BConfigError("selection contract is not frozen")
    generator = phase.get("generator", {})
    if generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise Phase10BConfigError("generator contract is not frozen")
    bridge = phase.get("bridge", {})
    if bridge.get("plugin_id") != "answer_hypothesis_evidence_bridge" or int(bridge.get("min_support", -1)) != 2 or float(bridge.get("min_probability", -1)) != 0.2 or bridge.get("decision_order") != "pre_generation_only" or bridge.get("gold_or_evaluator_available") is not False or bridge.get("generated_answer_available") is not False or bridge.get("qore_feedback") is not False or int(bridge.get("final_context_K", -1)) != 5:
        raise Phase10BConfigError("bridge contract is not frozen")
    if tuple(phase.get("arms", [])) != EXPECTED_ARMS:
        raise Phase10BConfigError("arm order is not frozen")
    outputs = phase.get("outputs", {})
    if outputs.get("compact_only") is not True or outputs.get("bridge_query_persisted") is not False or outputs.get("candidate_text_persisted") is not False:
        raise Phase10BConfigError("output privacy contract is not frozen")
    return phase, stage_spec


def validate_contract(config_path: Path, plan_path: Path, stage: str = "screen") -> dict[str, Any]:
    """Validate the frozen config/plan contract without loading a model or corpus."""
    phase, stage_spec = _load(config_path, stage)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase10BConfigError(f"cannot read plugin plan: {exc}") from exc
    if not isinstance(plan, dict):
        raise Phase10BConfigError("plugin plan must be a JSON object")
    if plan.get("authorization") != phase.get("authorization"):
        raise Phase10BConfigError("config and plugin-plan authorization differ")
    discovery = plan.get("discovery")
    composition = plan.get("composition")
    if not isinstance(discovery, dict) or discovery.get("mode") != "explicit_allowlist":
        raise Phase10BConfigError("plugin discovery must use the explicit allowlist")
    if not isinstance(composition, dict):
        raise Phase10BConfigError("plugin composition is missing")
    allowlist = tuple(discovery.get("allowlist", ()))
    order = tuple(composition.get("order", ()))
    if allowlist != order or allowlist != EXPECTED_PLUGIN_ORDER:
        raise Phase10BConfigError("plugin allowlist and composition order do not match")
    if plan.get("project") != "Q-DUET-VLM":
        raise Phase10BConfigError("plugin plan project is not Q-DUET-VLM")
    reproducibility = plan.get("reproducibility", {})
    if reproducibility.get("config_path") != "configs/experiments/phase10b_answer_hypothesis_bridge.yaml":
        raise Phase10BConfigError("plugin plan config path is not frozen")
    return {
        "status": "valid",
        "phase": phase["name"],
        "stage": stage,
        "slice": stage_spec["slice"],
        "arms": EXPECTED_ARMS,
        "selection_mutation": False,
        "report_only": True,
        "model_loaded": False,
        "wiki_dpr_started": False,
    }


def _resolve_generator(root: Path, specification: Mapping[str, Any], override: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    for raw in specification.get("model_path_candidates", []):
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    candidates.extend([root / "models" / "llama3-8b", Path.home() / ".cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct"])
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub/models--NousResearch--Meta-Llama-3-8B-Instruct")
    attempted: list[str] = []
    for candidate in candidates:
        attempted.append(str(candidate))
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate if (candidate / "config.json").is_file() else snapshot
        if (resolved / "config.json").is_file():
            return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "model_path": str(resolved.resolve()), "config_sha256": _sha256(resolved / "config.json"), "resolution": "cli_override" if override else "local_cache"}
    raise Phase10BConfigError("reference generator not found; attempted: " + ", ".join(attempted))


def _best_hypothesis(hypotheses: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in hypotheses:
        for item in row:
            text = str(item.get("text") or item.get("normalized") or "").strip()
            if not text:
                continue
            candidate = {"text": text, "probability": float(item.get("probability", 0.0)), "score": float(item.get("score", 0.0))}
            if best is None or (candidate["probability"], candidate["score"], candidate["text"]) > (best["probability"], best["score"], best["text"]):
                best = candidate
    return best


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _union(initial_embeddings: np.ndarray, initial_texts: Sequence[str], bridge_embeddings: np.ndarray, bridge_texts: Sequence[str]) -> tuple[np.ndarray, list[str], int]:
    embeddings: list[np.ndarray] = []
    texts: list[str] = []
    seen: set[str] = set()
    for embedding, text in list(zip(initial_embeddings, initial_texts)) + list(zip(bridge_embeddings, bridge_texts)):
        key = _digest(str(text))
        if key in seen:
            continue
        seen.add(key)
        embeddings.append(np.asarray(embedding))
        texts.append(str(text))
    if not embeddings:
        raise Phase10BConfigError("candidate union is empty")
    return np.asarray(embeddings), texts, max(0, len(texts) - len(initial_texts))


def _select_and_generate(query_embedding: np.ndarray, embeddings: np.ndarray, texts: Sequence[str], question: str, scorer: Any, generator: Any, gold_answers: list[str], selection: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    answer_scores = scorer.score_passages(question, list(texts))
    selected = select_passages(query_embedding, embeddings, K=int(selection["K"]), method="qore", num_reads=int(selection["num_reads"]), lam=float(selection["lam"]), gamma=float(selection["gamma"]), delta=float(selection["delta"]), complementarity_method=None, qore_prefilter_size=None, direct_solve_max_n=int(selection["direct_solve_max_n"]), answer_scorer=scorer, passage_texts=list(texts), question=question, seed=int(selection["seed"]), relevance_scores=answer_scores)
    selection_and_score_ms = (time.perf_counter() - started) * 1000.0
    selected_texts = [texts[int(index)] for index in selected]
    generation_started = time.perf_counter()
    prediction = generator.generate(question, selected_texts)
    generation_ms = (time.perf_counter() - generation_started) * 1000.0
    metrics = evaluate_answer(prediction, gold_answers)
    return {"em": float(metrics["em"]), "f1": float(metrics["f1"]), "generation_time_ms": generation_ms, "selected_texts": selected_texts, "selection_score_time_ms": selection_and_score_ms}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase10b_answer_hypothesis_bridge.yaml")
    parser.add_argument("--stage", choices=("screen", "formal", "replication"), default="screen")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _root()
    config_path = (args.config if args.config.is_absolute() else root / args.config).resolve()
    phase, stage_spec = _load(config_path, args.stage)
    identity = _resolve_generator(root, phase["generator"], args.model_path)
    if args.validate_only:
        print(json.dumps({"status": "valid", "phase": phase["name"], "stage": args.stage, "slice": stage_spec["slice"], "arms": EXPECTED_ARMS, "selection_mutation": False, "report_only": True, "model_loaded": False, "wiki_dpr_started": False}, sort_keys=True))
        return None
    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{timestamp}_{args.stage}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{args.stage}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {"schema_version": 1, "phase": phase["name"], "stage": args.stage, "status": "running", "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"path": str(config_path), "sha256": _sha256(config_path)}, "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")}, "python": {"executable": sys.executable, "version": sys.version}, "generator": identity, "dataset": {"name": phase["dataset"]["name"], "split": phase["dataset"]["split"], **stage_spec}, "retrieval": phase["retrieval"], "selection": phase["selection"], "bridge": phase["bridge"]}
    _write(run_dir / "run_metadata.json", metadata)
    sample_end = int(stage_spec["sample_offset"]) + int(stage_spec["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["split"], sample_end)[int(stage_spec["sample_offset"]):sample_end]
    if len(questions) != int(stage_spec["max_samples"]):
        raise Phase10BConfigError("fresh slice length mismatch")
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend="dpr")
    generator = Generator(identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    arms: dict[str, dict[str, Any]] = {name: {"config": {"arm": name}, "samples": []} for name in EXPECTED_ARMS}
    started_total = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        question_id = str(item["id"])
        question = str(item["question"])
        gold_answers = [str(value) for value in item.get("answers", []) if str(value).strip()]
        question_started = time.perf_counter()
        query_embedding = encoder.encode_queries([question])[0]
        _, initial_embeddings, initial_texts, _ = corpus_manager.retrieve_with_embeddings(query_embedding, int(phase["retrieval"]["initial_top_k"]))
        _, hypotheses = scorer.score_passages_with_hypotheses(question, initial_texts, top_m=3, max_answer_tokens=10)
        baseline = _select_and_generate(query_embedding, initial_embeddings, initial_texts, question, scorer, generator, gold_answers, phase["selection"])
        initial_retrieval_hit = any(answer_has_match_in_text(answer, text) for answer in gold_answers for text in initial_texts)
        always_hypothesis = _best_hypothesis(hypotheses)
        consensus = consensus_hypothesis(hypotheses, min_support=int(phase["bridge"]["min_support"]), min_probability=float(phase["bridge"]["min_probability"]))
        baseline_pipeline_ms = (time.perf_counter() - question_started) * 1000.0
        bridge_cache: dict[str, tuple[dict[str, Any], int, int, bool, float]] = {}

        def bridge_arm(hypothesis: Mapping[str, Any] | None) -> tuple[dict[str, Any], int, int, bool, float]:
            if hypothesis is None:
                return baseline, 0, len(initial_texts), False, 0.0
            key = json.dumps({"text": str(hypothesis.get("text", "")), "support": int(hypothesis.get("support", 1))}, sort_keys=True)
            if key in bridge_cache:
                return bridge_cache[key]
            bridge_started = time.perf_counter()
            bridge_query = build_bridge_query(question, hypothesis)
            bridge_embedding = encoder.encode_queries([bridge_query])[0]
            _, bridge_embeddings, bridge_texts, _ = corpus_manager.retrieve_with_embeddings(bridge_embedding, int(phase["retrieval"]["bridge_top_k"]))
            union_embeddings, union_texts, bridge_new_count = _union(initial_embeddings, initial_texts, bridge_embeddings, bridge_texts)
            bridge_result = _select_and_generate(query_embedding, union_embeddings, union_texts, question, scorer, generator, gold_answers, phase["selection"])
            bridge_hit = any(answer_has_match_in_text(answer, text) for answer in gold_answers for text in bridge_texts)
            value = (dict(bridge_result, bridge_retrieval_hit=bridge_hit), bridge_new_count, len(union_texts), True, (time.perf_counter() - bridge_started) * 1000.0)
            bridge_cache[key] = value
            return value

        always_result, always_new_count, always_union_size, always_available, always_extra_ms = bridge_arm(always_hypothesis)
        gated_result, gated_new_count, gated_union_size, gated_available, gated_extra_ms = bridge_arm(consensus)
        gated_applied = bool(consensus is not None)

        def compact(result: Mapping[str, Any], *, applied: bool, available: bool, new_count: int, union_size: int, hypothesis: Mapping[str, Any] | None, bridge_hit: bool, pipeline_ms: float) -> dict[str, Any]:
            selected_hit = any(answer_has_match_in_text(answer, text) for answer in gold_answers for text in result["selected_texts"])
            return {"question_id": question_id, "em": float(result["em"]), "f1": float(result["f1"]), "generation_time_ms": float(result["generation_time_ms"]), "pipeline_time_ms": float(pipeline_ms), "initial_retrieval_hit": bool(initial_retrieval_hit), "bridge_retrieval_hit": bool(bridge_hit), "selected_hit": bool(selected_hit), "bridge_applied": bool(applied), "bridge_available": bool(available), "hypothesis_support": int(hypothesis.get("support", 1) if hypothesis else 0), "hypothesis_probability": float(hypothesis.get("mean_probability", hypothesis.get("probability", 0.0)) if hypothesis else 0.0), "bridge_new_count": int(new_count), "union_size": int(union_size), "final_context_K": 5}

        arms["baseline_frozen_query"]["samples"].append(compact(baseline, applied=False, available=False, new_count=0, union_size=len(initial_texts), hypothesis=None, bridge_hit=False, pipeline_ms=baseline_pipeline_ms))
        arms["always_bridge"]["samples"].append(compact(always_result, applied=always_available, available=always_available, new_count=always_new_count, union_size=always_union_size, hypothesis=always_hypothesis, bridge_hit=bool(always_result.get("bridge_retrieval_hit", False)), pipeline_ms=baseline_pipeline_ms + always_extra_ms))
        arms["consensus_gated_bridge"]["samples"].append(compact(gated_result, applied=gated_applied, available=gated_available, new_count=gated_new_count, union_size=gated_union_size, hypothesis=consensus, bridge_hit=bool(gated_result.get("bridge_retrieval_hit", False)), pipeline_ms=baseline_pipeline_ms + (gated_extra_ms if gated_applied else 0.0)))
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 10B answer-hypothesis bridge: {index}/{len(questions)}")

    result = {"schema_version": 1, "phase": phase["name"], "stage": args.stage, "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"dataset": stage_spec, "retrieval": phase["retrieval"], "selection": phase["selection"], "bridge": {"plugin_id": phase["bridge"]["plugin_id"], "min_support": phase["bridge"]["min_support"], "min_probability": phase["bridge"]["min_probability"], "final_context_K": 5}}, "arms": arms}
    forbidden = _forbidden(result)
    if forbidden:
        raise Phase10BConfigError(f"compact result contains forbidden fields: {forbidden[:5]}")
    _write(run_dir / "result.json", result)
    gate = dict(phase["gate"][args.stage])
    if args.bootstrap_reps is not None:
        gate["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_phase10b(result, gate=gate)
    _write(run_dir / "summary.json", summary)
    metadata.update({"status": "completed", "timing_ms": {"total": (time.perf_counter() - started_total) * 1000.0}, "summary": {"path": str(run_dir / "summary.json"), "decision": summary["decision"]}})
    _write(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 10B screen: {run_dir}")
    print(f"Decision: {summary['decision']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (Phase10BConfigError, Phase10BError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
