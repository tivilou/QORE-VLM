#!/usr/bin/env python3
"""Run the observation-only Phase 9K gold-free applicability diagnostic."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

_SCRIPT_PATH = Path(__file__).resolve()
for _candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_candidate / "configs").is_dir() and (_candidate / "applications").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation import evaluate_answer
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text
from scripts.collab.five_ideas.phase9h_probe import run_context_probe
from scripts.collab.five_ideas.phase9i_probe import candidate_pairs, run_evidence_constrained_probe, run_extractive_candidate_probe
from scripts.collab.five_ideas.phase9j_metrics import FORBIDDEN_FIELDS as PHASE9J_FORBIDDEN
from scripts.collab.five_ideas.phase9j_probe import build_candidate_scores, choose_candidate
from scripts.collab.five_ideas.run_phase9j_context_lift_candidate_ranking import _reader_support_scores
from scripts.collab.five_ideas.phase9k_gold_free_applicability import decide_applicability, exact_span_supported
from scripts.collab.five_ideas.phase9k_gold_free_metrics import Phase9KError, summarize_phase9k, validate_result


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_PLUGINS = (
    "frozen_baseline_observer",
    "candidate_generation_adapter",
    "reader_span_support_candidate_ranker",
    "gold_free_applicability_gate",
    "ranker_order_invariance_audit",
)
RANKER_PROFILE = "reader_span_support_v1"


class Phase9KConfigError(RuntimeError):
    """Raised when the frozen Phase 9K contract is malformed."""


def _project_root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase9KConfigError("cannot locate project root")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_forbidden(value: Any, path: str = "$root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in PHASE9J_FORBIDDEN:
                findings.append(child_path)
            findings.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden(child, f"{path}[{index}]"))
    return findings


def _plugin_tree_hash(root: Path) -> str:
    paths = (
        root / "scripts/collab/five_ideas/phase9h_probe.py",
        root / "scripts/collab/five_ideas/phase9i_probe.py",
        root / "scripts/collab/five_ideas/phase9j_probe.py",
        root / "scripts/collab/five_ideas/phase9j_metrics.py",
        root / "scripts/collab/five_ideas/run_phase9j_context_lift_candidate_ranking.py",
        root / "scripts/collab/five_ideas/phase9k_gold_free_applicability.py",
        root / "scripts/collab/five_ideas/phase9k_gold_free_metrics.py",
        root / "scripts/collab/five_ideas/run_phase9k_gold_free_applicability.py",
    )
    entries = []
    for path in paths:
        if not path.is_file():
            raise Phase9KConfigError(f"plugin tree file is missing: {path}")
        entries.append({"path": str(path.relative_to(root)), "sha256": _sha256_file(path)})
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def _load_config(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Phase9KConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != "phase9k_gold_free_applicability":
        raise Phase9KConfigError("unexpected Phase 9K configuration")
    if phase.get("schema_version") != 1 or phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False or phase.get("authorization") != "implemented":
        raise Phase9KConfigError("Phase 9K must be authorized and observation-only")
    expected_slices = {"screen": (2200, 50), "formal": (2250, 200), "replication": (2450, 200)}
    dataset = phase.get("dataset", {})
    for name, (offset, count) in expected_slices.items():
        spec = dataset.get(name)
        if not isinstance(spec, dict) or int(spec.get("sample_offset", -1)) != offset or int(spec.get("max_samples", -1)) != count or spec.get("fresh_slice") is not True:
            raise Phase9KConfigError(f"dataset.{name} slice is not frozen")
    if phase.get("retrieval") != {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "top_k": 50}:
        raise Phase9KConfigError("retrieval contract is not frozen")
    selection = phase.get("selection", {})
    if selection != {"method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42, "gamma": 1.0, "delta": 0.0, "complementarity_method": None, "qore_prefilter_size": None, "direct_solve_max_n": 20, "lambda_mmr": 0.7, "saturation_alpha": 1.0, "lambda_submodular": 0.5, "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8, "use_answer_scorer": True, "answer_scorer_backend": "dpr"}:
        raise Phase9KConfigError("selection contract is not frozen")
    generator = phase.get("generator", {})
    if generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise Phase9KConfigError("generator contract is not frozen")
    applicability = phase.get("applicability", {})
    if applicability != {"profile": "gold_free_consensus_gate_v2", "min_candidate_count": 3, "min_unique_candidate_count": 2, "reader_margin_min": 0.0, "require_exact_span": True, "require_baseline_not_exact": True, "require_candidate_consensus": True, "gold_used_for_decision": False}:
        raise Phase9KConfigError("applicability policy is not frozen")
    if phase.get("plugins", {}).get("allowlist") != list(EXPECTED_PLUGINS) or phase.get("plugins", {}).get("diagnostic_outputs_used_for_selection") is not False or phase.get("plugins", {}).get("production_intervention") is not False:
        raise Phase9KConfigError("plugin contract is not frozen")
    if phase.get("gate", {}).get("formal", {}).get("primary_ranker_profile") != RANKER_PROFILE:
        raise Phase9KConfigError("formal primary ranker is not reader-span")
    return phase, dataset[stage]


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
        direct = candidate / "config.json"
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate.resolve() if direct.is_file() else snapshot.resolve()
        if (resolved / "config.json").is_file():
            return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "model_path": str(resolved), "config_sha256": _sha256_file(resolved / "config.json"), "resolution": "cli_override" if override else "local_cache"}
    raise Phase9KConfigError("reference generator not found; attempted: " + ", ".join(attempted))


def _arm(prediction: str, gold_answers: list[str], elapsed: float) -> dict[str, Any]:
    metrics = evaluate_answer(prediction, gold_answers)
    return {"attempted": True, "em": float(metrics["em"]), "f1": float(metrics["f1"]), "generation_time_ms": float(elapsed)}


def _empty_ranker() -> dict[str, Any]:
    return {"attempted": False, "choice_mode": None, "permuted_choice_mode": None, "order_agreement": None, "parse_status": "not_attempted", "em": None, "f1": None, "generation_time_ms": None, "score_time_ms": None, "score_to_baseline_ratio": None}


def _ranker_choice(answer_scorer: Any, question: str, passages: Sequence[str], pairs: Sequence[tuple[str, str]], baseline_time_ms: float) -> tuple[dict[str, Any], dict[str, float], dict[str, bool]]:
    started = time.perf_counter()
    texts = {mode: text for mode, text in pairs}
    reader_values = _reader_support_scores(answer_scorer, question, passages, texts)
    raw_scores = {mode: {"context_lift": 0.0, "reader_support": float(score)} for mode, score in reader_values.items()}
    scores = build_candidate_scores(raw_scores, reader_weight=1.0)
    original_modes = [mode for mode, _ in pairs]
    original = choose_candidate({mode: scores[mode] for mode in original_modes}, RANKER_PROFILE)
    permuted_modes = list(reversed(original_modes))
    permuted = choose_candidate({mode: scores[mode] for mode in permuted_modes}, RANKER_PROFILE)
    elapsed = (time.perf_counter() - started) * 1000.0
    return {"choice_mode": original, "permuted_choice_mode": permuted, "order_agreement": original == permuted, "score_time_ms": elapsed, "score_to_baseline_ratio": elapsed / max(float(baseline_time_ms), 1.0)}, reader_values, {mode: False for mode, _ in pairs}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase9k_gold_free_applicability.yaml")
    parser.add_argument("--stage", choices=("screen", "formal", "replication"), default="screen")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config_path = config_path.resolve()
    phase, stage_spec = _load_config(config_path, args.stage)
    if args.validate_only:
        print(json.dumps({"status": "valid", "phase": phase["name"], "stage": args.stage, "slice": stage_spec["slice"], "plugins": list(EXPECTED_PLUGINS), "applicability_profile": phase["applicability"]["profile"], "selection_mutation": False, "report_only": True}, sort_keys=True))
        return None
    identity = _resolve_generator(root, phase["generator"], args.model_path)
    output_root = Path(args.output_root or phase["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{timestamp}_{args.stage}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{args.stage}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {"schema_version": 1, "phase": phase["name"], "stage": args.stage, "status": "running", "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"path": str(config_path), "sha256": _sha256_file(config_path)}, "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")}, "python": {"executable": sys.executable, "version": sys.version}, "generator": identity, "dataset": {"name": phase["dataset"]["name"], "split": phase["dataset"]["split"], "sample_offset": stage_spec["sample_offset"], "max_samples": stage_spec["max_samples"], "slice": stage_spec["slice"], "fresh_slice": True}, "plugins": {"allowlist": list(EXPECTED_PLUGINS), "tree_sha256": _plugin_tree_hash(root), "applicability_profile": phase["applicability"]["profile"], "diagnostic_outputs_used_for_selection": False}}
    _write_json(run_dir / "run_metadata.json", metadata)
    sample_end = int(stage_spec["sample_offset"]) + int(stage_spec["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["split"], sample_end)
    questions = questions[int(stage_spec["sample_offset"]) : sample_end]
    if len(questions) != int(stage_spec["max_samples"]):
        raise Phase9KConfigError("fresh slice length mismatch")
    np.random.seed(int(phase["selection"]["seed"]))
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    corpus_manager.build(questions)
    answer_scorer = make_answer_scorer(backend="dpr")
    generator = Generator(identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        question = str(item["question"])
        question_id = str(item["id"])
        gold_answers = [str(value) for value in item.get("answers", []) if str(value).strip()]
        query_embedding = encoder.encode_queries([question])[0]
        _, retrieved_embeddings, retrieved_texts, _ = corpus_manager.retrieve_with_embeddings(query_embedding, int(phase["retrieval"]["top_k"]))
        answer_scores = answer_scorer.score_passages(question, retrieved_texts)
        selection_started = time.perf_counter()
        selected_local = select_passages(query_embedding, retrieved_embeddings, K=5, method="qore", num_reads=100, lam=2.0, gamma=1.0, delta=0.0, complementarity_method=None, qore_prefilter_size=None, direct_solve_max_n=20, lambda_mmr=0.7, saturation_alpha=1.0, lambda_submodular=0.5, dpp_quality_scale=2.0, dpp_jitter=1.0e-8, answer_scorer=answer_scorer, passage_texts=retrieved_texts, question=question, seed=42, relevance_scores=answer_scores)
        selection_time_ms = (time.perf_counter() - selection_started) * 1000.0
        selected_texts = [retrieved_texts[int(value)] for value in selected_local]
        retrieval_hit = any(answer_has_match_in_text(answer, passage) for answer in gold_answers for passage in retrieved_texts)
        selected_hit = any(answer_has_match_in_text(answer, passage) for answer in gold_answers for passage in selected_texts)
        baseline_result = run_context_probe(generator, question, selected_texts)
        baseline = _arm(baseline_result.prediction, gold_answers, baseline_result.generation_time_ms)
        extractive = run_extractive_candidate_probe(generator, question, selected_texts)
        constrained = run_evidence_constrained_probe(generator, question, selected_texts)
        candidate_generation_time_ms = float(extractive.generation_time_ms + constrained.generation_time_ms)
        pairs = candidate_pairs(baseline_result.prediction, extractive, constrained)
        ranker = _empty_ranker()
        decision = None
        if len(pairs) >= int(phase["applicability"]["min_candidate_count"]):
            texts = {mode: text for mode, text in pairs}
            ranker_started = time.perf_counter()
            reader_values = _reader_support_scores(answer_scorer, question, selected_texts, texts)
            raw_scores = {mode: {"context_lift": 0.0, "reader_support": float(score)} for mode, score in reader_values.items()}
            exact_modes = {mode: exact_span_supported(answer_scorer.tokenizer, selected_texts, text) for mode, text in pairs}
            decision, scores = decide_applicability(candidates=pairs, raw_scores=raw_scores, exact_span_modes=exact_modes, reader_margin_min=float(phase["applicability"]["reader_margin_min"]), min_candidate_count=int(phase["applicability"]["min_candidate_count"]), min_unique_candidate_count=int(phase["applicability"]["min_unique_candidate_count"]), baseline_exact_span=exact_modes.get("baseline_v1", False), require_baseline_not_exact=bool(phase["applicability"]["require_baseline_not_exact"]), require_candidate_consensus=bool(phase["applicability"]["require_candidate_consensus"]))
            chosen = decision.chosen_mode
            permuted = choose_candidate({mode: scores[mode] for mode, _ in reversed(pairs)}, RANKER_PROFILE) if scores else None
            ranker_score_time_ms = (time.perf_counter() - ranker_started) * 1000.0
            if decision.apply and chosen is not None:
                chosen_text = texts[chosen]
                chosen_metrics = evaluate_answer(chosen_text, gold_answers)
                ranker = {"attempted": True, "choice_mode": chosen, "permuted_choice_mode": permuted, "order_agreement": chosen == permuted, "parse_status": "ok", "em": float(chosen_metrics["em"]), "f1": float(chosen_metrics["f1"]), "generation_time_ms": float(baseline_result.generation_time_ms), "score_time_ms": float(ranker_score_time_ms), "score_to_baseline_ratio": float(ranker_score_time_ms / max(float(baseline_result.generation_time_ms), 1.0))}
        if decision is None:
            applicability = {"apply": False, "reason_code": "no_candidates", "chosen_mode": None, "baseline_score": None, "chosen_score": None, "reader_margin": None, "chosen_exact_span": None, "baseline_exact_span": exact_span_supported(answer_scorer.tokenizer, selected_texts, baseline_result.prediction), "candidate_consensus": False}
        else:
            applicability = {"apply": decision.apply, "reason_code": decision.reason_code, "chosen_mode": decision.chosen_mode if decision.apply else None, "baseline_score": decision.baseline_score if decision.apply else None, "chosen_score": decision.chosen_score if decision.apply else None, "reader_margin": decision.reader_margin if decision.apply else None, "chosen_exact_span": decision.chosen_exact_span if decision.apply else None, "baseline_exact_span": bool(decision.baseline_exact_span), "candidate_consensus": bool(decision.candidate_consensus)}
        samples.append({"question_id": question_id, "retrieval_hit": retrieval_hit, "selected_hit": selected_hit, "selection_time_ms": selection_time_ms, "candidate_generation_time_ms": candidate_generation_time_ms, "candidate_count": len(pairs), "unique_candidate_count": len({" ".join(text.strip().lower().split()) for _, text in pairs}), "parse_failures": 3 - len(pairs), "applicability": applicability, "baseline": baseline, "ranker": ranker})
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 9K {args.stage}: {index}/{len(questions)}")
    result = {"schema_version": 1, "phase": phase["name"], "stage": args.stage, "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"dataset": phase["dataset"]["name"], "split": phase["dataset"]["split"], "sample_offset": stage_spec["sample_offset"], "max_samples": stage_spec["max_samples"], "slice": stage_spec["slice"], "applicability_profile": phase["applicability"]["profile"], "ranker_profile": RANKER_PROFILE, "gold_used_for_decision": False}, "samples": samples}
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase9KConfigError(f"compact result contains forbidden fields: {forbidden[:5]}")
    validate_result(result)
    _write_json(run_dir / "result.json", result)
    summary = summarize_phase9k(result, gate=phase["gate"])
    _write_json(run_dir / "summary.json", summary)
    metadata.update({"status": "completed", "timing_ms": {"total": (time.perf_counter() - started) * 1000.0}, "summary": {"path": str(run_dir / "summary.json"), "primary_failure_class": summary["decision"]["primary_failure_class"]}})
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 9K {args.stage}: {run_dir}")
    print(f"Report-only gate: {summary['decision']['primary_failure_class']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (Phase9KConfigError, Phase9KError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
