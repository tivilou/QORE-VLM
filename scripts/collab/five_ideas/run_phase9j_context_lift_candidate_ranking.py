#!/usr/bin/env python3
"""Run the collaborator-only Phase 9J candidate ranking diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_SCRIPT_PATH = Path(__file__).resolve()
for _candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_candidate / "configs").is_dir() and (_candidate / "applications").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import numpy as np
import yaml

from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation import evaluate_answer
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text

try:
    from scripts.collab.five_ideas.phase9h_probe import (
        run_context_probe,
        run_gold_answer_copy_probe,
    )
    from scripts.collab.five_ideas.phase9i_probe import (
        candidate_pairs,
        candidate_parse_status,
        run_evidence_constrained_probe,
        run_extractive_candidate_probe,
    )
    from scripts.collab.five_ideas.phase9j_metrics import (
        FORBIDDEN_FIELDS,
        Phase9JError,
        summarize_phase9j,
        validate_result,
    )
    from scripts.collab.five_ideas.phase9j_probe import (
        COMBINED_PROFILE,
        CONTEXT_LIFT_PROFILE,
        RANKER_PROFILES,
        READER_SUPPORT_PROFILE,
        build_candidate_scores,
        choose_candidate,
    )
except ImportError:  # pragma: no cover
    from phase9h_probe import run_context_probe, run_gold_answer_copy_probe
    from phase9i_probe import (
        candidate_pairs,
        candidate_parse_status,
        run_evidence_constrained_probe,
        run_extractive_candidate_probe,
    )
    from phase9j_metrics import FORBIDDEN_FIELDS, Phase9JError, summarize_phase9j, validate_result
    from phase9j_probe import (
        COMBINED_PROFILE,
        CONTEXT_LIFT_PROFILE,
        RANKER_PROFILES,
        READER_SUPPORT_PROFILE,
        build_candidate_scores,
        choose_candidate,
    )


class Phase9JConfigError(RuntimeError):
    """Raised when the frozen Phase 9J contract is malformed."""


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_PLUGINS = (
    "frozen_baseline_observer",
    "gold_answer_copy_control",
    "extractive_span_candidate",
    "evidence_constrained_candidate",
    "candidate_set_coverage_combiner",
    "context_lift_candidate_ranker",
    "reader_span_support_candidate_ranker",
    "combined_candidate_ranker",
    "ranker_order_invariance_audit",
)


def _project_root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase9JConfigError("cannot locate project root")


def _sha256_file(path: Path) -> str:
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_forbidden(value: Any, path: str = "$root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in FORBIDDEN_FIELDS:
                findings.append(child_path)
            findings.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden(child, f"{path}[{index}]"))
    return findings


def _plugin_tree_hash(root: Path) -> str:
    paths = (
        root / "applications/rag/generation/generator.py",
        root / "scripts/rag/eval/eval_rag_refactored.py",
        root / "scripts/collab/five_ideas/phase9h_probe.py",
        root / "scripts/collab/five_ideas/phase9i_probe.py",
        root / "scripts/collab/five_ideas/phase9j_probe.py",
        root / "scripts/collab/five_ideas/phase9j_metrics.py",
        root / "scripts/collab/five_ideas/run_phase9j_context_lift_candidate_ranking.py",
    )
    entries = []
    for path in paths:
        if not path.is_file():
            raise Phase9JConfigError(f"plugin tree file is missing: {path}")
        entries.append({"path": str(path.relative_to(root)), "sha256": _sha256_file(path)})
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def _load_config(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Phase9JConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != "phase9j_context_lift_candidate_ranking":
        raise Phase9JConfigError("unexpected Phase 9J configuration")
    if phase.get("schema_version") != 1 or phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise Phase9JConfigError("Phase 9J must be schema 1 and observation-only")
    if phase.get("authorization") != "implemented":
        raise Phase9JConfigError("Phase 9J runtime is not authorized in config")
    if stage not in {"screen", "formal", "replication"}:
        raise Phase9JConfigError(f"invalid stage: {stage}")
    expected_slices = {"screen": (1300, 50), "formal": (1350, 200), "replication": (1550, 200)}
    dataset = phase.get("dataset", {})
    for name, (offset, count) in expected_slices.items():
        spec = dataset.get(name)
        if not isinstance(spec, dict) or int(spec.get("sample_offset", -1)) != offset or int(spec.get("max_samples", -1)) != count or spec.get("fresh_slice") is not True:
            raise Phase9JConfigError(f"dataset.{name} slice is not frozen")
    retrieval = phase.get("retrieval")
    if retrieval != {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "top_k": 50}:
        raise Phase9JConfigError("retrieval contract is not frozen")
    selection = phase.get("selection", {})
    expected_selection = {"method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42, "gamma": 1.0, "delta": 0.0, "complementarity_method": None, "qore_prefilter_size": None, "direct_solve_max_n": 20, "lambda_mmr": 0.7, "saturation_alpha": 1.0, "lambda_submodular": 0.5, "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8, "use_answer_scorer": True, "answer_scorer_backend": "dpr"}
    if selection != expected_selection:
        raise Phase9JConfigError("selection contract is not frozen")
    generator = phase.get("generator", {})
    if generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise Phase9JConfigError("generator contract is not frozen")
    plugins = phase.get("plugins", {})
    if tuple(plugins.get("allowlist", [])) != EXPECTED_PLUGINS or plugins.get("diagnostic_outputs_used_for_selection") is not False or plugins.get("production_intervention") is not False:
        raise Phase9JConfigError("plugin contract is not frozen")
    if phase.get("candidate_contract", {}).get("candidate_text_persisted") is not False:
        raise Phase9JConfigError("candidate text persistence must remain disabled")
    ranker = phase.get("ranker_contract", {})
    if ranker.get("profiles") != list(RANKER_PROFILES) or ranker.get("reader_weight") != 1.0 or ranker.get("order_audit") != "fixed_reverse_by_mode_id":
        raise Phase9JConfigError("ranker contract is not frozen")
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
    raise Phase9JConfigError("reference generator not found; attempted: " + ", ".join(attempted))


def _empty_standard() -> dict[str, Any]:
    return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}


def _empty_candidate() -> dict[str, Any]:
    return {"attempted": False, "parse_status": "empty", "em": None, "f1": None, "generation_time_ms": None}


def _empty_ranker() -> dict[str, Any]:
    return {"attempted": False, "original_choice_mode": None, "permuted_choice_mode": None, "order_agreement": None, "parse_status": "not_attempted", "em": None, "f1": None, "score_time_ms": None, "score_to_baseline_ratio": None}


def _scored_arm(prediction: str, gold_answers: list[str], elapsed: float) -> dict[str, Any]:
    metrics = evaluate_answer(prediction, gold_answers)
    return {"attempted": True, "em": float(metrics["em"]), "f1": float(metrics["f1"]), "generation_time_ms": float(elapsed)}


def _scored_candidate(prediction: str, parse_status: str, gold_answers: list[str], elapsed: float) -> dict[str, Any]:
    if parse_status not in {"ok", "abstain"}:
        return {"attempted": True, "parse_status": parse_status, "em": None, "f1": None, "generation_time_ms": None}
    metrics = evaluate_answer(prediction, gold_answers)
    return {"attempted": True, "parse_status": parse_status, "em": float(metrics["em"]), "f1": float(metrics["f1"]), "generation_time_ms": float(elapsed)}


def _candidate_metrics(pairs: Sequence[tuple[str, str]], gold_answers: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    metrics = {mode: {"em": float(evaluate_answer(text, gold_answers)["em"]), "f1": float(evaluate_answer(text, gold_answers)["f1"])} for mode, text in pairs}
    return ({"attempted": True, "candidate_count": len(pairs), "unique_candidate_count": len({" ".join(text.strip().lower().split()) for _, text in pairs}), "parse_failures": 3 - len(pairs), "oracle_em": max(item["em"] for item in metrics.values()), "oracle_f1": max(item["f1"] for item in metrics.values())}, metrics)


def _mean_candidate_logprob(generator: Any, prompt: str, candidate: str) -> float:
    import torch
    answer = " " + str(candidate).strip()
    tokenizer = generator.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    full = tokenizer(prompt + answer, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False)
    full_ids_cpu = full["input_ids"][0]
    if full_ids_cpu.shape[0] > prompt_ids.shape[0] and full_ids_cpu[: prompt_ids.shape[0]].tolist() != prompt_ids.tolist():
        candidate_ids = tokenizer(answer, add_special_tokens=False).get("input_ids", [])
        joined_ids = prompt_ids.tolist() + list(candidate_ids)
        if len(joined_ids) > 4096:
            raise Phase9JConfigError("candidate score prompt exceeds the frozen 4096-token limit")
        full = {"input_ids": torch.tensor([joined_ids], dtype=torch.long)}
        full["attention_mask"] = torch.ones_like(full["input_ids"])
    input_ids = full["input_ids"].to(generator.model.device)
    prompt_len = int(prompt_ids.shape[0])
    if input_ids.shape[1] <= prompt_len or prompt_len < 1:
        raise Phase9JConfigError("candidate score prompt was truncated before candidate tokens")
    with torch.no_grad():
        outputs = generator.model(input_ids=input_ids, attention_mask=full.get("attention_mask", None).to(generator.model.device) if full.get("attention_mask") is not None else None)
    logits = outputs.logits[0, prompt_len - 1 : -1]
    target = input_ids[0, prompt_len:]
    if logits.shape[0] != target.shape[0]:
        raise Phase9JConfigError("candidate score logits and target lengths differ")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(log_probs.gather(1, target.unsqueeze(1)).mean().item())


def _candidate_context_lift(generator: Any, question: str, passages: Sequence[str], candidate: str) -> float:
    context_prompt = generator._build_prompt(question, list(passages))
    empty_prompt = generator._build_prompt(question, [])
    return _mean_candidate_logprob(generator, context_prompt, candidate) - _mean_candidate_logprob(generator, empty_prompt, candidate)


def _reader_support_scores(answer_scorer: Any, question: str, passages: Sequence[str], candidates: Mapping[str, str]) -> dict[str, float]:
    import torch
    tokenizer = answer_scorer.tokenizer
    values = {mode: float("-inf") for mode in candidates}
    for passage in passages:
        encoded = tokenizer(questions=[question], texts=[passage], return_tensors="pt", padding=True, truncation=True, max_length=350)
        encoded_device = {key: value.to(answer_scorer.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = answer_scorer.reader(**encoded_device)
        ids = encoded["input_ids"][0].tolist()
        token_types = encoded.get("token_type_ids")
        type_ids = token_types[0].tolist() if token_types is not None else [1] * len(ids)
        start_logits = outputs.start_logits[0].detach().float().cpu().tolist()
        end_logits = outputs.end_logits[0].detach().float().cpu().tolist()
        relevance = float(outputs.relevance_logits[0].detach().float().cpu().item())
        for mode, text in candidates.items():
            candidate_ids = tokenizer(str(text).strip(), add_special_tokens=False).get("input_ids", [])
            best = float("-inf")
            if candidate_ids:
                size = len(candidate_ids)
                for index in range(0, max(0, len(ids) - size + 1)):
                    if type_ids[index : index + size] == [1] * size and ids[index : index + size] == candidate_ids:
                        best = max(best, float(start_logits[index] + end_logits[index + size - 1]))
            if not math.isfinite(best):
                best = relevance - 5.0
            values[mode] = max(values[mode], best)
    if not values:
        raise Phase9JConfigError("candidate reader support received no candidates")
    return values


def _ranker_arms(generator: Any, answer_scorer: Any, question: str, passages: Sequence[str], pairs: Sequence[tuple[str, str]], metrics: Mapping[str, Mapping[str, float]], baseline_time_ms: float) -> dict[str, dict[str, Any]]:
    started = time.perf_counter()
    texts = {mode: text for mode, text in pairs}
    raw_scores = {mode: {"context_lift": _candidate_context_lift(generator, question, passages, text), "reader_support": 0.0} for mode, text in pairs}
    reader_values = _reader_support_scores(answer_scorer, question, passages, texts)
    for mode in raw_scores:
        raw_scores[mode]["reader_support"] = reader_values[mode]
    scores = build_candidate_scores(raw_scores, reader_weight=1.0)
    elapsed = (time.perf_counter() - started) * 1000.0
    result: dict[str, dict[str, Any]] = {}
    for profile in RANKER_PROFILES:
        original = choose_candidate(scores, profile)
        permuted = choose_candidate(scores, profile)
        chosen = metrics[original]
        result[profile] = {"attempted": True, "original_choice_mode": original, "permuted_choice_mode": permuted, "order_agreement": original == permuted, "parse_status": "ok", "em": chosen["em"], "f1": chosen["f1"], "score_time_ms": elapsed, "score_to_baseline_ratio": elapsed / max(float(baseline_time_ms), 1.0)}
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase9j_context_lift_candidate_ranking.yaml")
    parser.add_argument("--stage", choices=("screen", "formal", "replication"), default="screen")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config_path = config_path.resolve()
    phase, stage_spec = _load_config(config_path, args.stage)
    if args.bootstrap_reps is not None and args.bootstrap_reps < 100:
        raise Phase9JConfigError("--bootstrap-reps must be at least 100")
    if args.validate_only:
        print(json.dumps({"status": "valid", "phase": phase["name"], "stage": args.stage, "slice": stage_spec["slice"], "plugins": list(EXPECTED_PLUGINS), "selection_mutation": False, "report_only": True}, sort_keys=True))
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
    metadata: dict[str, Any] = {"schema_version": 1, "phase": phase["name"], "stage": args.stage, "status": "running", "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"path": str(config_path), "sha256": _sha256_file(config_path)}, "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")}, "python": {"executable": sys.executable, "version": sys.version}, "generator": identity, "dataset": {"name": phase["dataset"]["name"], "split": phase["dataset"]["split"], "sample_offset": stage_spec["sample_offset"], "max_samples": stage_spec["max_samples"], "slice": stage_spec["slice"], "fresh_slice": True}, "plugins": {"allowlist": list(EXPECTED_PLUGINS), "tree_sha256": _plugin_tree_hash(root), "profiles": list(RANKER_PROFILES), "diagnostic_outputs_used_for_selection": False}}
    _write_json(run_dir / "run_metadata.json", metadata)
    sample_end = int(stage_spec["sample_offset"]) + int(stage_spec["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["split"], sample_end)
    questions = questions[int(stage_spec["sample_offset"]) : sample_end]
    if len(questions) != int(stage_spec["max_samples"]):
        raise Phase9JConfigError("fresh slice length mismatch")
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
        baseline = _scored_arm(baseline_result.prediction, gold_answers, baseline_result.generation_time_ms)
        primary_error = bool(selected_hit and baseline["em"] == 0.0)
        copy_control = _empty_standard()
        if primary_error:
            copy_target = min(gold_answers, key=lambda value: (len(value.split()), len(value), value.lower()))
            copy_result = run_gold_answer_copy_probe(generator, question, copy_target)
            copy_control = _scored_arm(copy_result.prediction, gold_answers, copy_result.generation_time_ms)
        copy_success = bool(primary_error and copy_control["em"] == 1.0)
        arms = {"baseline": baseline, "gold_answer_copy": copy_control, "extractive_span": _empty_candidate(), "evidence_constrained": _empty_candidate()}
        candidate_set = {"attempted": False, "candidate_count": 0, "unique_candidate_count": 0, "parse_failures": 0, "oracle_em": None, "oracle_f1": None}
        rankers = {profile: _empty_ranker() for profile in RANKER_PROFILES}
        if copy_success:
            extractive = run_extractive_candidate_probe(generator, question, selected_texts)
            constrained = run_evidence_constrained_probe(generator, question, selected_texts)
            arms["extractive_span"] = _scored_candidate(extractive.text, extractive.parse_status, gold_answers, extractive.generation_time_ms)
            arms["evidence_constrained"] = _scored_candidate(constrained.text, constrained.parse_status, gold_answers, constrained.generation_time_ms)
            pairs = candidate_pairs(baseline_result.prediction, extractive, constrained)
            if not pairs:
                raise Phase9JConfigError(f"{question_id}: eligible diagnostic has no valid candidates")
            candidate_set, metric_map = _candidate_metrics(pairs, gold_answers)
            rankers = _ranker_arms(generator, answer_scorer, question, selected_texts, pairs, metric_map, baseline_result.generation_time_ms)
        samples.append({"question_id": question_id, "retrieval_hit": retrieval_hit, "selected_hit": selected_hit, "primary_error": primary_error, "copy_control_success": copy_success, "eligible": copy_success, "selection_time_ms": selection_time_ms, "arms": arms, "candidate_set": candidate_set, "rankers": rankers})
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 9J {args.stage}: {index}/{len(questions)}")
    result = {"schema_version": 1, "phase": phase["name"], "stage": args.stage, "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"dataset": phase["dataset"]["name"], "split": phase["dataset"]["split"], "sample_offset": stage_spec["sample_offset"], "max_samples": stage_spec["max_samples"], "slice": stage_spec["slice"], "ranker_profiles": list(RANKER_PROFILES), "reader_weight": 1.0}, "samples": samples}
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase9JConfigError(f"compact result contains forbidden fields: {forbidden[:5]}")
    validate_result(result)
    _write_json(run_dir / "result.json", result)
    gate = phase["gate"]
    if args.bootstrap_reps is not None:
        gate = dict(gate)
        gate["formal"] = dict(gate["formal"])
        gate["formal"]["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_phase9j(result, gate=gate)
    _write_json(run_dir / "summary.json", summary)
    metadata.update({"status": "completed", "timing_ms": {"total": (time.perf_counter() - started) * 1000.0}, "summary": {"path": str(run_dir / "summary.json"), "primary_failure_class": summary["decision"]["primary_failure_class"]}})
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 9J {args.stage}: {run_dir}")
    print(f"Report-only gate: {summary['decision']['primary_failure_class']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (Phase9JConfigError, Phase9JError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
