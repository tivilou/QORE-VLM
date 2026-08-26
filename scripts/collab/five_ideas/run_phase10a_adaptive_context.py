#!/usr/bin/env python3
"""Run the collaborator-only Phase 10A adaptive-context screen."""

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

_SCRIPT_PATH = Path(__file__).resolve()
for _candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_candidate / "configs").is_dir() and (_candidate / "applications").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import yaml

from applications.rag.adaptive_context import build_wide_context, prefix_digest, risk_features, route_extra_context
from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation import evaluate_answer
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from scripts.collab.five_ideas.phase10a_metrics import Phase10AError, summarize_phase10a


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_ARMS = ("baseline_k5", "always_wide", "adaptive")
FORBIDDEN_FIELDS = {"question", "passages", "gold_answers", "prediction", "raw_prompt", "text"}


class Phase10AConfigError(RuntimeError):
    pass


def _root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase10AConfigError("cannot locate project root")


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


def _find_forbidden(value: Any, path: str = "$root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_FIELDS:
                found.append(child_path)
            found.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden(child, f"{path}[{index}]"))
    return found


def _load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != "phase10a_adaptive_context" or int(phase.get("schema_version", -1)) != 1:
        raise Phase10AConfigError("unexpected Phase 10A configuration")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False or phase.get("authorization") != "implemented_observation_only":
        raise Phase10AConfigError("Phase 10A must remain observation-only")
    dataset = phase.get("dataset", {})
    if (dataset.get("name"), dataset.get("split"), int(dataset.get("sample_offset", -1)), int(dataset.get("max_samples", -1)), dataset.get("fresh_slice")) != ("nq_open", "validation", 2650, 50, True):
        raise Phase10AConfigError("dataset slice is not frozen")
    retrieval = phase.get("retrieval", {})
    if retrieval != {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "top_k": 50}:
        raise Phase10AConfigError("retrieval contract is not frozen")
    selection = phase.get("selection", {})
    for key, expected in {"method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42, "gamma": 1.0, "delta": 0.0, "use_answer_scorer": True, "answer_scorer_backend": "dpr"}.items():
        if selection.get(key) != expected:
            raise Phase10AConfigError(f"selection.{key} is not frozen")
    generator = phase.get("generator", {})
    if generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise Phase10AConfigError("generator contract is not frozen")
    routing = phase.get("routing", {})
    if int(routing.get("extra_count", -1)) != 3 or float(routing.get("risk_threshold", -1)) != 0.55 or routing.get("selection_feedback") is not False or routing.get("gold_or_evaluator_available") is not False:
        raise Phase10AConfigError("routing contract is not frozen")
    if tuple(phase.get("arms", [])) != EXPECTED_ARMS:
        raise Phase10AConfigError("arm order is not frozen")
    return phase


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
    attempted = []
    for candidate in candidates:
        attempted.append(str(candidate))
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate if (candidate / "config.json").is_file() else snapshot
        if (resolved / "config.json").is_file():
            return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "model_path": str(resolved.resolve()), "config_sha256": _sha256(resolved / "config.json"), "resolution": "cli_override" if override else "local_cache"}
    raise Phase10AConfigError("reference generator not found; attempted: " + ", ".join(attempted))


def _arm(prediction: str, gold_answers: list[str], elapsed_ms: float, *, applied: bool, prefix_ok: bool, risk: Mapping[str, float], extra_count: int) -> dict[str, Any]:
    metrics = evaluate_answer(prediction, gold_answers)
    return {"em": float(metrics["em"]), "f1": float(metrics["f1"]), "generation_time_ms": float(elapsed_ms), "applied": bool(applied), "prefix_parity": bool(prefix_ok), "risk_score": float(risk["risk_score"]), "extra_count": int(extra_count)}


def _prefix_parity(
    retrieved_indices: Sequence[int],
    selected_local: Sequence[int],
    extra_local: Sequence[int],
    selected_digest: str,
) -> bool:
    """Check wide-context prefix identity through local and global passage IDs."""
    selected = tuple(int(value) for value in selected_local)
    extras = tuple(int(value) for value in extra_local)
    wide_indices = selected + extras
    if len(selected) != 5 or len(wide_indices) != len(selected) + len(extras):
        return False
    if wide_indices[: len(selected)] != selected:
        return False
    if len(set(wide_indices)) != len(wide_indices):
        return False
    if any(index < 0 or index >= len(retrieved_indices) for index in wide_indices):
        return False
    selected_global = tuple(int(retrieved_indices[index]) for index in selected)
    prefix_global = tuple(int(retrieved_indices[index]) for index in wide_indices[: len(selected)])
    if prefix_global != selected_global:
        return False
    return prefix_digest(retrieved_indices, wide_indices[: len(selected)]) == selected_digest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _root()
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase10a_adaptive_context.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _root()
    config_path = (args.config if args.config.is_absolute() else root / args.config).resolve()
    phase = _load(config_path)
    generator_identity = _resolve_generator(root, phase["generator"], args.model_path)
    if args.validate_only:
        print(json.dumps({"status": "valid", "dataset": phase["dataset"]["slice"], "arms": EXPECTED_ARMS, "generator": generator_identity}, sort_keys=True))
        return None
    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {"schema_version": 1, "phase": phase["name"], "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"path": str(config_path), "sha256": _sha256(config_path)}, "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")}, "python": {"executable": sys.executable, "version": sys.version}, "generator": generator_identity, "dataset": phase["dataset"], "retrieval": phase["retrieval"], "selection": phase["selection"], "routing": phase["routing"]}
    _write(run_dir / "run_metadata.json", metadata)
    requested = int(phase["dataset"]["sample_offset"]) + int(phase["dataset"]["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["split"], requested)[int(phase["dataset"]["sample_offset"]):requested]
    if len(questions) != int(phase["dataset"]["max_samples"]):
        raise Phase10AConfigError("fresh slice size mismatch")
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend="dpr")
    generator = Generator(generator_identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    arms: dict[str, dict[str, Any]] = {name: {"config": {"arm": name}, "samples": []} for name in EXPECTED_ARMS}
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        question = str(item["question"])
        question_id = str(item["id"])
        gold_answers = [str(value) for value in item.get("answers", []) if str(value).strip()]
        query_embedding = encoder.encode_queries([question])[0]
        retrieved_idx, retrieved_embeddings, retrieved_texts, _ = corpus_manager.retrieve_with_embeddings(query_embedding, int(phase["retrieval"]["top_k"]))
        answer_scores = scorer.score_passages(question, retrieved_texts)
        selected_local = select_passages(query_embedding, retrieved_embeddings, K=5, method="qore", num_reads=100, lam=2.0, gamma=1.0, delta=0.0, answer_scorer=scorer, passage_texts=retrieved_texts, question=question, seed=42, relevance_scores=answer_scores)
        selected_texts = tuple(retrieved_texts[int(value)] for value in selected_local)
        wide_texts, extra_local = build_wide_context(retrieved_texts, selected_local, extra_count=3)
        risk = risk_features(answer_scores, selected_local, selected_texts)
        applied = route_extra_context(risk["risk_score"], float(phase["routing"]["risk_threshold"]))
        digest = prefix_digest(retrieved_idx, selected_local)
        baseline_start = time.perf_counter()
        baseline_prediction = generator.generate(question, list(selected_texts))
        baseline_elapsed = (time.perf_counter() - baseline_start) * 1000.0
        wide_start = time.perf_counter()
        wide_prediction = generator.generate(question, list(wide_texts))
        wide_elapsed = (time.perf_counter() - wide_start) * 1000.0
        prefix_ok = tuple(wide_texts[: len(selected_texts)]) == selected_texts and _prefix_parity(retrieved_idx, selected_local, extra_local, digest)
        baseline_sample = _arm(baseline_prediction, gold_answers, baseline_elapsed, applied=False, prefix_ok=prefix_ok, risk=risk, extra_count=0)
        wide_sample = _arm(wide_prediction, gold_answers, wide_elapsed, applied=True, prefix_ok=prefix_ok, risk=risk, extra_count=len(extra_local))
        adaptive_sample = dict(wide_sample if applied else baseline_sample)
        adaptive_sample["applied"] = applied
        adaptive_sample["extra_count"] = len(extra_local) if applied else 0
        for name, sample in (("baseline_k5", baseline_sample), ("always_wide", wide_sample), ("adaptive", adaptive_sample)):
            sample["question_id"] = question_id
            sample["risk_score"] = float(risk["risk_score"])
            sample["selected_confidence"] = float(risk["selected_confidence"])
            sample["score_margin"] = float(risk["score_margin"])
            sample["selected_duplication"] = float(risk["selected_duplication"])
            sample["selected_prefix_digest"] = digest
            arms[name]["samples"].append(sample)
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 10A adaptive context: {index}/{len(questions)}")
    result = {"schema_version": 1, "phase": phase["name"], "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {"dataset": phase["dataset"], "retrieval": phase["retrieval"], "selection": phase["selection"], "risk_threshold": phase["routing"]["risk_threshold"], "extra_count": phase["routing"]["extra_count"]}, "arms": arms}
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase10AConfigError(f"compact result contains forbidden fields: {forbidden[:5]}")
    _write(run_dir / "result.json", result)
    gate = dict(phase["gate"])
    if args.bootstrap_reps is not None:
        gate["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_phase10a(result, gate=gate)
    _write(run_dir / "summary.json", summary)
    metadata.update({"status": "completed", "timing_ms": {"total": (time.perf_counter() - started) * 1000.0}, "summary": {"path": str(run_dir / "summary.json"), "decision": summary["decision"]}})
    _write(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 10A adaptive-context screen: {run_dir}")
    print(f"Decision: {summary['decision']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (Phase10AConfigError, Phase10AError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
