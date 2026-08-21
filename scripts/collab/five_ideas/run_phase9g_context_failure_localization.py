#!/usr/bin/env python3
"""Run the collaborator-only Phase 9G context-failure localization diagnostic."""

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

# Keep direct execution equivalent to the repository wrapper by making the
# project root importable before loading stable RAG components.
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
    from scripts.collab.five_ideas.phase9g_probe import (
        CONTEXT_VARIANT_PROFILE,
        GOLD_ANSWER_COPY_PROFILE,
        build_context_variants,
        run_context_probe,
        run_gold_answer_copy_probe,
    )
    from scripts.collab.five_ideas.phase9g_metrics import (
        Phase9GError,
        summarize_context_failure,
    )
except ImportError:  # pragma: no cover - direct script execution
    from phase9g_probe import (
        CONTEXT_VARIANT_PROFILE,
        GOLD_ANSWER_COPY_PROFILE,
        build_context_variants,
        run_context_probe,
        run_gold_answer_copy_probe,
    )
    from phase9g_metrics import Phase9GError, summarize_context_failure


class Phase9GConfigError(RuntimeError):
    """Raised when the frozen Phase 9G contract is malformed."""


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_PLUGINS = (
    "frozen_baseline_observer",
    "gold_answer_copy_control",
    "full_passage_plain_probe",
    "full_passage_highlight_probe",
    "answer_passage_first_probe",
    "retrieved_answer_oracle_probe",
    "context_failure_combiner",
)
FORBIDDEN_FIELDS = {
    "question", "passages", "gold_answers", "prediction", "raw_prompt",
    "oracle_passages", "gold_answer",
}


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase9GConfigError("cannot locate project root")


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
        root / "scripts/collab/five_ideas/phase9g_probe.py",
        root / "scripts/collab/five_ideas/phase9g_metrics.py",
        root / "scripts/collab/five_ideas/run_phase9g_context_failure_localization.py",
    )
    entries = [
        {"path": str(path.relative_to(root)), "sha256": _sha256_file(path)}
        for path in paths
    ]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or int(phase.get("schema_version", -1)) != 1:
        raise Phase9GConfigError("configuration must contain phase.schema_version: 1")
    required = {
        "name", "schema_version", "diagnostic_only", "selection_mutation",
        "authorization", "parent_phase", "dataset", "split", "sample_offset",
        "max_samples", "fresh_slice", "corpus_mode", "wiki_dpr_config",
        "wiki_dpr_nprobe", "top_k_retrieval", "selection", "generator",
        "plugins", "gate", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise Phase9GConfigError(f"configuration missing keys: {missing}")
    if phase["name"] != "phase9g_context_failure_localization":
        raise Phase9GConfigError("unexpected Phase 9G name")
    if phase["diagnostic_only"] is not True or phase["selection_mutation"] is not False:
        raise Phase9GConfigError("Phase 9G must be diagnostic-only with frozen selection")
    if phase["authorization"] != "one_targeted_followup_only" or phase["parent_phase"] != "phase9f_evidence_sufficiency_localization":
        raise Phase9GConfigError("Phase 9G authorization/parent is not frozen")
    frozen = (
        phase["dataset"], phase["split"], int(phase["sample_offset"]),
        int(phase["max_samples"]), phase["corpus_mode"], phase["wiki_dpr_config"],
        int(phase["wiki_dpr_nprobe"]), int(phase["top_k_retrieval"]),
    )
    if frozen != (
        "nq_open", "validation", 750, 50, "wiki_dpr",
        "psgs_w100.nq.compressed", 64, 50,
    ) or phase["fresh_slice"] is not True:
        raise Phase9GConfigError("Phase 9G dataset/retrieval contract is not frozen")
    selection = phase["selection"]
    expected_selection = {
        "method": "qore", "K": 5, "num_reads": 100, "lam": 2.0,
        "seed": 42, "gamma": 1.0, "delta": 0.0,
        "complementarity_method": None, "qore_prefilter_size": None,
        "direct_solve_max_n": 20, "lambda_mmr": 0.7,
        "saturation_alpha": 1.0, "lambda_submodular": 0.5,
        "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8,
        "use_answer_scorer": True, "answer_scorer_backend": "dpr",
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise Phase9GConfigError(f"selection.{key} is not frozen")
    generator = phase["generator"]
    if (
        not isinstance(generator, dict)
        or generator.get("model_id") != MODEL_ID
        or generator.get("revision") != MODEL_REVISION
        or int(generator.get("max_new_tokens", -1)) != 32
    ):
        raise Phase9GConfigError("generator identity/settings are not frozen")
    plugins = phase["plugins"]
    if not isinstance(plugins, dict) or tuple(plugins.get("allowlist", [])) != EXPECTED_PLUGINS:
        raise Phase9GConfigError("plugin allowlist/order is not frozen")
    if plugins.get("context_variant_profile") != CONTEXT_VARIANT_PROFILE:
        raise Phase9GConfigError("unexpected context variant profile")
    if plugins.get("gold_answer_copy_profile") != GOLD_ANSWER_COPY_PROFILE:
        raise Phase9GConfigError("unexpected gold-answer copy profile")
    gate = phase["gate"]
    for key in ("minimum_primary_errors", "minimum_copy_control_successes"):
        if int(gate.get(key, -1)) < 1:
            raise Phase9GConfigError(f"gate.{key} must be positive")
    for key in ("minimum_dominant_fraction", "minimum_dominance_margin"):
        if not 0.0 <= float(gate.get(key, -1.0)) <= 1.0:
            raise Phase9GConfigError(f"gate.{key} must be in [0,1]")
    if int(gate.get("bootstrap_repetitions", 0)) < 100:
        raise Phase9GConfigError("gate.bootstrap_repetitions must be at least 100")
    outputs = phase["outputs"]
    if not isinstance(outputs, dict) or outputs.get("compact_only") is not True:
        raise Phase9GConfigError("Phase 9G outputs must be compact_only")
    return phase


def _resolve_generator(
    root: Path, specification: Mapping[str, Any], override: str | None
) -> dict[str, Any]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    for raw in specification.get("model_path_candidates", []):
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    candidates.extend([
        root / "models" / "llama3-8b",
        Path.home() / ".cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct",
    ])
    if os.environ.get("HF_HOME"):
        candidates.append(
            Path(os.environ["HF_HOME"]) / "hub/models--NousResearch--Meta-Llama-3-8B-Instruct"
        )
    attempted: list[str] = []
    for candidate in candidates:
        attempted.append(str(candidate))
        direct = candidate / "config.json"
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate.resolve() if direct.is_file() else snapshot.resolve()
        if not (resolved / "config.json").is_file():
            continue
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_path": str(resolved),
            "config_sha256": _sha256_file(resolved / "config.json"),
            "resolution": "cli_override" if override else "local_cache",
        }
    raise Phase9GConfigError("reference generator not found; attempted: " + ", ".join(attempted))


def _empty_arm() -> dict[str, Any]:
    return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}


def _scored_arm(prediction: str, gold_answers: list[str], elapsed: float) -> dict[str, Any]:
    metrics = evaluate_answer(prediction, gold_answers)
    return {
        "attempted": True,
        "em": metrics["em"],
        "f1": metrics["f1"],
        "generation_time_ms": elapsed,
    }


def _probe_arm(generator: Generator, question: str, passages: Sequence[str], gold_answers: list[str]) -> dict[str, Any]:
    result = run_context_probe(generator, question, passages)
    return _scored_arm(result.prediction, gold_answers, result.generation_time_ms)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/experiments/phase9g_context_failure_localization.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config_path = config_path.resolve()
    phase = _load_config(config_path)
    if args.bootstrap_reps is not None and args.bootstrap_reps < 100:
        raise Phase9GConfigError("--bootstrap-reps must be at least 100")
    generator_identity = _resolve_generator(root, phase["generator"], args.model_path)
    if args.validate_only:
        print(json.dumps({
            "status": "valid",
            "phase": phase["name"],
            "slice": "nq_open validation[750:800]",
            "plugins": list(EXPECTED_PLUGINS),
            "model_path": generator_identity["model_path"],
            "selection_mutation": False,
            "report_only": True,
        }, sort_keys=True))
        return None

    output_root = Path(args.output_root or phase["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase["name"],
        "status": "running",
        "diagnostic_only": True,
        "selection_mutation": False,
        "report_only": True,
        "parent_phase": phase["parent_phase"],
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "status": _git(root, "status", "--short", "--branch"),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "generator": generator_identity,
        "dataset": {
            "name": phase["dataset"], "split": phase["split"],
            "sample_offset": phase["sample_offset"], "max_samples": phase["max_samples"],
            "fresh_slice": True,
        },
        "retrieval": {
            "corpus_mode": phase["corpus_mode"],
            "wiki_dpr_config": phase["wiki_dpr_config"],
            "nprobe": phase["wiki_dpr_nprobe"],
            "top_k_retrieval": phase["top_k_retrieval"],
        },
        "selection": phase["selection"],
        "plugins": {
            "allowlist": list(EXPECTED_PLUGINS),
            "tree_sha256": _plugin_tree_hash(root),
            "diagnostic_outputs_used_for_selection": False,
        },
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    requested = int(phase["sample_offset"]) + int(phase["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"], phase["split"], requested)
    questions = questions[int(phase["sample_offset"]):requested]
    if len(questions) != int(phase["max_samples"]):
        raise Phase9GConfigError(f"fresh slice returned {len(questions)} questions")

    np.random.seed(int(phase["selection"]["seed"]))
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager("wiki_dpr", {
        "wiki_dpr_config": phase["wiki_dpr_config"],
        "nprobe": int(phase["wiki_dpr_nprobe"]),
    })
    corpus_manager.build(questions)
    answer_scorer = make_answer_scorer(backend="dpr")
    generator = Generator(
        generator_identity["model_path"],
        max_new_tokens=int(phase["generator"]["max_new_tokens"]),
        use_chat_template=True,
    )

    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        question = str(item["question"])
        question_id = str(item["id"])
        gold_answers = [str(value) for value in item.get("answers", []) if str(value).strip()]
        if not gold_answers:
            raise Phase9GConfigError(f"{question_id} has no non-empty gold answer")
        query_embedding = encoder.encode_queries([question])[0]
        _, retrieved_embeddings, retrieved_texts, _ = corpus_manager.retrieve_with_embeddings(
            query_embedding, int(phase["top_k_retrieval"])
        )
        if len(retrieved_texts) != int(phase["top_k_retrieval"]):
            raise Phase9GConfigError(f"{question_id}: incomplete retrieval")
        answer_scores = answer_scorer.score_passages(question, retrieved_texts)
        selection_started = time.perf_counter()
        selected_local = select_passages(
            query_embedding, retrieved_embeddings,
            K=int(phase["selection"]["K"]), method="qore",
            num_reads=int(phase["selection"]["num_reads"]),
            lam=float(phase["selection"]["lam"]), gamma=float(phase["selection"]["gamma"]),
            delta=float(phase["selection"]["delta"]),
            complementarity_method=phase["selection"]["complementarity_method"],
            qore_prefilter_size=phase["selection"]["qore_prefilter_size"],
            direct_solve_max_n=int(phase["selection"]["direct_solve_max_n"]),
            lambda_mmr=float(phase["selection"]["lambda_mmr"]),
            saturation_alpha=float(phase["selection"]["saturation_alpha"]),
            lambda_submodular=float(phase["selection"]["lambda_submodular"]),
            dpp_quality_scale=float(phase["selection"]["dpp_quality_scale"]),
            dpp_jitter=float(phase["selection"]["dpp_jitter"]),
            answer_scorer=answer_scorer, passage_texts=retrieved_texts,
            question=question, seed=int(phase["selection"]["seed"]),
            relevance_scores=answer_scores,
        )
        selection_time_ms = (time.perf_counter() - selection_started) * 1000.0
        selected_texts = [retrieved_texts[int(value)] for value in selected_local]
        retrieval_hit = any(
            answer_has_match_in_text(answer, passage)
            for passage in retrieved_texts for answer in gold_answers
        )
        selected_hit = any(
            answer_has_match_in_text(answer, passage)
            for passage in selected_texts for answer in gold_answers
        )
        variants = build_context_variants(selected_texts, retrieved_texts, gold_answers)
        if selected_hit != (variants.selected_match_count > 0):
            raise Phase9GConfigError(f"{question_id}: selected match count disagrees with selected hit")
        if retrieval_hit != (variants.oracle_match_count > 0):
            raise Phase9GConfigError(f"{question_id}: oracle match count disagrees with retrieval hit")

        baseline = _probe_arm(generator, question, selected_texts, gold_answers)
        primary_error = bool(selected_hit and baseline["em"] == 0.0)
        copy = _empty_arm()
        if primary_error:
            copy_target = min(gold_answers, key=lambda value: (len(value.split()), len(value), value.lower()))
            copy_result = run_gold_answer_copy_probe(generator, question, copy_target)
            copy = _scored_arm(copy_result.prediction, gold_answers, copy_result.generation_time_ms)
        eligible = bool(primary_error and copy["em"] == 1.0)
        context_arms = {name: _empty_arm() for name in (
            "full_plain", "full_highlight", "answer_first", "retrieved_answer_oracle"
        )}
        if eligible:
            contexts = {
                "full_plain": variants.full_plain,
                "full_highlight": variants.full_highlight,
                "answer_first": variants.answer_first,
                "retrieved_answer_oracle": variants.retrieved_answer_oracle,
            }
            for name, context in contexts.items():
                context_arms[name] = _probe_arm(generator, question, context, gold_answers)

        samples.append({
            "question_id": question_id,
            "retrieval_hit": retrieval_hit,
            "selected_hit": selected_hit,
            "selected_match_count": variants.selected_match_count,
            "oracle_match_count": variants.oracle_match_count,
            "answer_first_changed": variants.answer_first_changed,
            "selection_time_ms": selection_time_ms,
            "arms": {"baseline": baseline, "gold_answer_copy": copy, **context_arms},
        })
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 9G context probes: {index}/{len(questions)}")

    result = {
        "schema_version": 1,
        "phase": phase["name"],
        "diagnostic_only": True,
        "selection_mutation": False,
        "report_only": True,
        "config": {
            "dataset": phase["dataset"], "split": phase["split"],
            "sample_offset": phase["sample_offset"], "max_samples": phase["max_samples"],
            "corpus_mode": phase["corpus_mode"], "wiki_dpr_config": phase["wiki_dpr_config"],
            "wiki_dpr_nprobe": phase["wiki_dpr_nprobe"], "top_k_retrieval": phase["top_k_retrieval"],
            "method": "qore", "K": 5, "num_reads": 100, "lam": 2.0,
            "seed": 42, "gamma": 1.0, "use_answer_scorer": True,
            "answer_scorer_backend": "dpr", "max_new_tokens": 32,
        },
        "samples": samples,
    }
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase9GConfigError(f"compact result contains forbidden fields: {forbidden[:5]}")
    _write_json(run_dir / "result.json", result)
    gate = dict(phase["gate"])
    if args.bootstrap_reps is not None:
        gate["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_context_failure(result, gate=gate)
    _write_json(run_dir / "summary.json", summary)
    metadata.update({
        "status": "completed",
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "summary": {
            "path": str(run_dir / "summary.json"),
            "primary_failure_class": summary["decision"]["primary_failure_class"],
        },
    })
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 9G context-failure localization: {run_dir}")
    print(f"Report-only gate: {summary['decision']['primary_failure_class']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (Phase9GConfigError, Phase9GError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
