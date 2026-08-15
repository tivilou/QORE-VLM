#!/usr/bin/env python3
"""Run the collaborator-only Phase 7A answer-identity diagnostic pilot.

The runner deliberately stops at diagnostics. It does not register an answer-
conditioned QUBO enhancer and it never writes raw passages or predictions to
Exchange. Full Wiki-DPR retrieval is expected to run on the collaborator host.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from applications.rag.answer_evidence import (
    build_answer_evidence_matrices,
    select_counterfactual_swap,
    stable_identifier,
    summarize_answer_evidence,
)
from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation.metrics import evaluate_answer
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages


class DiagnosticError(RuntimeError):
    """Raised when a diagnostic precondition is not met."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("schema_version") != 1:
        raise DiagnosticError("configuration must contain phase.schema_version: 1")
    required = {
        "name", "dataset", "split", "max_samples", "seed", "corpus_mode",
        "wiki_dpr_config", "wiki_dpr_nprobe", "top_k_retrieval", "selection_K",
        "selection_method", "answer_scorer_backend", "answer_top_m",
        "max_answer_tokens", "high_conflict_threshold", "generator", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise DiagnosticError(f"configuration missing keys: {missing}")
    if phase["corpus_mode"] != "wiki_dpr":
        raise DiagnosticError("Phase 7A collaborator runner requires corpus_mode=wiki_dpr")
    if not 1 <= int(phase["max_samples"]) <= 100:
        raise DiagnosticError("configuration max_samples must be between 1 and 100")
    return phase


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/experiments/phase7a_answer_identity_diagnostics.yaml",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config_path = config_path.resolve()
    phase = _load_config(config_path)
    max_samples = int(args.max_samples or phase["max_samples"])
    seed = int(phase["seed"] if args.seed is None else args.seed)
    if not 1 <= max_samples <= 100:
        raise DiagnosticError("--max-samples must be between 1 and 100")
    if seed < 0:
        raise DiagnosticError("--seed must be non-negative")

    if not args.allow_dirty:
        tracked_dirty = (
            subprocess.run(["git", "diff", "--quiet"], cwd=project_root).returncode != 0
            or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=project_root).returncode != 0
        )
        if tracked_dirty:
            raise DiagnosticError("tracked worktree changes detected; use --allow-dirty for debugging")

    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else project_root / output_root
    output_root = output_root.resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    np.random.seed(seed)
    started = time.perf_counter()
    generator_config = dict(phase["generator"])
    generation_enabled = bool(generator_config.get("enabled", True)) and not args.skip_generation
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase["name"],
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "dataset": {
            "name": phase["dataset"],
            "split": phase["split"],
            "max_samples": max_samples,
            "corpus_mode": phase["corpus_mode"],
            "wiki_dpr_config": phase["wiki_dpr_config"],
            "nprobe": int(phase["wiki_dpr_nprobe"]),
        },
        "seed": seed,
        "python": {"executable": sys.executable, "version": sys.version},
        "git": {"commit": _git(project_root, "rev-parse", "HEAD"), "status": _git(project_root, "status", "--short", "--branch")},
        "generation": {"enabled": generation_enabled, "model_path": generator_config.get("model_path") if generation_enabled else None},
        "diagnostic_only": True,
    }
    (run_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Loading {phase['dataset']} ({phase['split']}), max_samples={max_samples}...")
    questions = load_dataset_for_rag(phase["dataset"], phase["split"], max_samples)
    if len(questions) != max_samples:
        raise DiagnosticError(f"loader returned {len(questions)} questions, expected {max_samples}")

    print("Loading DPR encoder and Wiki-DPR corpus...")
    encoder = make_encoder("dpr", device=args.device)
    corpus_manager = make_corpus_manager(
        "wiki_dpr",
        {
            "wiki_dpr_config": phase["wiki_dpr_config"],
            "nprobe": int(phase["wiki_dpr_nprobe"]),
        },
    )
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend=phase["answer_scorer_backend"], device=args.device)
    generator = None
    if generation_enabled:
        generator = Generator(
            generator_config["model_path"],
            device=args.device,
            max_new_tokens=int(generator_config.get("max_new_tokens", 32)),
            use_chat_template=True,
        )

    sample_reports: list[dict[str, Any]] = []
    selection_times: list[float] = []
    scorer_times: list[float] = []
    generation_times: list[float] = []
    for index, item in enumerate(questions):
        question = item["question"]
        question_id = stable_identifier(str(item["id"]), namespace="question")
        query_embedding = encoder.encode_queries([question])[0]
        _, retrieved_embeddings, passages, retrieval_scores = corpus_manager.retrieve_with_embeddings(
            query_embedding, int(phase["top_k_retrieval"])
        )
        score_started = time.perf_counter()
        answer_scores, hypotheses = scorer.score_passages_with_hypotheses(
            question,
            passages,
            top_m=int(phase["answer_top_m"]),
            max_answer_tokens=int(phase["max_answer_tokens"]),
        )
        scorer_times.append((time.perf_counter() - score_started) * 1000.0)
        selection_started = time.perf_counter()
        selected = np.asarray(
            select_passages(
                query_embedding,
                retrieved_embeddings,
                K=int(phase["selection_K"]),
                method=phase["selection_method"],
                relevance_scores=answer_scores,
                seed=seed,
            ),
            dtype=np.int64,
        )
        selection_times.append((time.perf_counter() - selection_started) * 1000.0)
        matrices = build_answer_evidence_matrices(
            passages,
            hypotheses,
            passage_confidence=answer_scores,
        )
        evidence = summarize_answer_evidence(
            matrices,
            hypotheses,
            selected.tolist(),
            high_conflict_threshold=float(phase["high_conflict_threshold"]),
        )
        top_answer_ids = []
        for selected_index in selected.tolist():
            items = hypotheses[int(selected_index)]
            if items:
                top_answer_ids.append(stable_identifier(str(items[0].get("normalized", "")), namespace="answer"))
        report: dict[str, Any] = {
            "question_id": question_id,
            "retrieved_count": len(passages),
            "selected_ranks": [int(value) for value in selected.tolist()],
            "selected_top_answer_ids": top_answer_ids,
            "evidence": evidence,
            "counterfactual": {"available": False},
        }

        swap = select_counterfactual_swap(matrices["conflict"], selected.tolist())
        if swap is not None and generator is not None:
            source, replacement, conflict_value = swap
            selected_passages = [passages[int(value)] for value in selected.tolist()]
            replacement_position = [int(value) for value in selected.tolist()].index(source)
            counterfactual_passages = list(selected_passages)
            counterfactual_passages[replacement_position] = passages[replacement]
            generation_started = time.perf_counter()
            baseline_prediction = generator.generate(question, selected_passages)
            counterfactual_prediction = generator.generate(question, counterfactual_passages)
            elapsed = (time.perf_counter() - generation_started) * 1000.0
            generation_times.append(elapsed)
            baseline_metrics = evaluate_answer(baseline_prediction, item.get("answers", []))
            counterfactual_metrics = evaluate_answer(counterfactual_prediction, item.get("answers", []))
            report["counterfactual"] = {
                "available": True,
                "source_rank": int(source),
                "replacement_rank": int(replacement),
                "conflict": float(conflict_value),
                "prediction_changed": baseline_prediction != counterfactual_prediction,
                "baseline_prediction_id": stable_identifier(baseline_prediction, namespace="prediction"),
                "counterfactual_prediction_id": stable_identifier(counterfactual_prediction, namespace="prediction"),
                "baseline": baseline_metrics,
                "counterfactual": counterfactual_metrics,
            }
        sample_reports.append(report)
        if (index + 1) % 10 == 0 or index + 1 == len(questions):
            print(f"  Progress: {index + 1}/{len(questions)}")

    selected_agreement = [sample["evidence"]["agreement"]["selected_pair_mean"] for sample in sample_reports]
    selected_conflict = [sample["evidence"]["conflict"]["selected_pair_mean"] for sample in sample_reports]
    selected_corroboration = [sample["evidence"]["corroboration"]["selected_pair_mean"] for sample in sample_reports]
    counterfactuals = [sample["counterfactual"] for sample in sample_reports if sample["counterfactual"]["available"]]
    metadata["timing_ms"] = {
        "total": (time.perf_counter() - started) * 1000.0,
        "mean_answer_scorer": _mean(scorer_times),
        "mean_selection": _mean(selection_times),
        "mean_counterfactual_generation_pair": _mean(generation_times),
    }
    metadata["summary"] = {
        "samples": len(sample_reports),
        "mean_selected_agreement": _mean(selected_agreement),
        "mean_selected_conflict": _mean(selected_conflict),
        "mean_selected_corroboration": _mean(selected_corroboration),
        "counterfactual_available": len(counterfactuals),
        "counterfactual_prediction_changed": sum(item["prediction_changed"] for item in counterfactuals),
        "counterfactual_mean_f1_delta": _mean([
            item["counterfactual"]["f1"] - item["baseline"]["f1"] for item in counterfactuals
        ]),
    }
    (run_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "diagnostics.json").write_text(
        json.dumps({"schema_version": 1, "samples": sample_reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed Phase 7A diagnostics: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (DiagnosticError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
