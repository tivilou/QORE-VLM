#!/usr/bin/env python3
"""Run the collaborator-only Phase 7B answer-identity calibration gate."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
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
    stable_identifier,
    summarize_answer_evidence,
)
from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation.metrics import evaluate_answer
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from scripts.collab.five_ideas.phase7b_metrics import (
    summarize_calibration_rows,
    validate_compact_rows,
)


class CalibrationError(RuntimeError):
    """Raised when the calibration gate cannot run safely."""


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


def _parse_generator_override(value: str) -> dict[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("generator override must be NAME=MODEL_PATH")
    return {"name": name.strip(), "model_path": path.strip()}


def _resolve_generators(
    specifications: list[dict[str, Any]],
    overrides: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    source = overrides if overrides else specifications
    generators: list[dict[str, str]] = []
    names: set[str] = set()
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise CalibrationError(f"generator at index {index} must be a mapping")
        name = str(item.get("name", "")).strip()
        model_path = os.path.expandvars(str(item.get("model_path", "")).strip())
        if not name or not model_path or "$" in model_path:
            raise CalibrationError(
                f"generator at index {index} needs a name and resolved model_path; "
                "set the referenced environment variable or use --generator NAME=PATH"
            )
        if name in names:
            raise CalibrationError(f"duplicate generator name: {name}")
        names.add(name)
        generators.append({"name": name, "model_path": model_path})
    if len(generators) < 2:
        raise CalibrationError("Phase 7B requires a reference and at least one transfer generator")
    return generators


def _load_config(
    path: Path,
    generator_overrides: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("schema_version") != 1:
        raise CalibrationError("configuration must contain phase.schema_version: 1")
    required = {
        "name", "dataset", "split", "sample_offset", "max_samples", "seeds",
        "corpus_mode", "wiki_dpr_config", "wiki_dpr_nprobe", "top_k_retrieval",
        "K_values", "selection_variants", "answer_scorer_backend", "answer_top_m",
        "max_answer_tokens", "high_conflict_threshold", "generators", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise CalibrationError(f"configuration missing keys: {missing}")
    if phase["corpus_mode"] != "wiki_dpr":
        raise CalibrationError("Phase 7B requires corpus_mode=wiki_dpr")
    if int(phase["sample_offset"]) < 50:
        raise CalibrationError("sample_offset must be at least 50 to remain held out from Phase 7A")
    if not 1 <= int(phase["max_samples"]) <= 100:
        raise CalibrationError("max_samples must be between 1 and 100")

    seeds = [int(value) for value in phase["seeds"]]
    k_values = [int(value) for value in phase["K_values"]]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)) or any(value < 0 for value in seeds):
        raise CalibrationError("seeds must contain at least two unique non-negative values")
    if len(k_values) < 2 or len(k_values) != len(set(k_values)) or any(value < 1 for value in k_values):
        raise CalibrationError("K_values must contain at least two unique positive values")
    if max(k_values) >= int(phase["top_k_retrieval"]):
        raise CalibrationError("every K must be smaller than top_k_retrieval")

    variants: list[dict[str, str]] = []
    variant_names: set[str] = set()
    for index, item in enumerate(phase["selection_variants"]):
        if not isinstance(item, dict):
            raise CalibrationError(f"selection variant at index {index} must be a mapping")
        name = str(item.get("name", "")).strip()
        method = str(item.get("method", "")).strip()
        score_signal = str(item.get("score_signal", "")).strip()
        if not name or method not in {"qore", "topk"} or score_signal not in {"answer", "retrieval"}:
            raise CalibrationError(f"invalid selection variant at index {index}")
        if name in variant_names:
            raise CalibrationError(f"duplicate selection variant name: {name}")
        variant_names.add(name)
        variants.append({"name": name, "method": method, "score_signal": score_signal})
    required_variants = {"qore_answer", "topk_answer", "qore_dpr"}
    if not required_variants.issubset(variant_names):
        raise CalibrationError(f"selection variants must include {sorted(required_variants)}")

    normalized = dict(phase)
    normalized["seeds"] = seeds
    normalized["K_values"] = k_values
    normalized["selection_variants"] = variants
    normalized["generators"] = _resolve_generators(phase["generators"], generator_overrides)
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/experiments/phase7b_answer_identity_calibration.yaml",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-offset", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--generator", action="append", type=_parse_generator_override)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_row(evidence: dict[str, Any]) -> dict[str, float]:
    return {
        "selected_agreement": float(evidence["agreement"]["selected_pair_mean"]),
        "selected_conflict": float(evidence["conflict"]["selected_pair_mean"]),
        "selected_corroboration": float(evidence["corroboration"]["selected_pair_mean"]),
        "selected_duplication": float(evidence["duplication"]["selected_pair_mean"]),
    }


def _release_generator(generator: Generator) -> None:
    del generator
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run(args: argparse.Namespace) -> Path | None:
    project_root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    phase = _load_config(config_path.resolve(), args.generator)
    max_samples = int(args.max_samples or phase["max_samples"])
    sample_offset = int(phase["sample_offset"] if args.sample_offset is None else args.sample_offset)
    if not 1 <= max_samples <= 100:
        raise CalibrationError("--max-samples must be between 1 and 100")
    if sample_offset < 50:
        raise CalibrationError("--sample-offset must be at least 50")
    if args.validate_only:
        print(json.dumps({
            "status": "valid",
            "max_samples": max_samples,
            "sample_offset": sample_offset,
            "seeds": phase["seeds"],
            "K_values": phase["K_values"],
            "selection_variants": [item["name"] for item in phase["selection_variants"]],
            "generators": [item["name"] for item in phase["generators"]],
        }, sort_keys=True))
        return None

    if not args.allow_dirty:
        tracked_dirty = (
            subprocess.run(["git", "diff", "--quiet"], cwd=project_root).returncode != 0
            or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=project_root).returncode != 0
        )
        if tracked_dirty:
            raise CalibrationError("tracked worktree changes detected; use --allow-dirty only for debugging")

    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else project_root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root.resolve() / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root.resolve() / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "analysis").mkdir()

    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase["name"],
        "diagnostic_only": True,
        "status": "running",
        "config": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path.resolve())},
        "dataset": {
            "name": phase["dataset"], "split": phase["split"],
            "sample_offset": sample_offset, "max_samples": max_samples,
            "corpus_mode": phase["corpus_mode"],
            "wiki_dpr_config": phase["wiki_dpr_config"],
            "nprobe": int(phase["wiki_dpr_nprobe"]),
        },
        "design": {
            "seeds": phase["seeds"], "K_values": phase["K_values"],
            "selection_variants": phase["selection_variants"],
            "generators": phase["generators"],
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "git": {"commit": _git(project_root, "rev-parse", "HEAD"), "status": _git(project_root, "status", "--short", "--branch")},
        "generator_status": {},
    }
    _write_json(run_dir / "manifest.json", manifest)

    requested = sample_offset + max_samples
    questions = load_dataset_for_rag(phase["dataset"], phase["split"], requested)
    questions = questions[sample_offset:requested]
    if len(questions) != max_samples:
        raise CalibrationError(f"held-out slice returned {len(questions)} questions, expected {max_samples}")

    encoder = make_encoder("dpr", device=args.device)
    corpus_manager = make_corpus_manager(
        "wiki_dpr",
        {"wiki_dpr_config": phase["wiki_dpr_config"], "nprobe": int(phase["wiki_dpr_nprobe"])},
    )
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend=phase["answer_scorer_backend"], device=args.device)

    cached: list[dict[str, Any]] = []
    for question_index, item in enumerate(questions):
        question = item["question"]
        query_embedding = encoder.encode_queries([question])[0]
        _, embeddings, passages, retrieval_scores = corpus_manager.retrieve_with_embeddings(
            query_embedding, int(phase["top_k_retrieval"])
        )
        answer_scores, hypotheses = scorer.score_passages_with_hypotheses(
            question, passages, top_m=int(phase["answer_top_m"]),
            max_answer_tokens=int(phase["max_answer_tokens"]),
        )
        matrices = build_answer_evidence_matrices(
            passages, hypotheses, passage_confidence=answer_scores
        )
        selections: list[dict[str, Any]] = []
        for seed in phase["seeds"]:
            for k_value in phase["K_values"]:
                for variant in phase["selection_variants"]:
                    scores = answer_scores if variant["score_signal"] == "answer" else retrieval_scores
                    selected = np.asarray(select_passages(
                        query_embedding, embeddings, K=k_value, method=variant["method"],
                        relevance_scores=scores, seed=seed,
                    ), dtype=np.int64)
                    evidence = summarize_answer_evidence(
                        matrices, hypotheses, selected.tolist(),
                        high_conflict_threshold=float(phase["high_conflict_threshold"]),
                    )
                    selections.append({
                        "seed": seed, "K": k_value,
                        "selection_variant": variant["name"],
                        "selected_ranks": [int(value) for value in selected.tolist()],
                        "features": _feature_row(evidence),
                    })
        cached.append({
            "question_id": stable_identifier(str(item["id"]), namespace="question"),
            "question": question,
            "gold_answers": item.get("answers", []),
            "passages": passages,
            "selections": selections,
        })
        if (question_index + 1) % 10 == 0 or question_index + 1 == len(questions):
            print(f"  Evidence cache: {question_index + 1}/{len(questions)}")

    rows: list[dict[str, Any]] = []
    for generator_spec in phase["generators"]:
        name = generator_spec["name"]
        manifest["generator_status"][name] = {"status": "running", "unique_generations": 0}
        _write_json(run_dir / "manifest.json", manifest)
        generator = Generator(
            generator_spec["model_path"], device=args.device,
            max_new_tokens=int(phase.get("max_new_tokens", 32)), use_chat_template=True,
        )
        unique_generations = 0
        for question_index, item in enumerate(cached):
            prediction_cache: dict[tuple[int, ...], tuple[str, dict[str, float]]] = {}
            for selection in item["selections"]:
                selected_key = tuple(selection["selected_ranks"])
                cache_hit = selected_key in prediction_cache
                if not cache_hit:
                    prediction = generator.generate(
                        item["question"], [item["passages"][index] for index in selected_key]
                    )
                    prediction_cache[selected_key] = (
                        stable_identifier(prediction, namespace=f"prediction:{name}"),
                        evaluate_answer(prediction, item["gold_answers"]),
                    )
                    unique_generations += 1
                prediction_id, quality = prediction_cache[selected_key]
                rows.append({
                    "question_id": item["question_id"],
                    "generator": name,
                    "seed": selection["seed"],
                    "K": selection["K"],
                    "selection_variant": selection["selection_variant"],
                    "selected_count": len(selected_key),
                    "selected_ranks": list(selected_key),
                    "prediction_id": prediction_id,
                    "generation_cache_hit": cache_hit,
                    "em": float(quality["em"]),
                    "f1": float(quality["f1"]),
                    **selection["features"],
                })
            if (question_index + 1) % 10 == 0 or question_index + 1 == len(cached):
                print(f"  {name}: {question_index + 1}/{len(cached)}")
        manifest["generator_status"][name] = {
            "status": "completed", "unique_generations": unique_generations,
        }
        validate_compact_rows(rows)
        _write_json(run_dir / "calibration.json", {"schema_version": 1, "rows": rows})
        _write_json(run_dir / "analysis/summary.json", summarize_calibration_rows(rows))
        _write_json(run_dir / "manifest.json", manifest)
        _release_generator(generator)
        generator = None

    manifest["status"] = "completed"
    manifest["timing_ms"] = {"total": (time.perf_counter() - started) * 1000.0}
    manifest["summary"] = {
        "questions": len(cached), "rows": len(rows),
        "unique_generations": sum(item["unique_generations"] for item in manifest["generator_status"].values()),
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"Completed Phase 7B calibration: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (CalibrationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
