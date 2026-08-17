#!/usr/bin/env python3
"""Run the collaborator-only Phase 7D answer-evidence independence diagnostic."""

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

from applications.rag.answer_evidence import build_answer_evidence_matrices, stable_identifier
from applications.rag.answer_independence import (
    NUISANCE_FEATURES,
    TARGET_FEATURES,
    diagnostic_spec_hash,
    run_diagnostic_pipeline,
    selected_pair_mean,
)
from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation.metrics import evaluate_answer
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from applications.rag.signals_rag import passage_redundancy
from scripts.collab.five_ideas.phase7b_metrics import validate_compact_rows
from scripts.collab.five_ideas.phase7d_metrics import summarize_independence_rows


class DiagnosticError(RuntimeError):
    """Raised when Phase 7D cannot run under its frozen contract."""


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
            raise DiagnosticError(f"generator at index {index} must be a mapping")
        name = str(item.get("name", "")).strip()
        model_path = os.path.expandvars(str(item.get("model_path", "")).strip())
        if not name or not model_path or "$" in model_path:
            raise DiagnosticError(f"generator at index {index} needs a name and resolved model_path")
        if name in names:
            raise DiagnosticError(f"duplicate generator name: {name}")
        names.add(name)
        generators.append({"name": name, "model_path": model_path})
    if [item["name"] for item in generators] != ["reference"]:
        raise DiagnosticError("Phase 7D requires exactly one generator named reference")
    return generators


def _load_config(
    path: Path,
    generator_overrides: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("schema_version") != 1:
        raise DiagnosticError("configuration must contain phase.schema_version: 1")
    required = {
        "name", "diagnostic_only", "dataset", "split", "sample_offset", "max_samples",
        "seeds", "corpus_mode", "wiki_dpr_config", "wiki_dpr_nprobe", "top_k_retrieval",
        "K_values", "selection_variants", "answer_scorer_backend", "answer_top_m",
        "max_answer_tokens", "decisive_confidence_threshold", "decisive_margin_threshold",
        "diagnostic_plugins", "residualization", "bootstrap", "gate", "generators", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise DiagnosticError(f"configuration missing keys: {missing}")
    if phase["diagnostic_only"] is not True:
        raise DiagnosticError("Phase 7D must remain diagnostic_only")
    if phase["corpus_mode"] != "wiki_dpr":
        raise DiagnosticError("Phase 7D requires corpus_mode=wiki_dpr")
    if int(phase["sample_offset"]) < 150:
        raise DiagnosticError("sample_offset must be at least 150 to use a new held-out slice")
    if not 1 <= int(phase["max_samples"]) <= 100:
        raise DiagnosticError("max_samples must be between 1 and 100")

    seeds = [int(value) for value in phase["seeds"]]
    k_values = [int(value) for value in phase["K_values"]]
    if seeds != [42] or k_values != [5]:
        raise DiagnosticError("Phase 7D is frozen to seed 42 and K=5")
    if max(k_values) >= int(phase["top_k_retrieval"]):
        raise DiagnosticError("K must be smaller than top_k_retrieval")

    variants: list[dict[str, Any]] = []
    for index, item in enumerate(phase["selection_variants"]):
        if not isinstance(item, dict):
            raise DiagnosticError(f"selection variant at index {index} must be a mapping")
        name = str(item.get("name", "")).strip()
        method = str(item.get("method", "")).strip()
        score_signal = str(item.get("score_signal", "")).strip()
        enhancers = [str(value) for value in item.get("enhancers", [])]
        enhancer_configs = item.get("enhancer_configs", {})
        if method not in {"qore", "topk"} or score_signal not in {"answer", "retrieval"}:
            raise DiagnosticError(f"invalid selection variant at index {index}")
        if not isinstance(enhancer_configs, dict):
            raise DiagnosticError(f"invalid enhancer config at index {index}")
        if method == "qore" and enhancers != ["baseline"]:
            raise DiagnosticError("Phase 7D QORE controls may use only the baseline enhancer")
        if method == "topk" and (enhancers or enhancer_configs):
            raise DiagnosticError("Phase 7D Top-K control cannot use enhancers")
        variants.append({
            "name": name, "method": method, "score_signal": score_signal,
            "enhancers": enhancers, "enhancer_configs": enhancer_configs,
        })
    expected_variants = {
        "qore_answer": ("qore", "answer"),
        "topk_answer": ("topk", "answer"),
        "qore_dpr": ("qore", "retrieval"),
    }
    observed_variants = {item["name"]: (item["method"], item["score_signal"]) for item in variants}
    if observed_variants != expected_variants:
        raise DiagnosticError("selection variants must be fixed qore_answer, topk_answer, and qore_dpr controls")

    plugins = [str(value) for value in phase["diagnostic_plugins"]]
    if plugins != ["answer_evidence_observer", "independence_residual"]:
        raise DiagnosticError("Phase 7D diagnostic plugin order is frozen")
    residualization = phase["residualization"]
    if not isinstance(residualization, dict):
        raise DiagnosticError("residualization must be a mapping")
    ridge = float(residualization.get("ridge", -1.0))
    if not np.isfinite(ridge) or ridge < 0.0:
        raise DiagnosticError("residualization.ridge must be finite and non-negative")
    residual_variants: dict[str, list[str]] = {}
    for item in residualization.get("variants", []):
        if not isinstance(item, dict):
            raise DiagnosticError("every residualization variant must be a mapping")
        name = str(item.get("name", "")).strip()
        nuisances = [str(value) for value in item.get("nuisances", [])]
        if not name or name in residual_variants or not nuisances:
            raise DiagnosticError("residualization variant names must be unique and non-empty")
        residual_variants[name] = nuisances
    expected_residual_variants = {
        "confidence_only": ["answer_confidence_product"],
        "lexical_only": ["lexical_duplication"],
        "embedding_only": ["embedding_redundancy"],
        "all_nuisances": list(NUISANCE_FEATURES),
    }
    if residual_variants != expected_residual_variants:
        raise DiagnosticError("residualization variants do not match the frozen nuisance ablations")

    bootstrap = dict(phase["bootstrap"])
    if int(bootstrap.get("samples", 0)) < 100 or int(bootstrap.get("seed", -1)) < 0:
        raise DiagnosticError("bootstrap needs at least 100 samples and a non-negative seed")
    gate = dict(phase["gate"])
    if gate.get("residual_variant") != "all_nuisances":
        raise DiagnosticError("the gate must use the all_nuisances residual")
    if gate.get("primary_selector") != "qore_answer":
        raise DiagnosticError("the gate primary selector must be qore_answer")
    if gate.get("consistency_selectors") != ["topk_answer", "qore_dpr"]:
        raise DiagnosticError("gate consistency selectors must be topk_answer and qore_dpr")

    normalized = dict(phase)
    normalized.update({
        "seeds": seeds, "K_values": k_values, "selection_variants": variants,
        "diagnostic_plugins": plugins,
        "residualization": {"ridge": ridge, "variants": residual_variants},
        "bootstrap": {"samples": int(bootstrap["samples"]), "seed": int(bootstrap["seed"])},
        "gate": gate,
        "generators": _resolve_generators(phase["generators"], generator_overrides),
    })
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "configs/experiments/phase7d_independence_residual.yaml")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-offset", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--generator", action="append", type=_parse_generator_override)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--validate-only", action="store_true")
    validation.add_argument("--smoke-manifest", type=Path)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selection_features(diagnostics: dict[str, Any], selected: list[int]) -> dict[str, float]:
    features = {f"selected_{target}": selected_pair_mean(diagnostics["raw"][target], selected) for target in TARGET_FEATURES}
    output_names = {
        "answer_confidence_product": "selected_answer_confidence_product",
        "lexical_duplication": "selected_lexical_duplication",
        "embedding_redundancy": "selected_embedding_redundancy",
    }
    for nuisance, output_name in output_names.items():
        features[output_name] = selected_pair_mean(diagnostics["nuisances"][nuisance], selected)
    for variant, matrices in diagnostics["residuals"].items():
        for target in TARGET_FEATURES:
            features[f"selected_residual_{variant}_{target}"] = selected_pair_mean(matrices[target], selected)
    return features


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
        raise DiagnosticError("--max-samples must be between 1 and 100")
    if sample_offset < 150:
        raise DiagnosticError("--sample-offset must be at least 150")
    if args.validate_only:
        print(json.dumps({
            "status": "valid", "diagnostic_only": True, "max_samples": max_samples,
            "sample_offset": sample_offset,
            "selection_variants": [item["name"] for item in phase["selection_variants"]],
            "diagnostic_plugins": phase["diagnostic_plugins"],
            "residualization_variants": list(phase["residualization"]["variants"]),
            "generators": [item["name"] for item in phase["generators"]],
        }, sort_keys=True))
        return None
    if args.smoke_manifest is not None:
        smoke_path = args.smoke_manifest
        if not smoke_path.is_absolute():
            smoke_path = project_root / smoke_path
        smoke_path = smoke_path.resolve()
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        residual_variants = phase["residualization"]["variants"]
        smoke_manifest = {
            "schema_version": 1,
            "phase": phase["name"],
            "diagnostic_only": True,
            "selection_mutation": False,
            "smoke_only": True,
            "status": "smoke_validated",
            "config": {
                "path": str(config_path.resolve()),
                "sha256": _sha256_file(config_path.resolve()),
            },
            "dataset": {
                "name": phase["dataset"],
                "split": phase["split"],
                "sample_offset": sample_offset,
                "max_samples": max_samples,
                "corpus_mode": phase["corpus_mode"],
                "wiki_dpr_config": phase["wiki_dpr_config"],
            },
            "design": {
                "seeds": phase["seeds"],
                "K_values": phase["K_values"],
                "selection_variants": phase["selection_variants"],
                "diagnostic_plugins": phase["diagnostic_plugins"],
                "plugin_tree_hash": diagnostic_spec_hash(
                    phase["diagnostic_plugins"],
                    residual_variants,
                    phase["residualization"]["ridge"],
                ),
                "residualization_variants": residual_variants,
                "gold_labels_used_for_residual_fit": False,
                "diagnostic_outputs_used_for_selection": False,
            },
            "generators": phase["generators"],
            "python": {"executable": sys.executable, "version": sys.version},
            "git": {
                "commit": _git(project_root, "rev-parse", "HEAD"),
                "status": _git(project_root, "status", "--short", "--branch"),
            },
        }
        _write_json(smoke_path, smoke_manifest)
        print(f"Wrote Phase 7D smoke manifest: {smoke_path}")
        return smoke_path

    if not args.allow_dirty:
        tracked_dirty = (
            subprocess.run(["git", "diff", "--quiet"], cwd=project_root).returncode != 0
            or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=project_root).returncode != 0
        )
        if tracked_dirty:
            raise DiagnosticError("tracked worktree changes detected; use --allow-dirty only for debugging")

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
    residual_variants = phase["residualization"]["variants"]
    manifest: dict[str, Any] = {
        "schema_version": 1, "phase": phase["name"], "diagnostic_only": True,
        "selection_mutation": False, "status": "running",
        "config": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path.resolve())},
        "dataset": {
            "name": phase["dataset"], "split": phase["split"], "sample_offset": sample_offset,
            "max_samples": max_samples, "corpus_mode": phase["corpus_mode"],
            "wiki_dpr_config": phase["wiki_dpr_config"], "nprobe": int(phase["wiki_dpr_nprobe"]),
        },
        "design": {
            "seeds": phase["seeds"], "K_values": phase["K_values"],
            "selection_variants": phase["selection_variants"],
            "diagnostic_plugins": phase["diagnostic_plugins"],
            "plugin_tree_hash": diagnostic_spec_hash(phase["diagnostic_plugins"], residual_variants, phase["residualization"]["ridge"]),
            "residualization_variants": residual_variants,
            "gold_labels_used_for_residual_fit": False,
            "diagnostic_outputs_used_for_selection": False,
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
        raise DiagnosticError(f"held-out slice returned {len(questions)} questions, expected {max_samples}")

    encoder = make_encoder("dpr", device=args.device)
    corpus_manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["wiki_dpr_config"], "nprobe": int(phase["wiki_dpr_nprobe"])})
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend=phase["answer_scorer_backend"], device=args.device)

    cached: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for question_index, item in enumerate(questions):
        question = item["question"]
        query_embedding = encoder.encode_queries([question])[0]
        _, embeddings, passages, retrieval_scores = corpus_manager.retrieve_with_embeddings(query_embedding, int(phase["top_k_retrieval"]))
        answer_scores, hypotheses = scorer.score_passages_with_hypotheses(question, passages, top_m=int(phase["answer_top_m"]), max_answer_tokens=int(phase["max_answer_tokens"]))
        evidence = build_answer_evidence_matrices(passages, hypotheses, passage_confidence=answer_scores, decisive_confidence_threshold=float(phase["decisive_confidence_threshold"]), decisive_margin_threshold=float(phase["decisive_margin_threshold"]))
        diagnostics = run_diagnostic_pipeline(evidence, answer_scores, passage_redundancy(embeddings, method="cosine"), plugin_ids=phase["diagnostic_plugins"], residualization_variants=residual_variants, ridge=float(phase["residualization"]["ridge"]))
        question_id = stable_identifier(str(item["id"]), namespace="question")
        for residual_variant, target_fits in diagnostics["fits"].items():
            for target, fit in target_fits.items():
                fit_rows.append({"question_id": question_id, "residualization_variant": residual_variant, "target": target, **fit})

        selections: list[dict[str, Any]] = []
        for seed in phase["seeds"]:
            for k_value in phase["K_values"]:
                for variant in phase["selection_variants"]:
                    scores = answer_scores if variant["score_signal"] == "answer" else retrieval_scores
                    selected = np.asarray(select_passages(query_embedding, embeddings, K=k_value, method=variant["method"], relevance_scores=scores, enhancers=variant["enhancers"], enhancer_configs=variant["enhancer_configs"], seed=seed), dtype=np.int64)
                    selected_list = [int(value) for value in selected.tolist()]
                    selections.append({"seed": seed, "K": k_value, "selection_variant": variant["name"], "selected_ranks": selected_list, "features": _selection_features(diagnostics, selected_list)})
        cached.append({"question_id": question_id, "question": question, "gold_answers": item.get("answers", []), "passages": passages, "selections": selections})
        if (question_index + 1) % 10 == 0 or question_index + 1 == len(questions):
            print(f"  Residual cache: {question_index + 1}/{len(questions)}")

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    for generator_spec in phase["generators"]:
        name = generator_spec["name"]
        manifest["generator_status"][name] = {"status": "running", "unique_generations": 0}
        _write_json(run_dir / "manifest.json", manifest)
        generator = Generator(generator_spec["model_path"], device=args.device, max_new_tokens=int(phase.get("max_new_tokens", 32)), use_chat_template=True)
        unique_generations = 0
        for question_index, item in enumerate(cached):
            prediction_cache: dict[tuple[int, ...], tuple[str, dict[str, float]]] = {}
            for selection in item["selections"]:
                selected_key = tuple(selection["selected_ranks"])
                cache_hit = selected_key in prediction_cache
                if not cache_hit:
                    prediction = generator.generate(item["question"], [item["passages"][index] for index in selected_key])
                    prediction_cache[selected_key] = (stable_identifier(prediction, namespace=f"prediction:{name}"), evaluate_answer(prediction, item["gold_answers"]))
                    unique_generations += 1
                prediction_id, quality = prediction_cache[selected_key]
                rows.append({"question_id": item["question_id"], "generator": name, "seed": selection["seed"], "K": selection["K"], "selection_variant": selection["selection_variant"], "selected_count": len(selected_key), "selected_ranks": list(selected_key), "prediction_id": prediction_id, "generation_cache_hit": cache_hit, "em": float(quality["em"]), "f1": float(quality["f1"]), **selection["features"]})
            if (question_index + 1) % 10 == 0 or question_index + 1 == len(cached):
                print(f"  {name}: {question_index + 1}/{len(cached)}")
        manifest["generator_status"][name] = {"status": "completed", "unique_generations": unique_generations}
        validate_compact_rows(rows)
        _write_json(run_dir / "diagnostics.json", {"schema_version": 1, "diagnostic_only": True, "rows": rows, "residual_fits": fit_rows})
        summary = summarize_independence_rows(rows, residualization_variants=list(residual_variants), gate=phase["gate"], bootstrap_samples=int(phase["bootstrap"]["samples"]), bootstrap_seed=int(phase["bootstrap"]["seed"]))
        _write_json(run_dir / "analysis/summary.json", summary)
        _write_json(run_dir / "manifest.json", manifest)
        _release_generator(generator)

    if summary is None:
        raise DiagnosticError("no generator summary was produced")
    manifest["status"] = "completed"
    manifest["timing_ms"] = {"total": (time.perf_counter() - started) * 1000.0}
    manifest["summary"] = {"questions": len(cached), "rows": len(rows), "residual_fit_rows": len(fit_rows), "unique_generations": sum(item["unique_generations"] for item in manifest["generator_status"].values()), "gate_status": summary["gate"]["status"], "gate_decision": summary["gate"]["decision"]}
    _write_json(run_dir / "manifest.json", manifest)
    print(f"Completed Phase 7D independence residual diagnostic: {run_dir}")
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
