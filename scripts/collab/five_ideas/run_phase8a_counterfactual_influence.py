#!/usr/bin/env python3
"""Run the collaborator-only Phase 8A counterfactual influence diagnostic.

This runner is an observation-only adapter around the frozen QORE plus Answer
Scorer baseline. Diagnostic estimates never flow back into ``select_passages``
and only compact identifiers, masks, numeric covariates, metrics, and timing
are written to Exchange.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
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

try:
    from applications.rag.counterfactual_influence import (
        COVARIATE_NAMES,
        build_context_probes,
        stable_id,
        validate_compact_payload,
    )
except ImportError:  # pragma: no cover - direct local module tests
    from counterfactual_influence import (
        COVARIATE_NAMES,
        build_context_probes,
        stable_id,
        validate_compact_payload,
    )

try:
    from scripts.collab.five_ideas.phase8a_metrics import (
        FORBIDDEN_FIELDS,
        summarize_counterfactual_influence,
    )
except ImportError:  # pragma: no cover - direct local module tests
    from phase8a_metrics import FORBIDDEN_FIELDS, summarize_counterfactual_influence


class DiagnosticError(RuntimeError):
    """Raised when the frozen Phase 8A contract cannot be satisfied."""


EXPECTED_PLUGINS = [
    "full_context_probe",
    "balanced_subset_probe",
    "doubly_robust_influence",
]


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    # The temporary implementation lives outside the server repository.
    return here.parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_generator_override(value: str) -> dict[str, str]:
    name, separator, model_path = value.partition("=")
    if not separator or not name.strip() or not model_path.strip():
        raise argparse.ArgumentTypeError("generator override must be NAME=MODEL_PATH")
    return {"name": name.strip(), "model_path": model_path.strip()}


def _normalize_generators(
    specifications: Sequence[Mapping[str, Any]],
    overrides: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    source: Sequence[Mapping[str, Any]] = overrides if overrides else specifications
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise DiagnosticError(f"generator at index {index} must be a mapping")
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            raise DiagnosticError(f"generator names must be unique and non-empty: {name!r}")
        names.add(name)
        candidates = item.get("model_path_candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise DiagnosticError(f"generator {name} needs model_path_candidates")
        result.append({
            "name": name,
            "model_id": str(item.get("model_id", "")).strip(),
            "revision": str(item.get("revision", "")).strip(),
            "model_path_candidates": [str(value) for value in candidates],
            "resolver_order": [str(value) for value in item.get("resolver_order", [])],
            "record_resolved_identity": bool(item.get("record_resolved_identity", False)),
        })
    if [item["name"] for item in result] != ["reference"]:
        raise DiagnosticError("Phase 8A requires exactly one generator named reference")
    return result


def _load_config(path: Path, generator_overrides: Sequence[Mapping[str, str]] | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or int(phase.get("schema_version", -1)) != 1:
        raise DiagnosticError("configuration must contain phase.schema_version: 1")
    required = {
        "name", "schema_version", "diagnostic_only", "selection_mutation", "dataset",
        "split", "sample_offset", "max_samples", "corpus_mode", "wiki_dpr_config",
        "wiki_dpr_nprobe", "top_k_retrieval", "selection", "answer_scorer",
        "diagnostic_plugins", "randomization", "estimator", "gate", "max_new_tokens",
        "generators", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise DiagnosticError(f"configuration missing keys: {missing}")
    if phase["name"] != "phase8a_counterfactual_influence":
        raise DiagnosticError("unexpected Phase 8A configuration name")
    if phase["diagnostic_only"] is not True or phase["selection_mutation"] is not False:
        raise DiagnosticError("Phase 8A must remain diagnostic_only with selection_mutation=false")
    if (phase["dataset"], phase["split"]) != ("nq_open", "validation"):
        raise DiagnosticError("Phase 8A is frozen to nq_open validation")
    if int(phase["sample_offset"]) != 200 or int(phase["max_samples"]) != 50:
        raise DiagnosticError("Phase 8A is frozen to validation questions 200-249")
    if phase["corpus_mode"] != "wiki_dpr" or phase["wiki_dpr_config"] != "psgs_w100.nq.compressed":
        raise DiagnosticError("Phase 8A requires the compressed Wiki-DPR corpus")
    if int(phase["wiki_dpr_nprobe"]) != 64 or int(phase["top_k_retrieval"]) != 50:
        raise DiagnosticError("Wiki-DPR retrieval settings are not frozen")

    selection = phase["selection"]
    if not isinstance(selection, dict):
        raise DiagnosticError("selection must be a mapping")
    if (
        selection.get("name"), selection.get("method"), selection.get("score_signal")
    ) != ("qore_answer", "qore", "answer"):
        raise DiagnosticError("selection must be the qore_answer baseline")
    if int(selection.get("K", -1)) != 5 or int(selection.get("seed", -1)) != 42:
        raise DiagnosticError("selection is frozen to K=5 and seed=42")
    if list(selection.get("enhancers", [])) != ["baseline"]:
        raise DiagnosticError("only the baseline enhancer is allowed")
    baseline_cfg = selection.get("enhancer_configs", {}).get("baseline", {})
    if float(baseline_cfg.get("gamma", -1)) != 1.0:
        raise DiagnosticError("baseline gamma must be 1.0")
    scorer = phase["answer_scorer"]
    if not isinstance(scorer, dict) or str(scorer.get("backend")) != "dpr":
        raise DiagnosticError("Phase 8A requires the DPR Answer Scorer")

    plugins = [str(value) for value in phase["diagnostic_plugins"]]
    if plugins != EXPECTED_PLUGINS:
        raise DiagnosticError(f"diagnostic plugin order is frozen: {EXPECTED_PLUGINS}")
    randomization = phase["randomization"]
    if not isinstance(randomization, dict):
        raise DiagnosticError("randomization must be a mapping")
    frozen_randomization = {
        "seed": 8101, "selected_passage_count": 5, "complement_pairs": 8,
        "effect_fit_probes": 16, "include_full_anchor": True,
        "include_empty_anchor": True, "anchors_in_effect_fit": False,
        "treatment_propensity": 0.5, "randomize_included_order": True,
        "max_generations_per_question": 18,
    }
    for key, expected in frozen_randomization.items():
        observed = randomization.get(key)
        if isinstance(expected, float):
            if float(observed) != expected:
                raise DiagnosticError(f"randomization.{key} is not frozen")
        elif observed != expected:
            raise DiagnosticError(f"randomization.{key} is not frozen")

    estimator = phase["estimator"]
    if not isinstance(estimator, dict):
        raise DiagnosticError("estimator must be a mapping")
    if (
        estimator.get("name"), estimator.get("outcome"), estimator.get("grouped_by"),
        int(estimator.get("folds", -1)), int(estimator.get("fold_seed", -1)),
        estimator.get("nuisance_model"), float(estimator.get("ridge", -1)),
    ) != ("cross_fitted_aipw", "f1", "question_id", 5, 8102, "ridge", 1e-6):
        raise DiagnosticError("estimator settings are not frozen")
    if list(estimator.get("secondary_outcomes", [])) != ["em"]:
        raise DiagnosticError("EM must remain the only secondary outcome")
    if list(estimator.get("covariates", [])) != list(COVARIATE_NAMES):
        raise DiagnosticError("covariate list does not match the frozen estimator")
    bootstrap = estimator.get("bootstrap", {})
    placebo = estimator.get("placebo", {})
    if (
        int(bootstrap.get("samples", -1)), int(bootstrap.get("seed", -1)), bootstrap.get("cluster")
    ) != (5000, 8103, "question_id"):
        raise DiagnosticError("bootstrap settings are not frozen")
    if placebo.get("mode") != "complement_preserving_within_question" or int(placebo.get("repetitions", -1)) != 200:
        raise DiagnosticError("placebo settings are not frozen")
    gate = phase["gate"]
    required_gate = {
        "primary_endpoint", "minimum_split_half_spearman", "minimum_split_half_ci95_low",
        "minimum_aipw_dim_spearman", "minimum_incremental_cross_fitted_r2",
        "minimum_questions_with_effect_range_0_10_fraction", "maximum_placebo_p_value",
        "maximum_generations_per_question", "pass_requires_all", "pass_action", "fail_action",
    }
    if not isinstance(gate, dict) or required_gate - set(gate):
        raise DiagnosticError("gate is incomplete")
    if gate["primary_endpoint"] != "split_half_spearman_influence" or gate["pass_requires_all"] is not True:
        raise DiagnosticError("Phase 8A gate must require all pre-registered checks")
    if int(gate["maximum_generations_per_question"]) != 18:
        raise DiagnosticError("generation cap is not frozen at 18")

    # CLI overrides replace only the resolved path. Keep the frozen model ID,
    # revision, and candidate metadata from the committed configuration.
    generators = _normalize_generators(phase["generators"])
    generator = generators[0]
    if generator["model_id"] != "NousResearch/Meta-Llama-3-8B-Instruct":
        raise DiagnosticError("reference model ID is not pinned")
    if generator["revision"] != "53346005fb0ef11d3b6a83b12c895cca40156b6c":
        raise DiagnosticError("reference model revision is not pinned")
    if generator["resolver_order"] != ["cli_override", "existing_candidate"] or not generator["record_resolved_identity"]:
        raise DiagnosticError("generator resolver contract is not frozen")
    outputs = phase["outputs"]
    if not isinstance(outputs, dict) or outputs.get("compact_only") is not True:
        raise DiagnosticError("Phase 8A outputs must be compact_only")
    if list(outputs.get("forbidden_fields", [])) != list(FORBIDDEN_FIELDS):
        raise DiagnosticError("forbidden output fields do not match the privacy contract")

    normalized = dict(phase)
    normalized.update({
        "sample_offset": int(phase["sample_offset"]), "max_samples": int(phase["max_samples"]),
        "wiki_dpr_nprobe": int(phase["wiki_dpr_nprobe"]), "top_k_retrieval": int(phase["top_k_retrieval"]),
        "selection": dict(selection), "answer_scorer": dict(scorer),
        "diagnostic_plugins": plugins, "randomization": dict(randomization),
        "estimator": dict(estimator), "gate": dict(gate), "generators": generators,
        "outputs": dict(outputs),
    })
    return normalized


def _resolve_generator(
    specification: Mapping[str, Any],
    overrides: Sequence[Mapping[str, str]] | None = None,
    *,
    require_resolved: bool,
) -> dict[str, Any]:
    override = None
    if overrides:
        if len(overrides) != 1 or str(overrides[0].get("name")) != "reference":
            raise DiagnosticError("--generator accepts exactly reference=MODEL_PATH")
        override = str(overrides[0]["model_path"])
    candidates = [override] if override is not None else list(specification["model_path_candidates"])
    attempted: list[str] = []
    for raw_path in candidates:
        model_path = Path(os.path.expanduser(os.path.expandvars(str(raw_path))))
        attempted.append(str(model_path))
        config_path = model_path / "config.json"
        if model_path.is_dir() and config_path.is_file():
            return {
                "name": specification["name"], "model_id": specification["model_id"],
                "revision": specification["revision"], "model_path": str(model_path.resolve()),
                "resolution": "resolved", "source": "cli_override" if override is not None else "existing_candidate",
                "config_path": str(config_path.resolve()), "config_sha256": _sha256_file(config_path),
            }
    status = {
        "name": specification["name"], "model_id": specification["model_id"],
        "revision": specification["revision"], "model_path": None,
        "resolution": "unresolved", "source": "cli_override" if override is not None else "existing_candidate",
        "attempted_paths": attempted, "config_path": None, "config_sha256": None,
    }
    if require_resolved:
        raise DiagnosticError(f"reference generator not resolved; attempted: {attempted}")
    return status


def _plugin_tree_hash(project_root: Path) -> str:
    paths = [
        project_root / "applications/rag/counterfactual_influence.py",
        project_root / "scripts/collab/five_ideas/phase8a_metrics.py",
        project_root / "scripts/collab/five_ideas/run_phase8a_counterfactual_influence.py",
    ]
    entries = []
    for path in paths:
        try:
            relative = str(path.relative_to(project_root))
        except ValueError:
            relative = str(path)
        entries.append({"path": relative, "sha256": _sha256_file(path) if path.is_file() else None})
    payload = json.dumps({"plugins": EXPECTED_PLUGINS, "files": entries}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase8a_counterfactual_influence.yaml")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-offset", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--generator", action="append", type=_parse_generator_override)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--validate-only", action="store_true")
    checks.add_argument("--smoke-manifest", type=Path)
    return parser.parse_args(argv)


def _base_manifest(
    project_root: Path, config_path: Path, phase: Mapping[str, Any],
    generator: Mapping[str, Any], sample_offset: int, max_samples: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase["name"],
        "diagnostic_only": True,
        "selection_mutation": False,
        "config": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path)},
        "dataset": {
            "name": phase["dataset"], "split": phase["split"], "sample_offset": sample_offset,
            "max_samples": max_samples, "slice_end_exclusive": sample_offset + max_samples,
            "corpus_mode": phase["corpus_mode"], "wiki_dpr_config": phase["wiki_dpr_config"],
            "nprobe": phase["wiki_dpr_nprobe"], "top_k_retrieval": phase["top_k_retrieval"],
        },
        "selection": phase["selection"],
        "design": {
            "diagnostic_plugins": phase["diagnostic_plugins"],
            "plugin_tree_hash": _plugin_tree_hash(project_root),
            "probe_seed": phase["randomization"]["seed"],
            "fold_seed": phase["estimator"]["fold_seed"],
            "bootstrap_seed": phase["estimator"]["bootstrap"]["seed"],
            "selected_passage_count": phase["randomization"]["selected_passage_count"],
            "complement_pairs": phase["randomization"]["complement_pairs"],
            "effect_fit_probes": phase["randomization"]["effect_fit_probes"],
            "generations_per_question": phase["randomization"]["max_generations_per_question"],
            "anchors_in_effect_fit": False,
            "diagnostic_outputs_used_for_selection": False,
            "gold_labels_used_for_assignment_or_covariates": False,
            "covariate_conventions": {
                "retrieval_rank": "one_based_rank_in_retrieved_top_k",
                "original_selection_position": "one_based_position_in_baseline_K5_order",
                "passage_token_count": "whitespace_token_count",
            },
        },
        "generator": dict(generator),
        "python": {"executable": sys.executable, "version": sys.version},
        "git": {"commit": _git(project_root, "rev-parse", "HEAD"),
                "status": _git(project_root, "status", "--short", "--branch")},
    }


def _pair_redundancy(embedding_matrix: np.ndarray, positions: Sequence[int]) -> float:
    if len(positions) < 2:
        return 0.0
    block = np.asarray(embedding_matrix)[np.ix_(list(positions), list(positions))]
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    normed = block / np.where(norms < 1e-12, 1.0, norms)
    similarities = normed @ normed.T
    upper = similarities[np.triu_indices(len(positions), k=1)]
    return float(np.mean(upper)) if len(upper) else 0.0


def _covariates(
    position: int, probe_mask: Sequence[int], selected_ranks: Sequence[int],
    answer_scores: np.ndarray, passages: Sequence[str], embeddings: np.ndarray,
) -> dict[str, float]:
    included = [index for index, value in enumerate(probe_mask) if int(value)]
    others = [index for index in included if index != position]
    passage_rank = int(selected_ranks[position])
    token_counts = [len(str(passages[int(rank)]).split()) for rank in selected_ranks]
    return {
        "answer_score": float(answer_scores[passage_rank]),
        "retrieval_rank": float(passage_rank + 1),
        "original_selection_position": float(position + 1),
        "passage_token_count": float(token_counts[position]),
        "other_context_size": float(len(others)),
        "other_context_answer_score_sum": float(sum(float(answer_scores[int(selected_ranks[index])]) for index in others)),
        "other_context_token_count": float(sum(token_counts[index] for index in others)),
        "other_context_embedding_redundancy": _pair_redundancy(embeddings, others),
    }


def _release_generator(generator: Any) -> None:
    del generator
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _write_smoke(
    args: argparse.Namespace, project_root: Path, config_path: Path,
    phase: Mapping[str, Any], generator: Mapping[str, Any], sample_offset: int, max_samples: int,
) -> Path:
    target = args.smoke_manifest
    if not target.is_absolute():
        target = project_root / target
    payload = _base_manifest(project_root, config_path, phase, generator, sample_offset, max_samples)
    payload.update({
        "status": "smoke_validated", "smoke_only": True,
        "runtime": {"data_access": False, "model_load": False},
    })
    _write_json(target.resolve(), payload)
    print(f"Wrote Phase 8A smoke manifest: {target.resolve()}")
    return target.resolve()


def run(args: argparse.Namespace) -> Path | None:
    project_root = _project_root()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config_path = config_path.resolve()
    phase = _load_config(config_path, args.generator)
    max_samples = phase["max_samples"] if args.max_samples is None else int(args.max_samples)
    sample_offset = phase["sample_offset"] if args.sample_offset is None else int(args.sample_offset)
    if not 1 <= max_samples <= phase["max_samples"]:
        raise DiagnosticError(f"--max-samples must be between 1 and {phase['max_samples']}")
    if sample_offset < phase["sample_offset"]:
        raise DiagnosticError("--sample-offset cannot move before the frozen held-out slice")

    generator_spec = phase["generators"][0]
    resolved_generator = _resolve_generator(
        generator_spec, args.generator, require_resolved=not bool(args.validate_only or args.smoke_manifest)
    )
    if args.validate_only:
        print(json.dumps({
            "status": "valid", "phase": phase["name"], "diagnostic_only": True, "selection_mutation": False,
            "sample_offset": sample_offset, "max_samples": max_samples,
            "diagnostic_plugins": phase["diagnostic_plugins"],
            "generations_per_question": phase["randomization"]["max_generations_per_question"],
            "generator": {"model_id": generator_spec["model_id"], "revision": generator_spec["revision"],
                          "resolution": resolved_generator["resolution"]},
        }, sort_keys=True))
        return None
    if args.smoke_manifest is not None:
        return _write_smoke(args, project_root, config_path, phase, resolved_generator, sample_offset, max_samples)

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
    manifest = _base_manifest(project_root, config_path, phase, resolved_generator, sample_offset, max_samples)
    manifest.update({"status": "running", "output_root": str(output_root.resolve()), "generator_status": {"reference": {"status": "starting"}}})
    _write_json(run_dir / "manifest.json", manifest)

    # Stable-core imports are deliberately below validation and smoke paths.
    from applications.rag.answer_scorer import make_answer_scorer
    from applications.rag.data import load_dataset_for_rag, make_corpus_manager
    from applications.rag.evaluation.metrics import evaluate_answer
    from applications.rag.generation import Generator
    from applications.rag.retrieval import make_encoder
    from applications.rag.selector import select_passages

    requested = sample_offset + max_samples
    questions = load_dataset_for_rag(phase["dataset"], phase["split"], requested)
    questions = questions[sample_offset:requested]
    if len(questions) != max_samples:
        raise DiagnosticError(f"held-out slice returned {len(questions)} questions, expected {max_samples}")
    encoder = make_encoder("dpr", device=args.device)
    corpus_manager = make_corpus_manager(
        "wiki_dpr", {"wiki_dpr_config": phase["wiki_dpr_config"], "nprobe": phase["wiki_dpr_nprobe"]}
    )
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend=phase["answer_scorer"]["backend"], device=args.device)
    generator = Generator(
        resolved_generator["model_path"], device=args.device,
        max_new_tokens=int(phase["max_new_tokens"]), use_chat_template=True,
    )
    manifest["generator_status"]["reference"] = {"status": "running", "generations": 0}
    _write_json(run_dir / "manifest.json", manifest)

    probe_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    probe_config = phase["randomization"]
    for question_index, item in enumerate(questions):
        question = str(item["question"])
        question_id = stable_id(str(item.get("id", question_index)), namespace="phase8a-question")
        query_embedding = encoder.encode_queries([question])[0]
        _, embeddings, passages, _retrieval_scores = corpus_manager.retrieve_with_embeddings(
            query_embedding, phase["top_k_retrieval"]
        )
        answer_scores = scorer.score_passages(question, passages)
        selected = np.asarray(select_passages(
            query_embedding, embeddings, K=phase["selection"]["K"],
            method=phase["selection"]["method"], relevance_scores=answer_scores,
            enhancers=phase["selection"]["enhancers"],
            enhancer_configs=phase["selection"]["enhancer_configs"],
            seed=phase["selection"]["seed"],
        ), dtype=np.int64)
        selected_ranks = [int(value) for value in selected.tolist()]
        if len(selected_ranks) != 5 or len(set(selected_ranks)) != 5:
            raise DiagnosticError("baseline selector did not return five unique passage ranks")
        probes = build_context_probes(
            question_id, selected_ranks, seed=probe_config["seed"],
            complement_pairs=probe_config["complement_pairs"],
            include_full_anchor=probe_config["include_full_anchor"],
            include_empty_anchor=probe_config["include_empty_anchor"],
            randomize_included_order=probe_config["randomize_included_order"],
        )
        if len(probes) != int(probe_config["max_generations_per_question"]):
            raise DiagnosticError("probe generation cap or design count mismatch")
        gold_answers = item.get("answers", [])
        passage_ids = [stable_id(str(passages[rank]), namespace="phase8a-passage") for rank in selected_ranks]
        for probe in probes:
            context = [passages[int(rank)] for rank in probe.ordered_ranks]
            generation_started = time.perf_counter()
            prediction = generator.generate(question, context)
            quality = evaluate_answer(prediction, gold_answers)
            generation_time_ms = (time.perf_counter() - generation_started) * 1000.0
            prediction_id = stable_id(prediction, namespace="phase8a-prediction:reference")
            probe_rows.append({
                "question_id": question_id, "probe_id": probe.probe_id,
                "pair_index": probe.pair_index, "anchor": probe.anchor,
                "mask": list(probe.mask), "ordered_ranks": list(probe.ordered_ranks),
                "selected_ranks": list(selected_ranks), "selected_passage_ids": passage_ids,
                "propensity": probe.propensity, "prediction_id": prediction_id,
                "em": float(quality["em"]), "f1": float(quality["f1"]),
                "generation_time_ms": float(generation_time_ms),
            })
            if probe.anchor == "effect":
                for position, treatment in enumerate(probe.mask):
                    effect_rows.append({
                        "question_id": question_id, "probe_id": probe.probe_id,
                        "pair_index": int(probe.pair_index), "passage_index": int(position),
                        "mask": list(probe.mask), "treatment": int(treatment),
                        "propensity": 0.5, "outcome": float(quality["f1"]),
                        "em_outcome": float(quality["em"]),
                        "covariates": _covariates(
                            position, probe.mask, selected_ranks, answer_scores, passages, embeddings
                        ),
                    })
        manifest["generator_status"]["reference"]["generations"] = len(probe_rows)
        if (question_index + 1) % 10 == 0 or question_index + 1 == len(questions):
            _write_json(run_dir / "manifest.json", manifest)
            print(f"  reference: {question_index + 1}/{len(questions)} questions")

    validate_compact_payload({"probe_rows": probe_rows, "effect_rows": effect_rows}, FORBIDDEN_FIELDS)
    _write_json(run_dir / "diagnostics.json", {
        "schema_version": 1, "diagnostic_only": True,
        "probe_rows": probe_rows, "effect_rows": effect_rows,
    })
    estimator = phase["estimator"]
    summary_kwargs = {
        "folds": estimator["folds"], "fold_seed": estimator["fold_seed"],
        "ridge": float(estimator["ridge"]), "bootstrap_samples": estimator["bootstrap"]["samples"],
        "bootstrap_seed": estimator["bootstrap"]["seed"], "placebo_repetitions": estimator["placebo"]["repetitions"],
        "placebo_seed": estimator["bootstrap"]["seed"] + 1, "gate": phase["gate"],
    }
    primary = summarize_counterfactual_influence(effect_rows, outcome_name="f1", **summary_kwargs)
    em_rows = [dict(row, outcome=row["em_outcome"]) for row in effect_rows]
    secondary = summarize_counterfactual_influence(em_rows, outcome_name="em", **summary_kwargs)
    analysis = {"schema_version": 1, "diagnostic_only": True, "primary": primary, "secondary": {"em": secondary}}
    validate_compact_payload(analysis, FORBIDDEN_FIELDS)
    _write_json(run_dir / "analysis/summary.json", analysis)
    _release_generator(generator)
    manifest["status"] = "completed"
    manifest["generator_status"]["reference"]["status"] = "completed"
    manifest["timing_ms"] = {"total": (time.perf_counter() - started) * 1000.0}
    manifest["summary"] = {
        "questions": len(questions), "probe_rows": len(probe_rows), "effect_rows": len(effect_rows),
        "generations": len(probe_rows), "gate_status": primary["gate"]["status"],
        "gate_decision": primary["gate"]["decision"],
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"Completed Phase 8A counterfactual influence diagnostic: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (DiagnosticError, OSError, ValueError, KeyError, subprocess.SubprocessError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
