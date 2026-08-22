#!/usr/bin/env python3
"""Run the collaborator-only Phase 9I candidate coverage diagnostic."""

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
    from scripts.collab.five_ideas.phase9i_metrics import (
        FORBIDDEN_FIELDS,
        Phase9IError,
        summarize_phase9i,
        validate_result,
    )
    from scripts.collab.five_ideas.phase9i_probe import (
        candidate_pairs,
        candidate_parse_status,
        fixed_candidate_permutation,
        normalize_candidate,
        run_evidence_constrained_probe,
        run_extractive_candidate_probe,
        run_verifier_probe,
    )
except ImportError:  # pragma: no cover
    from phase9h_probe import run_context_probe, run_gold_answer_copy_probe
    from phase9i_metrics import FORBIDDEN_FIELDS, Phase9IError, summarize_phase9i, validate_result
    from phase9i_probe import (
        candidate_pairs,
        candidate_parse_status,
        fixed_candidate_permutation,
        normalize_candidate,
        run_evidence_constrained_probe,
        run_extractive_candidate_probe,
        run_verifier_probe,
    )


class Phase9IConfigError(RuntimeError):
    """Raised when the frozen Phase 9I contract is malformed."""


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_PLUGINS = (
    "frozen_baseline_observer",
    "gold_answer_copy_control",
    "extractive_span_candidate",
    "evidence_constrained_candidate",
    "candidate_set_coverage_combiner",
    "grounded_candidate_verifier",
    "verifier_order_invariance_audit",
)
SELECTION_KEYS = {
    "method", "K", "num_reads", "lam", "seed", "gamma", "delta",
    "complementarity_method", "qore_prefilter_size", "direct_solve_max_n",
    "lambda_mmr", "saturation_alpha", "lambda_submodular",
    "dpp_quality_scale", "dpp_jitter", "use_answer_scorer",
    "answer_scorer_backend",
}


def _project_root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase9IConfigError("cannot locate project root")


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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
        root / "scripts/collab/five_ideas/phase9i_metrics.py",
        root / "scripts/collab/five_ideas/run_phase9i_candidate_coverage_grounded_ranking.py",
    )
    entries = []
    for path in paths:
        if not path.is_file():
            raise Phase9IConfigError(f"plugin tree file is missing: {path}")
        entries.append(
            {"path": str(path.relative_to(root)), "sha256": _sha256_file(path)}
        )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_config(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Phase9IConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict) or int(phase.get("schema_version", -1)) != 1:
        raise Phase9IConfigError("configuration must contain phase.schema_version: 1")
    if phase.get("name") != "phase9i_candidate_coverage_grounded_ranking":
        raise Phase9IConfigError("unexpected Phase 9I name")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise Phase9IConfigError("Phase 9I must be diagnostic-only with frozen selection")
    if stage not in {"screen", "formal", "replication"}:
        raise Phase9IConfigError(f"invalid stage: {stage}")
    if phase.get("authorization") not in {"plan_only", "implemented"}:
        raise Phase9IConfigError("authorization must be plan_only or implemented")
    required = {
        "dataset", "retrieval", "selection", "generator", "plugins",
        "candidate_contract", "verifier_contract", "eligibility", "gate", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise Phase9IConfigError(f"configuration missing keys: {missing}")
    for name in ("screen", "formal", "replication"):
        spec = phase["dataset"].get(name)
        if not isinstance(spec, dict):
            raise Phase9IConfigError(f"dataset.{name} is required")
        if int(spec.get("sample_offset", -1)) < 850 or int(spec.get("max_samples", 0)) < 1:
            raise Phase9IConfigError(f"dataset.{name} has invalid slice")
        if spec.get("fresh_slice") is not True:
            raise Phase9IConfigError(f"dataset.{name} must be fresh_slice")
    expected_slices = {
        "screen": (850, 50),
        "formal": (900, 200),
        "replication": (1100, 200),
    }
    actual = {
        name: (
            int(phase["dataset"][name]["sample_offset"]),
            int(phase["dataset"][name]["max_samples"]),
        )
        for name in expected_slices
    }
    if actual != expected_slices:
        raise Phase9IConfigError(f"dataset slices are not frozen: {actual}")
    retrieval = phase["retrieval"]
    expected_retrieval = {
        "corpus_mode": "wiki_dpr",
        "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64,
        "top_k": 50,
    }
    if retrieval != expected_retrieval:
        raise Phase9IConfigError(f"retrieval contract is not frozen: {retrieval}")
    selection = phase["selection"]
    if set(selection) != SELECTION_KEYS:
        raise Phase9IConfigError("selection keys are not frozen")
    expected_selection = {
        "method": "qore", "K": 5, "num_reads": 100, "lam": 2.0,
        "seed": 42, "gamma": 1.0, "delta": 0.0,
        "complementarity_method": None, "qore_prefilter_size": None,
        "direct_solve_max_n": 20, "lambda_mmr": 0.7,
        "saturation_alpha": 1.0, "lambda_submodular": 0.5,
        "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8,
        "use_answer_scorer": True, "answer_scorer_backend": "dpr",
    }
    if selection != expected_selection:
        raise Phase9IConfigError("selection values are not frozen")
    generator = phase["generator"]
    if (
        not isinstance(generator, dict)
        or generator.get("model_id") != MODEL_ID
        or generator.get("revision") != MODEL_REVISION
        or int(generator.get("max_new_tokens", -1)) != 32
        or generator.get("decoding") != "greedy"
    ):
        raise Phase9IConfigError("generator identity/settings are not frozen")
    plugins = phase["plugins"]
    if (
        not isinstance(plugins, dict)
        or tuple(plugins.get("allowlist", [])) != EXPECTED_PLUGINS
        or plugins.get("candidate_profile")
        != ["baseline_v1", "extractive_span_v1", "evidence_constrained_v1"]
        or plugins.get("verifier_profile") != "grounded_candidate_choice_v1"
        or plugins.get("order_audit_profile") != "fixed_permutation_v1"
        or plugins.get("diagnostic_outputs_used_for_selection") is not False
        or plugins.get("production_intervention") is not False
    ):
        raise Phase9IConfigError("plugin contract is not frozen")
    candidate_contract = phase["candidate_contract"]
    if (
        candidate_contract.get("candidate_count_max") != 3
        or candidate_contract.get("candidate_modes")
        != ["baseline_v1", "extractive_span_v1", "evidence_constrained_v1"]
        or candidate_contract.get("candidate_input") != "question_and_selected_context_only"
        or candidate_contract.get("candidate_text_persisted") is not False
        or candidate_contract.get("parse_failures_excluded_from_oracle") is not True
    ):
        raise Phase9IConfigError("candidate contract is not frozen")
    verifier_contract = phase["verifier_contract"]
    if (
        verifier_contract.get("gold_access") is not False
        or verifier_contract.get("evaluator_access") is not False
        or verifier_contract.get("selection_score_access") is not False
        or verifier_contract.get("passes_per_question") != 2
        or verifier_contract.get("second_pass_order") != "fixed_candidate_permutation"
        or verifier_contract.get("raw_output_persisted") is not False
    ):
        raise Phase9IConfigError("verifier contract is not frozen")
    outputs = phase["outputs"]
    if (
        outputs.get("root")
        != "exchange/five_ideas/phase9i_candidate_coverage_grounded_ranking"
        or outputs.get("compact_only") is not True
        or outputs.get("candidate_text_persisted") is not False
    ):
        raise Phase9IConfigError("output contract is not frozen")
    if phase.get("authorization") != "implemented":
        raise Phase9IConfigError(
            "runtime is disabled until the implementation authorization is recorded"
        )
    return phase, phase["dataset"][stage]


def _resolve_generator(
    root: Path, specification: Mapping[str, Any], override: str | None
) -> dict[str, Any]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    for raw in specification.get("model_path_candidates", []):
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    candidates.extend(
        [
            root / "models" / "llama3-8b",
            Path.home()
            / ".cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct",
        ]
    )
    if os.environ.get("HF_HOME"):
        candidates.append(
            Path(os.environ["HF_HOME"])
            / "hub/models--NousResearch--Meta-Llama-3-8B-Instruct"
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
    raise Phase9IConfigError(
        "reference generator not found; attempted: " + ", ".join(attempted)
    )


def _empty_standard() -> dict[str, Any]:
    return {
        "attempted": False,
        "em": None,
        "f1": None,
        "generation_time_ms": None,
    }


def _empty_candidate() -> dict[str, Any]:
    return {
        "attempted": False,
        "parse_status": "empty",
        "em": None,
        "f1": None,
        "generation_time_ms": None,
    }


def _empty_verifier() -> dict[str, Any]:
    return {
        "attempted": False,
        "original_choice_mode": None,
        "permuted_choice_mode": None,
        "order_agreement": None,
        "parse_status": "not_attempted",
        "em": None,
        "f1": None,
        "generation_time_ms": None,
    }


def _scored_arm(
    prediction: str, gold_answers: list[str], elapsed: float
) -> dict[str, Any]:
    metrics = evaluate_answer(prediction, gold_answers)
    return {
        "attempted": True,
        "em": float(metrics["em"]),
        "f1": float(metrics["f1"]),
        "generation_time_ms": float(elapsed),
    }


def _scored_candidate(
    prediction: str,
    parse_status: str,
    gold_answers: list[str],
    elapsed: float,
) -> dict[str, Any]:
    if parse_status not in {"ok", "abstain"}:
        return {
            "attempted": True,
            "parse_status": parse_status,
            "em": None,
            "f1": None,
            "generation_time_ms": None,
        }
    metrics = evaluate_answer(prediction, gold_answers)
    return {
        "attempted": True,
        "parse_status": parse_status,
        "em": float(metrics["em"]),
        "f1": float(metrics["f1"]),
        "generation_time_ms": float(elapsed),
    }


def _candidate_metrics(
    pairs: Sequence[tuple[str, str]], gold_answers: list[str]
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    if not pairs:
        raise Phase9IConfigError("candidate set has no valid candidates")
    metrics: dict[str, dict[str, float]] = {}
    for mode, text in pairs:
        result = evaluate_answer(text, gold_answers)
        metrics[mode] = {
            "em": float(result["em"]),
            "f1": float(result["f1"]),
        }
    oracle = {
        "attempted": True,
        "candidate_count": len(pairs),
        "unique_candidate_count": len(
            {normalize_candidate(text) for _, text in pairs}
        ),
        "parse_failures": 3 - len(pairs),
        "oracle_em": max(value["em"] for value in metrics.values()),
        "oracle_f1": max(value["f1"] for value in metrics.values()),
    }
    return oracle, metrics


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/phase9i_candidate_coverage_grounded_ranking.yaml",
    )
    parser.add_argument(
        "--stage", choices=("screen", "formal", "replication"), default="screen"
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
    phase, stage_spec = _load_config(config_path, args.stage)
    if args.bootstrap_reps is not None and args.bootstrap_reps < 100:
        raise Phase9IConfigError("--bootstrap-reps must be at least 100")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "phase": phase["name"],
                    "stage": args.stage,
                    "slice": stage_spec["slice"],
                    "plugins": list(EXPECTED_PLUGINS),
                    "selection_mutation": False,
                    "report_only": True,
                    "runtime_enabled": phase["authorization"] == "implemented",
                },
                sort_keys=True,
            )
        )
        return None
    generator_identity = _resolve_generator(
        root, phase["generator"], args.model_path
    )
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

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase["name"],
        "stage": args.stage,
        "status": "running",
        "diagnostic_only": True,
        "selection_mutation": False,
        "report_only": True,
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        },
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "status": _git(root, "status", "--short", "--branch"),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "generator": generator_identity,
        "dataset": {
            "name": phase["dataset"]["name"],
            "split": phase["dataset"]["split"],
            "sample_offset": stage_spec["sample_offset"],
            "max_samples": stage_spec["max_samples"],
            "slice": stage_spec["slice"],
            "fresh_slice": True,
        },
        "retrieval": phase["retrieval"],
        "selection": phase["selection"],
        "plugins": {
            "allowlist": list(EXPECTED_PLUGINS),
            "tree_sha256": _plugin_tree_hash(root),
            "candidate_profile": phase["plugins"]["candidate_profile"],
            "verifier_profile": phase["plugins"]["verifier_profile"],
            "order_audit_profile": phase["plugins"]["order_audit_profile"],
            "diagnostic_outputs_used_for_selection": False,
        },
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    sample_end = int(stage_spec["sample_offset"]) + int(stage_spec["max_samples"])
    questions = load_dataset_for_rag(
        phase["dataset"]["name"], phase["dataset"]["split"], sample_end
    )
    questions = questions[int(stage_spec["sample_offset"]):sample_end]
    if len(questions) != int(stage_spec["max_samples"]):
        raise Phase9IConfigError(
            f"fresh slice returned {len(questions)} questions, expected {stage_spec['max_samples']}"
        )

    np.random.seed(int(phase["selection"]["seed"]))
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager(
        "wiki_dpr",
        {
            "wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"],
            "nprobe": int(phase["retrieval"]["nprobe"]),
        },
    )
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
        gold_answers = [
            str(value) for value in item.get("answers", []) if str(value).strip()
        ]
        if not gold_answers:
            raise Phase9IConfigError(f"{question_id} has no non-empty gold answer")
        query_embedding = encoder.encode_queries([question])[0]
        _, retrieved_embeddings, retrieved_texts, _ = corpus_manager.retrieve_with_embeddings(
            query_embedding, int(phase["retrieval"]["top_k"])
        )
        if len(retrieved_texts) != int(phase["retrieval"]["top_k"]):
            raise Phase9IConfigError(f"{question_id}: incomplete retrieval")
        answer_scores = answer_scorer.score_passages(question, retrieved_texts)
        selection_started = time.perf_counter()
        selected_local = select_passages(
            query_embedding,
            retrieved_embeddings,
            K=int(phase["selection"]["K"]),
            method="qore",
            num_reads=int(phase["selection"]["num_reads"]),
            lam=float(phase["selection"]["lam"]),
            gamma=float(phase["selection"]["gamma"]),
            delta=float(phase["selection"]["delta"]),
            complementarity_method=phase["selection"]["complementarity_method"],
            qore_prefilter_size=phase["selection"]["qore_prefilter_size"],
            direct_solve_max_n=int(phase["selection"]["direct_solve_max_n"]),
            lambda_mmr=float(phase["selection"]["lambda_mmr"]),
            saturation_alpha=float(phase["selection"]["saturation_alpha"]),
            lambda_submodular=float(phase["selection"]["lambda_submodular"]),
            dpp_quality_scale=float(phase["selection"]["dpp_quality_scale"]),
            dpp_jitter=float(phase["selection"]["dpp_jitter"]),
            answer_scorer=answer_scorer,
            passage_texts=retrieved_texts,
            question=question,
            seed=int(phase["selection"]["seed"]),
            relevance_scores=answer_scores,
        )
        selection_time_ms = (time.perf_counter() - selection_started) * 1000.0
        selected_texts = [retrieved_texts[int(value)] for value in selected_local]
        retrieval_match_count = sum(
            any(
                answer_has_match_in_text(answer, passage)
                for answer in gold_answers
            )
            for passage in retrieved_texts
        )
        selected_match_count = sum(
            any(
                answer_has_match_in_text(answer, passage)
                for answer in gold_answers
            )
            for passage in selected_texts
        )
        retrieval_hit = retrieval_match_count > 0
        selected_hit = selected_match_count > 0

        baseline_result = run_context_probe(generator, question, selected_texts)
        baseline = _scored_arm(
            baseline_result.prediction,
            gold_answers,
            baseline_result.generation_time_ms,
        )
        primary_error = bool(selected_hit and baseline["em"] == 0.0)
        copy_control = _empty_standard()
        if primary_error:
            copy_target = min(
                gold_answers,
                key=lambda value: (len(value.split()), len(value), value.lower()),
            )
            copy_result = run_gold_answer_copy_probe(
                generator, question, copy_target
            )
            copy_control = _scored_arm(
                copy_result.prediction,
                gold_answers,
                copy_result.generation_time_ms,
            )
        copy_success = bool(primary_error and copy_control["em"] == 1.0)
        extractive_arm = _empty_candidate()
        constrained_arm = _empty_candidate()
        candidate_set = {
            "attempted": False,
            "candidate_count": 0,
            "unique_candidate_count": 0,
            "parse_failures": 0,
            "oracle_em": None,
            "oracle_f1": None,
        }
        verifier = _empty_verifier()

        if copy_success:
            extractive_result = run_extractive_candidate_probe(
                generator, question, selected_texts
            )
            constrained_result = run_evidence_constrained_probe(
                generator, question, selected_texts
            )
            extractive_arm = _scored_candidate(
                extractive_result.text,
                extractive_result.parse_status,
                gold_answers,
                extractive_result.generation_time_ms,
            )
            constrained_arm = _scored_candidate(
                constrained_result.text,
                constrained_result.parse_status,
                gold_answers,
                constrained_result.generation_time_ms,
            )
            pairs = candidate_pairs(
                baseline_result.prediction, extractive_result, constrained_result
            )
            if not pairs:
                raise Phase9IConfigError(
                    f"{question_id}: eligible diagnostic has no valid candidate"
                )
            candidate_set, candidate_metric_map = _candidate_metrics(
                pairs, gold_answers
            )
            original_verifier = run_verifier_probe(
                generator, question, selected_texts, pairs
            )
            permuted_pairs = fixed_candidate_permutation(pairs)
            permuted_verifier = run_verifier_probe(
                generator, question, selected_texts, permuted_pairs
            )
            original_choice = original_verifier.choice_mode
            permuted_choice = permuted_verifier.choice_mode
            verifier_valid = (
                original_verifier.parse_status == "ok"
                and permuted_verifier.parse_status == "ok"
                and original_choice is not None
                and permuted_choice is not None
                and original_choice in candidate_metric_map
            )
            if verifier_valid:
                chosen_metrics = candidate_metric_map[original_choice]
                verifier = {
                    "attempted": True,
                    "original_choice_mode": original_choice,
                    "permuted_choice_mode": permuted_choice,
                    "order_agreement": original_choice == permuted_choice,
                    "parse_status": "ok",
                    "em": chosen_metrics["em"],
                    "f1": chosen_metrics["f1"],
                    "generation_time_ms": (
                        original_verifier.generation_time_ms
                        + permuted_verifier.generation_time_ms
                    ),
                }
            else:
                verifier = {
                    "attempted": True,
                    "original_choice_mode": original_choice,
                    "permuted_choice_mode": permuted_choice,
                    "order_agreement": (
                        original_choice == permuted_choice
                        if original_choice is not None and permuted_choice is not None
                        else None
                    ),
                    "parse_status": "invalid_choice",
                    "em": None,
                    "f1": None,
                    "generation_time_ms": (
                        original_verifier.generation_time_ms
                        + permuted_verifier.generation_time_ms
                    ),
                }

        sample = {
            "question_id": question_id,
            "retrieval_hit": retrieval_hit,
            "selected_hit": selected_hit,
            "retrieval_match_count": retrieval_match_count,
            "selected_match_count": selected_match_count,
            "primary_error": primary_error,
            "copy_control_success": copy_success,
            "eligible": copy_success,
            "selection_time_ms": selection_time_ms,
            "arms": {
                "baseline": baseline,
                "gold_answer_copy": copy_control,
                "extractive_span": extractive_arm,
                "evidence_constrained": constrained_arm,
            },
            "candidate_set": candidate_set,
            "verifier": verifier,
        }
        samples.append(sample)
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 9I {args.stage}: {index}/{len(questions)}")

    result = {
        "schema_version": 1,
        "phase": phase["name"],
        "stage": args.stage,
        "diagnostic_only": True,
        "selection_mutation": False,
        "report_only": True,
        "config": {
            "dataset": phase["dataset"]["name"],
            "split": phase["dataset"]["split"],
            "sample_offset": stage_spec["sample_offset"],
            "max_samples": stage_spec["max_samples"],
            "slice": stage_spec["slice"],
            "corpus_mode": phase["retrieval"]["corpus_mode"],
            "wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"],
            "nprobe": phase["retrieval"]["nprobe"],
            "top_k_retrieval": phase["retrieval"]["top_k"],
            "selection": phase["selection"],
            "candidate_profile": phase["plugins"]["candidate_profile"],
            "verifier_profile": phase["plugins"]["verifier_profile"],
            "order_audit_profile": phase["plugins"]["order_audit_profile"],
        },
        "samples": samples,
    }
    forbidden = _find_forbidden(result)
    if forbidden:
        raise Phase9IConfigError(
            f"compact result contains forbidden fields: {forbidden[:5]}"
        )
    validate_result(result)
    _write_json(run_dir / "result.json", result)
    gate = phase["gate"]
    if args.bootstrap_reps is not None:
        gate = dict(gate)
        gate["formal"] = dict(gate["formal"])
        gate["formal"]["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_phase9i(result, gate=gate)
    _write_json(run_dir / "summary.json", summary)
    metadata.update(
        {
            "status": "completed",
            "timing_ms": {
                "total": (time.perf_counter() - started) * 1000.0
            },
            "summary": {
                "path": str(run_dir / "summary.json"),
                "primary_failure_class": summary["decision"]["primary_failure_class"],
            },
        }
    )
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 9I {args.stage}: {run_dir}")
    print(f"Report-only gate: {summary['decision']['primary_failure_class']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (
        Phase9IConfigError,
        Phase9IError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
