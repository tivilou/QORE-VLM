#!/usr/bin/env python3
"""Run the report-only Phase 10D minimal selected-set support diagnostic."""

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
from scripts.collab.five_ideas.phase10d_metrics import (
    EXPECTED_MASKS,
    FORBIDDEN_FIELDS,
    Phase10DError,
    summarize_support_topology,
)
from scripts.collab.five_ideas.phase9g_probe import run_context_probe, run_gold_answer_copy_probe
from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_PLUGIN_ORDER = (
    "frozen_qore_context_observer",
    "gold_copy_eligibility_observer",
    "ordered_selected_subset_scheduler",
    "subset_outcome_oracle",
    "support_topology_compact_audit",
)
EXPECTED_SCREEN_SLICE = (1800, 50)


class Phase10DConfigError(RuntimeError):
    pass


def _root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase10DConfigError("cannot locate project root")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
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


def _load_config(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage != "screen":
        raise Phase10DConfigError("replication is frozen but unavailable pending a passed screen gate and explicit user approval")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Phase10DConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != "phase10d_minimal_set_support_topology" or int(phase.get("schema_version", -1)) != 1:
        raise Phase10DConfigError("unexpected Phase 10D config")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False or phase.get("authorization") != "implemented_observation_only":
        raise Phase10DConfigError("Phase 10D must remain an implemented observation-only diagnostic")
    dataset = phase.get("dataset")
    if not isinstance(dataset, Mapping) or (dataset.get("name"), dataset.get("split")) != ("nq_open", "validation"):
        raise Phase10DConfigError("dataset identity is not frozen")
    screen = dataset.get("screen")
    if not isinstance(screen, Mapping) or (
        int(screen.get("sample_offset", -1)),
        int(screen.get("max_samples", -1)),
        bool(screen.get("fresh_slice")),
    ) != (*EXPECTED_SCREEN_SLICE, True):
        raise Phase10DConfigError("screen slice is not frozen")
    replication = dataset.get("replication")
    if not isinstance(replication, Mapping) or (
        int(replication.get("sample_offset", -1)),
        int(replication.get("max_samples", -1)),
        bool(replication.get("fresh_slice")),
        replication.get("authorization"),
    ) != (1850, 50, True, "unavailable_pending_screen_gate_and_user_approval"):
        raise Phase10DConfigError("replication slice or authorization is not frozen")
    expected_retrieval = {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "initial_top_k": 50}
    if phase.get("retrieval") != expected_retrieval:
        raise Phase10DConfigError("retrieval contract is not frozen")
    expected_selection = {
        "method": "qore",
        "K": 5,
        "num_reads": 100,
        "lam": 2.0,
        "seed": 42,
        "gamma": 1.0,
        "delta": 0.0,
        "complementarity_method": None,
        "qore_prefilter_size": None,
        "direct_solve_max_n": 20,
        "use_answer_scorer": True,
        "answer_scorer_backend": "dpr",
    }
    if phase.get("selection") != expected_selection:
        raise Phase10DConfigError("selection contract is not frozen")
    generator = phase.get("generator")
    if not isinstance(generator, Mapping) or generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise Phase10DConfigError("generator contract is not frozen")
    expected_probe = {
        "plugin_id": "minimal_set_support_topology",
        "selected_context_only": True,
        "selected_K": 5,
        "preserve_selected_order": True,
        "nonempty_subset_count": 31,
        "full_mask_reuses_frozen_baseline": True,
        "eligibility": {"selected_hit": True, "baseline_em": 0.0, "gold_answer_copy_em": 1.0, "decision_order": "post_inference_only"},
        "success_metric": "em",
        "singleton_definition": "subset_cardinality_equals_1",
        "distributed_set_interaction": "all_singletons_fail_and_any_multi_passage_subset_succeeds",
        "strict_interaction": "distributed_set_interaction_with_successful_subset_containing_non_answer_literal_member",
        "gold_or_evaluator_before_generation": False,
        "qore_feedback": False,
        "prompt_change": False,
        "evaluator_input": False,
    }
    if phase.get("support_topology") != expected_probe:
        raise Phase10DConfigError("minimal-set support topology contract is not frozen")
    gate = phase.get("gate", {}).get("screen")
    expected_gate = {
        "minimum_primary_errors": 10,
        "minimum_copy_control_successes": 8,
        "strict_interaction_fraction_min": 0.40,
        "strict_interaction_bootstrap_ci95_low_min": 0.20,
        "bootstrap_repetitions": 2000,
        "bootstrap_seed": 10601,
    }
    if not isinstance(gate, Mapping) or any(gate.get(key) != value for key, value in expected_gate.items()):
        raise Phase10DConfigError("screen gate is not frozen")
    outputs = phase.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("compact_only") is not True or outputs.get("result_exchange_only") is not True or outputs.get("root") != "exchange/five_ideas/phase10d_minimal_set_support_topology":
        raise Phase10DConfigError("output contract is not frozen")
    if set(outputs.get("forbidden_fields", ())) != FORBIDDEN_FIELDS:
        raise Phase10DConfigError("compact forbidden fields are not frozen")
    return phase, dict(screen)


def validate_contract(config_path: Path, plan_path: Path, dossier_path: Path, evidence_path: Path, stage: str = "screen") -> dict[str, Any]:
    """Validate frozen Phase 10D artifacts without loading a model or Wiki-DPR."""

    phase, stage_spec = _load_config(config_path, stage)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase10DConfigError(f"cannot read Phase 10D preregistration artifact: {exc}") from exc
    if not isinstance(plan, Mapping) or plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v2" or plan.get("project") != "Q-DUET-VLM" or plan.get("authorization") != "implemented":
        raise Phase10DConfigError("plugin plan identity is invalid")
    discovery = plan.get("discovery")
    composition = plan.get("composition")
    if not isinstance(discovery, Mapping) or not isinstance(composition, Mapping):
        raise Phase10DConfigError("plugin plan discovery or composition is invalid")
    if tuple(discovery.get("allowlist", ())) != EXPECTED_PLUGIN_ORDER or tuple(composition.get("order", ())) != EXPECTED_PLUGIN_ORDER:
        raise Phase10DConfigError("plugin allowlist or composition order does not match")
    if plan.get("reproducibility", {}).get("config_path") != "configs/experiments/phase10d_minimal_set_support_topology.yaml":
        raise Phase10DConfigError("plugin plan config path is not frozen")
    if not isinstance(dossier, Mapping) or dossier.get("schema_version") != "experiment-grounded-ideation.mechanism-recovery.v1":
        raise Phase10DConfigError("mechanism recovery dossier is invalid")
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != "experiment-grounded-ideation.experiment-evidence.v1":
        raise Phase10DConfigError("mechanism evidence packet is invalid")
    recovery = plan.get("recovery_contract")
    if not isinstance(recovery, Mapping) or recovery.get("candidate_id") != "phase10d_minimal_set_support_topology":
        raise Phase10DConfigError("plugin plan recovery candidate is invalid")
    if recovery.get("dossier_sha256") != _sha256(dossier_path):
        raise Phase10DConfigError("plugin plan mechanism dossier hash does not match")
    dossier_packet = dossier.get("evidence_packet")
    if not isinstance(dossier_packet, Mapping) or dossier_packet.get("sha256") != _sha256(evidence_path):
        raise Phase10DConfigError("mechanism dossier evidence packet hash does not match")
    return {
        "status": "valid",
        "phase": phase["name"],
        "stage": stage,
        "slice": stage_spec["slice"],
        "plugins": list(EXPECTED_PLUGIN_ORDER),
        "selection_mutation": False,
        "report_only": True,
        "model_loaded": False,
        "wiki_dpr_started": False,
    }


def _resolve_generator(root: Path, specification: Mapping[str, Any], override: str | None) -> dict[str, str]:
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
        candidates.append(Path(os.environ["HF_HOME"]) / "hub/models--NousResearch--Meta-Llama-3-8B-Instruct")
    attempted: list[str] = []
    for candidate in candidates:
        attempted.append(str(candidate))
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate if (candidate / "config.json").is_file() else snapshot
        if (resolved / "config.json").is_file():
            return {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_path": str(resolved.resolve()),
                "config_sha256": _sha256(resolved / "config.json"),
                "resolution": "cli_override" if override else "local_cache",
            }
    raise Phase10DConfigError("reference generator not found; attempted: " + ", ".join(attempted))


def _selected_context(
    query_embedding: Any,
    embeddings: Any,
    texts: Sequence[str],
    question: str,
    scorer: Any,
    selection: Mapping[str, Any],
) -> tuple[list[int], list[str]]:
    answer_scores = scorer.score_passages(question, list(texts))
    selected = select_passages(
        query_embedding,
        embeddings,
        K=int(selection["K"]),
        method="qore",
        num_reads=int(selection["num_reads"]),
        lam=float(selection["lam"]),
        gamma=float(selection["gamma"]),
        delta=float(selection["delta"]),
        complementarity_method=None,
        qore_prefilter_size=None,
        direct_solve_max_n=int(selection["direct_solve_max_n"]),
        answer_scorer=scorer,
        passage_texts=list(texts),
        question=question,
        seed=int(selection["seed"]),
        relevance_scores=answer_scores,
    )
    ids = [int(index) for index in selected]
    if len(ids) != 5 or len(set(ids)) != 5:
        raise Phase10DConfigError("frozen QORE did not return exactly five unique passages")
    return ids, [str(texts[index]) for index in ids]


def _empty_arm() -> dict[str, None | bool | float]:
    return {"attempted": False, "em": None, "f1": None, "generation_time_ms": None}


def _scored_arm(prediction: str, gold_answers: Sequence[str], generation_time_ms: float) -> dict[str, bool | float]:
    metrics = evaluate_answer(prediction, list(gold_answers))
    return {
        "attempted": True,
        "em": float(metrics["em"]),
        "f1": float(metrics["f1"]),
        "generation_time_ms": float(generation_time_ms),
    }


def _context_arm(generator: Generator, question: str, passages: Sequence[str], gold_answers: Sequence[str]) -> dict[str, bool | float]:
    probe = run_context_probe(generator, question, list(passages))
    return _scored_arm(probe.prediction, gold_answers, probe.generation_time_ms)


def _answer_match_flags(selected_texts: Sequence[str], gold_answers: Sequence[str]) -> list[bool]:
    return [
        any(answer_has_match_in_text(answer, passage) for answer in gold_answers)
        for passage in selected_texts
    ]


def _subset_outcomes(
    generator: Generator,
    question: str,
    selected_texts: Sequence[str],
    gold_answers: Sequence[str],
    answer_match_flags: Sequence[bool],
    baseline: Mapping[str, bool | float],
) -> list[dict[str, int | float | bool]]:
    if len(selected_texts) != 5 or len(answer_match_flags) != 5:
        raise Phase10DConfigError("subset probe requires exactly the frozen selected K=5 context")
    outcomes: list[dict[str, int | float | bool]] = []
    for mask in EXPECTED_MASKS:
        indices = [index for index in range(5) if mask & (1 << index)]
        cardinality = len(indices)
        answer_match_count = sum(1 for index in indices if answer_match_flags[index])
        reused = mask == 31
        if reused:
            arm = baseline
        else:
            arm = _context_arm(generator, question, [selected_texts[index] for index in indices], gold_answers)
        outcomes.append({
            "mask": mask,
            "cardinality": cardinality,
            "answer_match_count": answer_match_count,
            "contains_non_answer_literal": answer_match_count < cardinality,
            "em": float(arm["em"]),
            "f1": float(arm["f1"]),
            "generation_time_ms": float(arm["generation_time_ms"]),
            "reused_frozen_baseline": reused,
        })
    return outcomes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase10d_minimal_set_support_topology.yaml")
    parser.add_argument("--stage", choices=("screen", "replication"), default="screen")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _root()
    config_path = (args.config if args.config.is_absolute() else root / args.config).resolve()
    plan_path = root / "configs/experiments/phase10d_minimal_set_support_topology_plan.json"
    dossier_path = root / "configs/experiments/phase10d_minimal_set_support_topology_mechanism_recovery.json"
    evidence_path = root / "configs/experiments/phase10d_minimal_set_support_topology_evidence_packet.json"
    contract = validate_contract(config_path, plan_path, dossier_path, evidence_path, args.stage)
    if args.validate_only:
        print(json.dumps(contract, sort_keys=True))
        return None
    if args.bootstrap_reps is not None and args.bootstrap_reps < 100:
        raise Phase10DConfigError("--bootstrap-reps must be at least 100")
    phase, stage_spec = _load_config(config_path, args.stage)
    identity = _resolve_generator(root, phase["generator"], args.model_path)
    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{timestamp}_{args.stage}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{args.stage}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    plugin_files = (
        "applications/rag/generation/generator.py",
        "scripts/rag/eval/eval_rag_refactored.py",
        "scripts/collab/five_ideas/phase9g_probe.py",
        "scripts/collab/five_ideas/phase10d_metrics.py",
        "scripts/collab/five_ideas/run_phase10d_minimal_set_support_topology.py",
        "configs/experiments/phase10d_minimal_set_support_topology.yaml",
        "configs/experiments/phase10d_minimal_set_support_topology_plan.json",
        "configs/experiments/phase10d_minimal_set_support_topology_mechanism_recovery.json",
        "configs/experiments/phase10d_minimal_set_support_topology_evidence_packet.json",
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase["name"],
        "stage": args.stage,
        "status": "running",
        "diagnostic_only": True,
        "selection_mutation": False,
        "report_only": True,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "plugin_plan": {"path": str(plan_path), "sha256": _sha256(plan_path), "tree_sha256": _tree_sha256(root, plugin_files)},
        "mechanism_recovery": {"path": str(dossier_path), "sha256": _sha256(dossier_path)},
        "evidence_packet": {"path": str(evidence_path), "sha256": _sha256(evidence_path)},
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current")},
        "environment": {"python_executable": sys.executable, "python_version": sys.version},
        "generator": identity,
        "dataset": {"name": phase["dataset"]["name"], "split": phase["dataset"]["split"], **stage_spec},
        "retrieval": phase["retrieval"],
        "selection": phase["selection"],
        "support_topology": phase["support_topology"],
    }
    _write(run_dir / "run_metadata.json", metadata)
    sample_end = int(stage_spec["sample_offset"]) + int(stage_spec["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["split"], sample_end)[int(stage_spec["sample_offset"]):sample_end]
    if len(questions) != int(stage_spec["max_samples"]):
        raise Phase10DConfigError("fresh screen slice length mismatch")
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend="dpr")
    generator = Generator(identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        question_id = str(item["id"])
        question = str(item["question"])
        gold_answers = [str(value) for value in item.get("answers", []) if str(value).strip()]
        if not gold_answers:
            raise Phase10DConfigError(f"{question_id}: no non-empty gold answer")
        query_embedding = encoder.encode_queries([question])[0]
        _, embeddings, texts, _ = corpus_manager.retrieve_with_embeddings(query_embedding, int(phase["retrieval"]["initial_top_k"]))
        selection_started = time.perf_counter()
        selected_ids, selected_texts = _selected_context(query_embedding, embeddings, texts, question, scorer, phase["selection"])
        selection_time_ms = (time.perf_counter() - selection_started) * 1000.0
        if len(selected_ids) != 5 or len(selected_texts) != 5:
            raise Phase10DConfigError(f"{question_id}: selected context contract failed")
        retrieval_hit = any(answer_has_match_in_text(answer, passage) for answer in gold_answers for passage in texts)
        answer_flags = _answer_match_flags(selected_texts, gold_answers)
        selected_match_count = sum(answer_flags)
        selected_hit = selected_match_count > 0
        baseline = _context_arm(generator, question, selected_texts, gold_answers)
        primary_error = bool(selected_hit and baseline["em"] == 0.0)
        copy = _empty_arm()
        if primary_error:
            copy_target = min(gold_answers, key=lambda value: (len(value.split()), len(value), value.lower()))
            copy_probe = run_gold_answer_copy_probe(generator, question, copy_target)
            copy = _scored_arm(copy_probe.prediction, gold_answers, copy_probe.generation_time_ms)
        eligible = bool(primary_error and copy["em"] == 1.0)
        subsets: list[dict[str, int | float | bool]] = []
        if eligible:
            subsets = _subset_outcomes(generator, question, selected_texts, gold_answers, answer_flags, baseline)
        samples.append({
            "question_id": question_id,
            "retrieval_hit": retrieval_hit,
            "selected_hit": selected_hit,
            "selected_answer_match_count": selected_match_count,
            "selection_time_ms": selection_time_ms,
            "arms": {"baseline": baseline, "gold_answer_copy": copy},
            "subset_outcomes": subsets,
        })
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 10D minimal-set topology: {index}/{len(questions)}")
    result = {
        "schema_version": 1,
        "phase": phase["name"],
        "stage": args.stage,
        "diagnostic_only": True,
        "selection_mutation": False,
        "report_only": True,
        "config": {
            "dataset": stage_spec,
            "retrieval": phase["retrieval"],
            "selection": phase["selection"],
            "support_topology": {
                "plugin_id": phase["support_topology"]["plugin_id"],
                "selected_K": 5,
                "nonempty_subset_count": 31,
                "full_mask_reuses_frozen_baseline": True,
                "success_metric": "em",
            },
        },
        "samples": samples,
    }
    forbidden = _forbidden(result)
    if forbidden:
        raise Phase10DConfigError(f"compact result contains forbidden fields: {forbidden[:5]}")
    _write(run_dir / "result.json", result)
    gate = dict(phase["gate"][args.stage])
    if args.bootstrap_reps is not None:
        gate["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_support_topology(result, gate=gate, stage=args.stage)
    _write(run_dir / "summary.json", summary)
    metadata.update({
        "status": "completed",
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "summary": {"path": str(run_dir / "summary.json"), "primary_failure_class": summary["decision"]["primary_failure_class"]},
    })
    _write(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 10D minimal-set support topology: {run_dir}")
    print(f"Report-only gate: {summary['decision']['primary_failure_class']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (Phase10DConfigError, Phase10DError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
