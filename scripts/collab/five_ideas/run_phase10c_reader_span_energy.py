#!/usr/bin/env python3
"""Run the report-only Phase 10C reader-span energy decoding screen."""

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
from applications.rag.reader_span_energy import (
    DecodeResult,
    SpanLattice,
    decode_with_span_energy,
    lattice_from_reader_hypotheses,
    matched_control_lattice,
)
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from scripts.collab.five_ideas.phase10c_metrics import FORBIDDEN_FIELDS, Phase10CError, summarize_phase10c
from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_ARMS = ("baseline_greedy", "reader_span_energy", "matched_span_control")
EXPECTED_SCREEN_SLICE = (3150, 50)
EXPECTED_PLUGIN_ORDER = (
    "frozen_qore_context_observer",
    "selected_context_reader_lattice",
    "reader_span_energy_decoder",
    "matched_span_control_decoder",
    "decoder_compact_audit",
)


class Phase10CConfigError(RuntimeError):
    pass


def _root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase10CConfigError("cannot locate project root")


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


def _load(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage != "screen":
        raise Phase10CConfigError("only the user-authorized report-only screen is executable")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Phase10CConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != "phase10c_reader_span_energy" or int(phase.get("schema_version", -1)) != 1:
        raise Phase10CConfigError("unexpected Phase 10C config")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False or phase.get("authorization") != "implemented_screen_only":
        raise Phase10CConfigError("Phase 10C must remain an implemented screen-only diagnostic")
    dataset = phase.get("dataset", {})
    if (dataset.get("name"), dataset.get("split")) != ("nq_open", "validation"):
        raise Phase10CConfigError("dataset identity is not frozen")
    screen = dataset.get("screen")
    if not isinstance(screen, dict):
        raise Phase10CConfigError("dataset.screen is missing")
    if (int(screen.get("sample_offset", -1)), int(screen.get("max_samples", -1)), bool(screen.get("fresh_slice"))) != (*EXPECTED_SCREEN_SLICE, True):
        raise Phase10CConfigError("screen slice is not frozen")
    if phase.get("retrieval") != {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "initial_top_k": 50}:
        raise Phase10CConfigError("retrieval contract is not frozen")
    expected_selection = {"method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42, "gamma": 1.0, "delta": 0.0, "complementarity_method": None, "qore_prefilter_size": None, "direct_solve_max_n": 20, "use_answer_scorer": True, "answer_scorer_backend": "dpr"}
    if phase.get("selection") != expected_selection:
        raise Phase10CConfigError("selection contract is not frozen")
    generator = phase.get("generator", {})
    if generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise Phase10CConfigError("generator contract is not frozen")
    reader = phase.get("reader_span_energy", {})
    expected_reader = {
        "plugin_id": "reader_span_energy_decoding", "span_source": "frozen_selected_context_only", "reader_backend": "dpr", "top_m": 3,
        "max_answer_span_tokens": 10, "max_spans": 10, "energy_rule": "log1p_reader_mass", "energy_coefficient": 1.0,
        "candidate_generation": False, "candidate_reranking": False, "applicability_gate": False, "qore_feedback": False,
        "prompt_change": False, "evaluator_input": False, "final_context_K": 5,
    }
    if reader != expected_reader:
        raise Phase10CConfigError("reader-span energy contract is not frozen")
    if tuple(phase.get("arms", [])) != EXPECTED_ARMS:
        raise Phase10CConfigError("arm order is not frozen")
    gate = phase.get("gate", {}).get("screen")
    if not isinstance(gate, dict) or float(gate.get("total_pipeline_cost_ratio_max", -1.0)) != 1.25 or float(gate.get("screen_f1_ci95_low_min", 1.0)) != -0.02 or float(gate.get("screen_reader_minus_control_mean_min", -1.0)) != 0.0:
        raise Phase10CConfigError("screen gate is not frozen")
    outputs = phase.get("outputs", {})
    if outputs.get("compact_only") is not True or outputs.get("root") != "exchange/five_ideas/phase10c_reader_span_energy":
        raise Phase10CConfigError("output contract is not frozen")
    if set(outputs.get("forbidden_fields", [])) != FORBIDDEN_FIELDS:
        raise Phase10CConfigError("compact forbidden fields are not frozen")
    return phase, screen


def validate_contract(config_path: Path, plan_path: Path, stage: str = "screen") -> dict[str, Any]:
    """Validate the screen contract without loading a model or Wiki-DPR."""

    phase, stage_spec = _load(config_path, stage)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase10CConfigError(f"cannot read plugin plan: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v1":
        raise Phase10CConfigError("plugin plan schema is invalid")
    if plan.get("project") != "Q-DUET-VLM" or plan.get("authorization") != "implemented":
        raise Phase10CConfigError("plugin plan authorization is invalid")
    discovery = plan.get("discovery")
    composition = plan.get("composition")
    if not isinstance(discovery, dict) or discovery.get("mode") != "explicit_allowlist" or not isinstance(composition, dict):
        raise Phase10CConfigError("plugin plan discovery or composition is invalid")
    allowlist = tuple(discovery.get("allowlist", ()))
    order = tuple(composition.get("order", ()))
    if allowlist != EXPECTED_PLUGIN_ORDER or order != EXPECTED_PLUGIN_ORDER or allowlist != order:
        raise Phase10CConfigError("plugin allowlist and composition order do not match")
    if plan.get("reproducibility", {}).get("config_path") != "configs/experiments/phase10c_reader_span_energy.yaml":
        raise Phase10CConfigError("plugin plan config path is not frozen")
    return {
        "status": "valid", "phase": phase["name"], "stage": "screen", "slice": stage_spec["slice"], "arms": EXPECTED_ARMS,
        "selection_mutation": False, "report_only": True, "model_loaded": False, "wiki_dpr_started": False,
    }


def _resolve_generator(root: Path, specification: Mapping[str, Any], override: str | None) -> dict[str, str]:
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
        snapshot = candidate / "snapshots" / MODEL_REVISION
        resolved = candidate if (candidate / "config.json").is_file() else snapshot
        if (resolved / "config.json").is_file():
            return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "model_path": str(resolved.resolve()), "config_sha256": _sha256(resolved / "config.json"), "resolution": "cli_override" if override else "local_cache"}
    raise Phase10CConfigError("reference generator not found; attempted: " + ", ".join(attempted))


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
        query_embedding, embeddings, K=int(selection["K"]), method="qore", num_reads=int(selection["num_reads"]), lam=float(selection["lam"]),
        gamma=float(selection["gamma"]), delta=float(selection["delta"]), complementarity_method=None, qore_prefilter_size=None,
        direct_solve_max_n=int(selection["direct_solve_max_n"]), answer_scorer=scorer, passage_texts=list(texts), question=question,
        seed=int(selection["seed"]), relevance_scores=answer_scores,
    )
    ids = [int(index) for index in selected]
    if len(ids) != 5 or len(set(ids)) != 5:
        raise Phase10CConfigError("frozen QORE did not return exactly five unique passages")
    return ids, [str(texts[index]) for index in ids]


def _prediction_token_count(generator: Any, prediction: str) -> int:
    encoded = generator.tokenizer(prediction, add_special_tokens=False)
    values = encoded.get("input_ids", []) if isinstance(encoded, Mapping) else []
    if values and isinstance(values[0], list):
        values = values[0]
    return len(values)


def _compact_sample(
    question_id: str,
    prediction: str,
    decode: DecodeResult | None,
    generator: Any,
    gold_answers: Sequence[str],
    selected_texts: Sequence[str],
    *,
    generation_time_ms: float,
    pipeline_time_ms: float,
    lattice: SpanLattice,
    decoder_mode: str,
) -> dict[str, Any]:
    metrics = evaluate_answer(prediction, list(gold_answers))
    if decode is None:
        generated_tokens, active_steps = _prediction_token_count(generator, prediction), 0
    else:
        generated_tokens, active_steps = decode.generated_token_count, decode.energy_active_steps
    return {
        "question_id": question_id, "em": float(metrics["em"]), "f1": float(metrics["f1"]),
        "generation_time_ms": float(generation_time_ms), "pipeline_time_ms": float(pipeline_time_ms),
        "selected_hit": bool(any(answer_has_match_in_text(answer, passage) for answer in gold_answers for passage in selected_texts)),
        "final_context_K": 5, "selected_context_parity": True, "lattice_available": bool(lattice.spans),
        "lattice_span_count": lattice.span_count, "lattice_token_count": lattice.token_count,
        "energy_active_steps": int(active_steps), "generated_token_count": int(generated_tokens), "decoder_mode": decoder_mode,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase10c_reader_span_energy.yaml")
    parser.add_argument("--stage", choices=("screen", "formal", "replication"), default="screen")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _root()
    config_path = (args.config if args.config.is_absolute() else root / args.config).resolve()
    plan_path = root / "configs/experiments/phase10c_reader_span_energy_plan.json"
    contract = validate_contract(config_path, plan_path, args.stage)
    if args.validate_only:
        print(json.dumps(contract, sort_keys=True))
        return None
    phase, stage_spec = _load(config_path, args.stage)
    identity = _resolve_generator(root, phase["generator"], args.model_path)
    output_root = args.output_root or Path(phase["outputs"]["root"])
    output_root = output_root if output_root.is_absolute() else root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{timestamp}_screen"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_screen_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    plugin_files = (
        "applications/rag/reader_span_energy.py", "scripts/collab/five_ideas/phase10c_metrics.py",
        "scripts/collab/five_ideas/run_phase10c_reader_span_energy.py", "configs/experiments/phase10c_reader_span_energy.yaml",
        "configs/experiments/phase10c_reader_span_energy_plan.json",
    )
    metadata: dict[str, Any] = {
        "schema_version": 1, "phase": phase["name"], "stage": "screen", "status": "running", "diagnostic_only": True,
        "selection_mutation": False, "report_only": True, "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "plugin_plan": {"path": str(plan_path), "sha256": _sha256(plan_path), "tree_sha256": _tree_sha256(root, plugin_files)},
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current")},
        "environment": {"python_executable": sys.executable, "python_version": sys.version}, "generator": identity,
        "dataset": {"name": phase["dataset"]["name"], "split": phase["dataset"]["split"], **stage_spec},
        "retrieval": phase["retrieval"], "selection": phase["selection"], "reader_span_energy": phase["reader_span_energy"],
    }
    _write(run_dir / "run_metadata.json", metadata)
    sample_end = int(stage_spec["sample_offset"]) + int(stage_spec["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["split"], sample_end)[int(stage_spec["sample_offset"]):sample_end]
    if len(questions) != int(stage_spec["max_samples"]):
        raise Phase10CConfigError("fresh screen slice length mismatch")
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    corpus_manager.build(questions)
    scorer = make_answer_scorer(backend="dpr")
    generator = Generator(identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    arms: dict[str, dict[str, Any]] = {name: {"config": {"arm": name}, "samples": []} for name in EXPECTED_ARMS}
    started_total = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        question_id = str(item["id"])
        question = str(item["question"])
        gold_answers = [str(value) for value in item.get("answers", []) if str(value).strip()]
        shared_started = time.perf_counter()
        query_embedding = encoder.encode_queries([question])[0]
        _, embeddings, texts, _ = corpus_manager.retrieve_with_embeddings(query_embedding, int(phase["retrieval"]["initial_top_k"]))
        selected_ids, selected_texts = _selected_context(query_embedding, embeddings, texts, question, scorer, phase["selection"])
        if selected_ids != list(selected_ids) or len(selected_texts) != 5:
            raise Phase10CConfigError("selected-context order contract failed")
        frozen_prefix_ms = (time.perf_counter() - shared_started) * 1000.0

        baseline_started = time.perf_counter()
        baseline_prediction = generator.generate(question, selected_texts)
        baseline_generation_ms = (time.perf_counter() - baseline_started) * 1000.0
        baseline_lattice = SpanLattice(())

        lattice_started = time.perf_counter()
        selected_scores, hypotheses = scorer.score_passages_with_hypotheses(
            question, selected_texts, top_m=int(phase["reader_span_energy"]["top_m"]),
            max_answer_tokens=int(phase["reader_span_energy"]["max_answer_span_tokens"]),
        )
        reader_lattice = lattice_from_reader_hypotheses(hypotheses, selected_scores, generator.tokenizer, max_spans=int(phase["reader_span_energy"]["max_spans"]))
        control_lattice = matched_control_lattice(generator.tokenizer, selected_texts, reader_lattice, seed_key=question_id)
        if tuple(len(tokens) for tokens, _ in reader_lattice.spans) != tuple(len(tokens) for tokens, _ in control_lattice.spans):
            raise Phase10CConfigError("matched control shape contract failed")
        lattice_build_ms = (time.perf_counter() - lattice_started) * 1000.0

        reader_started = time.perf_counter()
        reader_decode = decode_with_span_energy(generator, question, selected_texts, reader_lattice)
        reader_generation_ms = (time.perf_counter() - reader_started) * 1000.0
        control_started = time.perf_counter()
        control_decode = decode_with_span_energy(generator, question, selected_texts, control_lattice)
        control_generation_ms = (time.perf_counter() - control_started) * 1000.0

        arms["baseline_greedy"]["samples"].append(_compact_sample(
            question_id, baseline_prediction, None, generator, gold_answers, selected_texts, generation_time_ms=baseline_generation_ms,
            pipeline_time_ms=frozen_prefix_ms + baseline_generation_ms, lattice=baseline_lattice, decoder_mode="frozen_generator",
        ))
        arms["reader_span_energy"]["samples"].append(_compact_sample(
            question_id, reader_decode.text, reader_decode, generator, gold_answers, selected_texts, generation_time_ms=reader_generation_ms,
            pipeline_time_ms=frozen_prefix_ms + lattice_build_ms + reader_generation_ms, lattice=reader_lattice, decoder_mode="reader_span_energy",
        ))
        arms["matched_span_control"]["samples"].append(_compact_sample(
            question_id, control_decode.text, control_decode, generator, gold_answers, selected_texts, generation_time_ms=control_generation_ms,
            pipeline_time_ms=frozen_prefix_ms + lattice_build_ms + control_generation_ms, lattice=control_lattice, decoder_mode="matched_span_control",
        ))
        if index % 5 == 0 or index == len(questions):
            print(f"  Phase 10C reader-span energy: {index}/{len(questions)}")

    result = {
        "schema_version": 1, "phase": phase["name"], "stage": "screen", "diagnostic_only": True, "selection_mutation": False,
        "report_only": True, "config": {"dataset": stage_spec, "retrieval": phase["retrieval"], "selection": phase["selection"],
        "reader_span_energy": {"plugin_id": phase["reader_span_energy"]["plugin_id"], "top_m": phase["reader_span_energy"]["top_m"],
        "max_answer_span_tokens": phase["reader_span_energy"]["max_answer_span_tokens"], "max_spans": phase["reader_span_energy"]["max_spans"],
        "energy_rule": phase["reader_span_energy"]["energy_rule"], "energy_coefficient": phase["reader_span_energy"]["energy_coefficient"], "final_context_K": 5}}, "arms": arms,
    }
    forbidden = _forbidden(result)
    if forbidden:
        raise Phase10CConfigError(f"compact result contains forbidden fields: {forbidden[:5]}")
    _write(run_dir / "result.json", result)
    gate = dict(phase["gate"]["screen"])
    if args.bootstrap_reps is not None:
        gate["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_phase10c(result, gate=gate)
    _write(run_dir / "summary.json", summary)
    metadata.update({"status": "completed", "timing_ms": {"total": (time.perf_counter() - started_total) * 1000.0}, "summary": {"path": str(run_dir / "summary.json"), "decision": summary["decision"]}})
    _write(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 10C screen: {run_dir}")
    print(f"Decision: {summary['decision']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (Phase10CConfigError, Phase10CError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
