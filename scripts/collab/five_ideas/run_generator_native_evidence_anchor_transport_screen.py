#!/usr/bin/env python3
"""Run the authorized report-only screen for Generator-native residual transport."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents):
    if (candidate / "applications").is_dir() and (candidate / "configs").is_dir():
        PROJECT_ROOT = candidate
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:  # pragma: no cover
    raise SystemExit("ERROR: cannot locate project root")

from applications.rag.answer_scorer import make_answer_scorer  # noqa: E402
from applications.rag.data import load_dataset_for_rag, make_corpus_manager  # noqa: E402
from applications.rag.evaluation import evaluate_answer  # noqa: E402
from applications.rag.generation import Generator  # noqa: E402
from applications.rag.retrieval import make_encoder  # noqa: E402
from applications.rag.selector import select_passages  # noqa: E402
from applications.rag.generator_native_evidence_anchor_transport import (  # noqa: E402
    CANDIDATE_ID,
    FrozenDPRReaderSpanProvider,
    GeneratorNativeEvidenceAnchorObserver,
    PLUGIN_ORDER,
    PLUGIN_VERSION,
    find_forbidden_fields,
)
from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text  # noqa: E402


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
EXPECTED_SLICE = (1900, 50)
ARMS = ("baseline", "disabled", "reader", "control")
FORBIDDEN_FIELDS = {
    "question", "questions", "passage", "passages", "answer", "answers",
    "gold", "gold_answers", "prediction", "predictions", "text", "raw_prompt",
    "prompt_text", "span_text", "token_ids", "evaluator_trace",
}


class ScreenError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


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


def _tree_hash(paths: Sequence[Path]) -> str:
    entries = [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)} for path in paths]
    return hashlib.sha256(json.dumps(entries, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _resolve_generator(specification: Mapping[str, Any], override: str | None) -> dict[str, str]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    for raw in specification.get("model_path_candidates", []):
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)
    candidates.extend([
        PROJECT_ROOT / "models" / "llama3-8b",
        Path.home() / ".cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct",
    ])
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub/models--NousResearch--Meta-Llama-3-8B-Instruct")
    attempted: list[str] = []
    for candidate in candidates:
        attempted.append(str(candidate))
        direct = candidate if (candidate / "config.json").is_file() else candidate / "snapshots" / MODEL_REVISION
        if (direct / "config.json").is_file():
            return {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_path": str(direct.resolve()),
                "config_sha256": _sha256(direct / "config.json"),
                "resolution": "cli_override" if override else "local_cache",
            }
    raise ScreenError("reference generator not found; attempted: " + ", ".join(attempted))


def _load_config(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ScreenError(f"cannot read screen config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != CANDIDATE_ID + "_screen":
        raise ScreenError("unexpected screen configuration")
    if phase.get("authorization") != "user_authorized_report_only_screen":
        raise ScreenError("screen authorization is not frozen")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise ScreenError("screen must remain diagnostic-only")
    dataset = phase.get("dataset", {})
    if (dataset.get("name"), dataset.get("split"), int(dataset.get("sample_offset", -1)), int(dataset.get("max_samples", -1)), dataset.get("fresh_slice")) != ("nq_open", "validation", *EXPECTED_SLICE, True):
        raise ScreenError("screen slice is not frozen")
    if phase.get("retrieval") != {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "top_k": 50}:
        raise ScreenError("retrieval contract is not frozen")
    expected_selection = {
        "method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42,
        "gamma": 1.0, "delta": 0.0, "complementarity_method": None,
        "qore_prefilter_size": None, "direct_solve_max_n": 20,
        "lambda_mmr": 0.7, "saturation_alpha": 1.0, "lambda_submodular": 0.5,
        "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8,
        "use_answer_scorer": True, "answer_scorer_backend": "dpr",
    }
    if phase.get("selection") != expected_selection:
        raise ScreenError("selection contract is not frozen")
    generator = phase.get("generator", {})
    if generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise ScreenError("generator contract is not frozen")
    if tuple(phase.get("arms", ())) != ARMS:
        raise ScreenError("arm order is not frozen")
    transport = phase.get("transport", {})
    if transport.get("candidate_id") != CANDIDATE_ID or transport.get("plugin_version") != PLUGIN_VERSION or tuple(transport.get("plugin_allowlist", ())) != PLUGIN_ORDER or tuple(transport.get("active_arms", ())) != ("disabled", "reader", "control") or transport.get("diagnostic_outputs_used_for_selection") is not False or transport.get("reader_scores_used_for_selection") is not False or transport.get("production_generator_source_modified") is not False:
        raise ScreenError("transport contract is not frozen")
    gate = phase.get("gate", {})
    if float(gate.get("reader_control_f1_delta_min", 1.0)) != 0.0 or float(gate.get("positive_reader_control_f1_delta_min", -1.0)) != 0.02 or float(gate.get("reader_baseline_harm_ci95_low_min", -1.0)) != -0.05 or float(gate.get("max_pipeline_cost_ratio", -1.0)) != 2.0:
        raise ScreenError("screen gate is not frozen")
    outputs = phase.get("outputs", {})
    if outputs.get("compact_only") is not True or outputs.get("root") != "exchange/five_ideas/generator_native_evidence_anchor_transport":
        raise ScreenError("output contract is not frozen")
    return phase


def _selected_context(question: str, gold_answers: Sequence[str], encoder: Any, corpus: Any, scorer: Any, phase: Mapping[str, Any]) -> tuple[list[str], bool]:
    query_embedding = encoder.encode_queries([question])[0]
    _, embeddings, texts, _ = corpus.retrieve_with_embeddings(query_embedding, int(phase["retrieval"]["top_k"]))
    if len(texts) != int(phase["retrieval"]["top_k"]):
        raise ScreenError("incomplete retrieval")
    answer_scores = scorer.score_passages(question, list(texts))
    selection = phase["selection"]
    selected = select_passages(
        query_embedding, embeddings, K=int(selection["K"]), method="qore",
        num_reads=int(selection["num_reads"]), lam=float(selection["lam"]),
        gamma=float(selection["gamma"]), delta=float(selection["delta"]),
        complementarity_method=selection["complementarity_method"],
        qore_prefilter_size=selection["qore_prefilter_size"],
        direct_solve_max_n=int(selection["direct_solve_max_n"]),
        lambda_mmr=float(selection["lambda_mmr"]),
        saturation_alpha=float(selection["saturation_alpha"]),
        lambda_submodular=float(selection["lambda_submodular"]),
        dpp_quality_scale=float(selection["dpp_quality_scale"]),
        dpp_jitter=float(selection["dpp_jitter"]),
        answer_scorer=scorer, passage_texts=list(texts), question=question,
        seed=int(selection["seed"]), relevance_scores=answer_scores,
    )
    selected_ids = [int(index) for index in selected]
    if len(selected_ids) != 5 or len(set(selected_ids)) != 5:
        raise ScreenError("QORE did not return exactly five unique passages")
    selected_texts = [str(texts[index]) for index in selected_ids]
    retrieval_hit = bool(any(answer_has_match_in_text(answer, passage) for answer in gold_answers for passage in texts))
    return selected_texts, retrieval_hit


def _metric(prediction: str, gold_answers: Sequence[str], selected_texts: Sequence[str], elapsed_ms: float, pipeline_ms: float) -> dict[str, Any]:
    scores = evaluate_answer(str(prediction), list(gold_answers))
    return {
        "em": float(scores["em"]),
        "f1": float(scores["f1"]),
        "generation_time_ms": float(elapsed_ms),
        "pipeline_time_ms": float(pipeline_ms),
        "selected_hit": bool(any(answer_has_match_in_text(answer, passage) for answer in gold_answers for passage in selected_texts)),
    }


def _bootstrap(values: Sequence[float], repetitions: int, seed: int) -> dict[str, float | int]:
    if not values:
        raise ScreenError("cannot bootstrap empty values")
    rng = random.Random(seed)
    draws = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(repetitions)]
    ordered = sorted(float(value) for value in draws)
    def percentile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"mean": float(statistics.fmean(values)), "ci95_low": percentile(0.025), "ci95_high": percentile(0.975), "bootstrap_repetitions": repetitions, "bootstrap_seed": seed}


def _summarize(result: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    samples = result.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SLICE[1]:
        raise ScreenError("result sample count is malformed")
    ids = [str(sample.get("question_id")) for sample in samples]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ScreenError("question IDs are not unique")
    arms = {arm: [sample["arms"][arm] for sample in samples] for arm in ARMS}
    for arm in ARMS:
        if any(set(row) != {"em", "f1", "generation_time_ms", "pipeline_time_ms", "selected_hit", "diagnostic"} for row in arms[arm]):
            raise ScreenError(f"{arm} compact arm schema mismatch")
    baseline = arms["baseline"]
    def delta(arm: str, metric: str, seed_offset: int) -> dict[str, Any]:
        values = [float(arms[arm][i][metric]) - float(baseline[i][metric]) for i in range(len(samples))]
        return _bootstrap(values, int(gate["bootstrap_repetitions"]), int(gate["bootstrap_seed"]) + seed_offset)
    reader_control = {
        metric: _bootstrap(
            [float(arms["reader"][i][metric]) - float(arms["control"][i][metric]) for i in range(len(samples))],
            int(gate["bootstrap_repetitions"]), int(gate["bootstrap_seed"]) + 20 + offset,
        )
        for offset, metric in enumerate(("em", "f1"))
    }
    reader = {"em": delta("reader", "em", 0), "f1": delta("reader", "f1", 1)}
    control = {"em": delta("control", "em", 2), "f1": delta("control", "f1", 3)}
    reader_coverage = statistics.fmean(1.0 if row["diagnostic"]["provider"]["located_count"] > 0 and not row["diagnostic"]["decision"]["fallback"] else 0.0 for row in arms["reader"])
    reader_pipeline = sum(float(row["pipeline_time_ms"]) for row in arms["reader"])
    baseline_pipeline = sum(float(row["pipeline_time_ms"]) for row in baseline)
    cost_ratio = reader_pipeline / baseline_pipeline if baseline_pipeline else float("inf")
    gates = {
        "all_questions_accounted": len(samples) == EXPECTED_SLICE[1],
        "reader_span_coverage": reader_coverage >= float(gate["minimum_reader_span_coverage"]),
        "reader_baseline_harm": reader["f1"]["ci95_low"] >= float(gate["reader_baseline_harm_ci95_low_min"]),
        "reader_control_nonnegative": reader_control["f1"]["mean"] >= float(gate["reader_control_f1_delta_min"]),
        "pipeline_cost": cost_ratio <= float(gate["max_pipeline_cost_ratio"]),
    }
    fatal = not gates["all_questions_accounted"] or not gates["pipeline_cost"] or reader["f1"]["ci95_low"] < -0.20
    positive_signal = all((gates["reader_span_coverage"], gates["reader_baseline_harm"], reader_control["f1"]["mean"] >= float(gate["positive_reader_control_f1_delta_min"]), reader_control["f1"]["ci95_low"] > 0.0))
    decision = "screen_positive_signal" if positive_signal and not fatal else ("screen_contract_or_harm_failure" if fatal else "screen_inconclusive")
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "stage": "screen",
        "current_tier": "L0_diagnostic",
        "claim_ceiling": "diagnostic; no L1/L2 claim from this screen",
        "decision": decision,
        "n_questions": len(samples),
        "reader_span_coverage": float(reader_coverage),
        "pipeline_cost_ratio_reader_vs_baseline": float(cost_ratio),
        "reader_vs_baseline": reader,
        "control_vs_baseline": control,
        "reader_minus_control": reader_control,
        "gates": gates,
        "next_step": "pre-register full-data single-seed run only if user authorizes after inspecting this screen; otherwise close as inconclusive",
    }


def run(args: argparse.Namespace) -> Path | None:
    config_path = (args.config if args.config.is_absolute() else PROJECT_ROOT / args.config).resolve()
    phase = _load_config(config_path)
    identity = _resolve_generator(phase["generator"], args.model_path)
    if args.validate_only:
        print(json.dumps({"status": "valid", "phase": phase["name"], "slice": phase["dataset"]["slice"], "arms": list(ARMS), "model_id": identity["model_id"], "selection_mutation": False, "report_only": True}, sort_keys=True))
        return None
    output_root = Path(args.output_root or phase["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{timestamp}_screen"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_screen_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    tree_files = (
        PROJECT_ROOT / "applications/rag/generator_native_evidence_anchor_transport.py",
        PROJECT_ROOT / "applications/rag/generation/generator.py",
        PROJECT_ROOT / "applications/rag/selector.py",
        PROJECT_ROOT / "applications/rag/answer_scorer.py",
        SCRIPT_PATH,
        config_path,
    )
    metadata = {
        "schema_version": 1, "phase": phase["name"], "stage": "screen", "status": "running",
        "diagnostic_only": True, "selection_mutation": False, "report_only": True,
        "candidate_id": CANDIDATE_ID, "plugin_version": PLUGIN_VERSION,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current")},
        "python": {"executable": sys.executable, "version": sys.version},
        "generator": identity,
        "dataset": phase["dataset"], "retrieval": phase["retrieval"], "selection": phase["selection"],
        "transport": {"plugin_allowlist": list(PLUGIN_ORDER), "diagnostic_outputs_used_for_selection": False, "production_generator_source_modified": False, "tree_sha256": _tree_hash(tree_files)},
    }
    _write(run_dir / "run_metadata.json", metadata)
    requested = int(phase["dataset"]["sample_offset"]) + int(phase["dataset"]["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["split"], requested)[int(phase["dataset"]["sample_offset"]):requested]
    if len(questions) != int(phase["dataset"]["max_samples"]):
        raise ScreenError("fresh screen slice length mismatch")
    np.random.seed(int(phase["selection"]["seed"]))
    encoder = make_encoder("dpr")
    corpus = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
    corpus.build(questions)
    scorer = make_answer_scorer(backend="dpr")
    generator = Generator(identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    observer = GeneratorNativeEvidenceAnchorObserver(generator, FrozenDPRReaderSpanProvider(scorer))
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        question = str(item["question"])
        question_id = str(item["id"])
        gold_answers = [str(value) for value in item.get("answers", []) if str(value).strip()]
        if not gold_answers:
            raise ScreenError(f"{question_id} has no non-empty answers")
        selection_started = time.perf_counter()
        selected_texts, retrieval_hit = _selected_context(question, gold_answers, encoder, corpus, scorer, phase)
        selection_ms = (time.perf_counter() - selection_started) * 1000.0
        selected_hit = bool(any(answer_has_match_in_text(answer, passage) for answer in gold_answers for passage in selected_texts))
        arms: dict[str, dict[str, Any]] = {}
        baseline_started = time.perf_counter()
        baseline_prediction = generator.generate(question, selected_texts)
        baseline_ms = (time.perf_counter() - baseline_started) * 1000.0
        arms["baseline"] = _metric(baseline_prediction, gold_answers, selected_texts, baseline_ms, selection_ms + baseline_ms)
        arms["baseline"]["diagnostic"] = {"mode": "frozen_generator", "fallback": False}
        for arm_name, requested_arm in (("disabled", "disabled"), ("reader", "reader"), ("control", "control")):
            arm_started = time.perf_counter()
            observation = observer.generate(question, selected_texts, requested_arm=requested_arm)
            elapsed_ms = (time.perf_counter() - arm_started) * 1000.0
            arms[arm_name] = _metric(observation.text, gold_answers, selected_texts, elapsed_ms, selection_ms + elapsed_ms)
            arms[arm_name]["diagnostic"] = observation.compact()
        for arm in ARMS:
            arms[arm]["selected_hit"] = selected_hit
        samples.append({"question_id": question_id, "retrieval_hit": retrieval_hit, "selected_hit": selected_hit, "selected_context_K": 5, "selected_context_parity": True, "selection_time_ms": selection_ms, "arms": arms})
        if index % 5 == 0 or index == len(questions):
            print(f"  Generator-native transport screen: {index}/{len(questions)}")
    result = {
        "schema_version": 1, "phase": phase["name"], "stage": "screen", "diagnostic_only": True,
        "selection_mutation": False, "report_only": True,
        "config": {"dataset": phase["dataset"], "retrieval": phase["retrieval"], "selection": phase["selection"], "arms": list(ARMS), "transport": {"candidate_id": CANDIDATE_ID, "plugin_version": PLUGIN_VERSION, "alpha_rule": "1/sqrt(num_hidden_layers)", "layer": 30, "diagnostic_outputs_used_for_selection": False}},
        "samples": samples,
    }
    forbidden = _forbidden(result)
    if forbidden:
        raise ScreenError(f"compact result contains forbidden fields: {forbidden[:5]}")
    _write(run_dir / "result.json", result)
    size = (run_dir / "result.json").stat().st_size
    if size > 1_048_576:
        raise ScreenError("compact result exceeds 1 MiB publication threshold")
    summary = _summarize(result, phase["gate"])
    _write(run_dir / "summary.json", summary)
    metadata.update({"status": "completed", "result_bytes": size, "result_sha256": _sha256(run_dir / "result.json"), "timing_ms": {"total": (time.perf_counter() - started) * 1000.0}, "summary": {"path": str(run_dir / "summary.json"), "decision": summary["decision"]}})
    _write(run_dir / "run_metadata.json", metadata)
    print(f"Completed Generator-native transport screen: {run_dir}")
    print(f"Decision: {summary['decision']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/generator_native_evidence_anchor_transport_screen.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except (ScreenError, OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
