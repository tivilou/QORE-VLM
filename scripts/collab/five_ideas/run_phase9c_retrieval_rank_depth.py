#!/usr/bin/env python3
"""Run the collaborator-only Phase 9C retrieval rank-depth diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

# Keep direct execution equivalent to the repository wrapper's PYTHONPATH setup.
_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PROJECT_ROOT))

from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.retrieval import make_encoder
from scripts.rag.eval.eval_rag_refactored import answer_has_match_in_text

try:
    from scripts.collab.five_ideas.retrieval_rank_depth_metrics import (
        EXPECTED_CUTOFFS,
        RankDepthError,
        summarize_rank_depth,
    )
except ImportError:  # pragma: no cover - direct script execution
    from retrieval_rank_depth_metrics import EXPECTED_CUTOFFS, RankDepthError, summarize_rank_depth


class RankDepthConfigError(RuntimeError):
    """Raised when the frozen Phase 9C contract is malformed."""


FORBIDDEN_FIELDS = {"question", "passages", "gold_answers", "prediction", "raw_prompt"}


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise RankDepthConfigError("cannot locate project root from the runner path")


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


def _find_forbidden(payload: Any, path: str = "$root") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = f"{path}.{key}"
            if str(key) in FORBIDDEN_FIELDS:
                findings.append(child)
            findings.extend(_find_forbidden(value, child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_find_forbidden(value, f"{path}[{index}]"))
    return findings


def _plugin_tree_hash(root: Path) -> str:
    paths = (
        root / "scripts/rag/eval/eval_rag_refactored.py",
        root / "scripts/collab/five_ideas/retrieval_rank_depth_metrics.py",
        root / "scripts/collab/five_ideas/run_phase9c_retrieval_rank_depth.py",
    )
    entries = [
        {"path": str(path.relative_to(root)), "sha256": _sha256_file(path)}
        for path in paths
    ]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or int(phase.get("schema_version", -1)) != 1:
        raise RankDepthConfigError("configuration must contain phase.schema_version: 1")
    required = {
        "name", "schema_version", "diagnostic_only", "selection_mutation", "generation",
        "answer_scorer", "dataset", "split", "sample_offset", "max_samples", "fresh_slice",
        "corpus_mode", "wiki_dpr_config", "wiki_dpr_nprobe", "top_k_retrieval", "cutoffs", "seed",
        "encoder", "gate", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise RankDepthConfigError(f"configuration missing keys: {missing}")
    if phase["name"] != "phase9c_retrieval_rank_depth":
        raise RankDepthConfigError("unexpected Phase 9C configuration name")
    if any(phase[key] is not False for key in ("selection_mutation", "generation", "answer_scorer")):
        raise RankDepthConfigError("Phase 9C must disable selection mutation, generation, and Answer Scorer")
    if phase["diagnostic_only"] is not True or phase["fresh_slice"] is not True:
        raise RankDepthConfigError("Phase 9C must be diagnostic-only on a fresh slice")
    if (phase["dataset"], phase["split"]) != ("nq_open", "validation"):
        raise RankDepthConfigError("Phase 9C is frozen to nq_open validation")
    if (int(phase["sample_offset"]), int(phase["max_samples"])) != (250, 200):
        raise RankDepthConfigError("Phase 9C is frozen to validation questions 250-449")
    if phase["corpus_mode"] != "wiki_dpr" or phase["wiki_dpr_config"] != "psgs_w100.nq.compressed":
        raise RankDepthConfigError("Phase 9C requires the compressed Wiki-DPR corpus")
    if int(phase["wiki_dpr_nprobe"]) != 64 or int(phase["top_k_retrieval"]) != 200:
        raise RankDepthConfigError("Phase 9C retrieval settings are not frozen")
    if [int(value) for value in phase["cutoffs"]] != list(EXPECTED_CUTOFFS):
        raise RankDepthConfigError("Phase 9C cutoffs must be [50, 100, 200]")
    if int(phase["seed"]) != 42:
        raise RankDepthConfigError("Phase 9C seed is frozen to 42")
    encoder = phase["encoder"]
    if not isinstance(encoder, dict) or encoder.get("type") != "dpr":
        raise RankDepthConfigError("Phase 9C requires the DPR query encoder")
    if not encoder.get("query_model") or not encoder.get("passage_model"):
        raise RankDepthConfigError("encoder model IDs must be recorded")
    gate = phase["gate"]
    if not isinstance(gate, dict):
        raise RankDepthConfigError("gate must be a mapping")
    for key in (
        "maximum_top50_failure_rate",
        "minimum_top200_gain_over_top50",
        "maximum_top200_failure_rate",
    ):
        value = float(gate.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise RankDepthConfigError(f"gate.{key} must be in [0,1]")
    outputs = phase["outputs"]
    if not isinstance(outputs, dict) or outputs.get("compact_only") is not True:
        raise RankDepthConfigError("Phase 9C outputs must be compact_only")
    return phase


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/experiments/phase9c_retrieval_rank_depth.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config_path = config_path.resolve()
    phase = _load_config(config_path)
    if args.validate_only:
        print(json.dumps({
            "status": "valid",
            "phase": phase["name"],
            "dataset": phase["dataset"],
            "sample_offset": phase["sample_offset"],
            "max_samples": phase["max_samples"],
            "top_k_retrieval": phase["top_k_retrieval"],
            "cutoffs": phase["cutoffs"],
            "seed": phase["seed"],
            "selection_mutation": False,
            "generation": False,
            "answer_scorer": False,
        }, sort_keys=True))
        return None

    output_root_raw = args.output_root or phase["outputs"].get(
        "root", "exchange/five_ideas/phase9c_retrieval_rank_depth"
    )
    output_root = Path(output_root_raw)
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
        "diagnostic_only": True,
        "selection_mutation": False,
        "generation": False,
        "answer_scorer": False,
        "status": "running",
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "status": _git(root, "status", "--short", "--branch"),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "dataset": {
            "name": phase["dataset"],
            "split": phase["split"],
            "sample_offset": phase["sample_offset"],
            "max_samples": phase["max_samples"],
            "fresh_slice": phase["fresh_slice"],
        },
        "retrieval": {
            "corpus_mode": phase["corpus_mode"],
            "wiki_dpr_config": phase["wiki_dpr_config"],
            "nprobe": phase["wiki_dpr_nprobe"],
            "top_k_retrieval": phase["top_k_retrieval"],
            "cutoffs": phase["cutoffs"],
            "seed": phase["seed"],
        },
        "encoder": phase["encoder"],
        "matcher": "scripts.rag.eval.eval_rag_refactored.answer_has_match_in_text",
        "plugins": {
            "allowlist": ["top200_retrieval_observer", "rank_depth_gate"],
            "tree_sha256": _plugin_tree_hash(root),
            "diagnostic_outputs_used_for_selection": False,
        },
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path)]
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    requested = int(phase["sample_offset"]) + int(phase["max_samples"])
    questions = load_dataset_for_rag(phase["dataset"], phase["split"], requested)
    questions = questions[int(phase["sample_offset"]):requested]
    if len(questions) != int(phase["max_samples"]):
        raise RankDepthConfigError(
            f"held-out slice returned {len(questions)} questions, expected {phase['max_samples']}"
        )

    encoder = make_encoder(
        phase["encoder"]["type"],
        query_encoder=phase["encoder"]["query_model"],
        passage_encoder=phase["encoder"]["passage_model"],
    )
    corpus_manager = make_corpus_manager(
        phase["corpus_mode"],
        {
            "wiki_dpr_config": phase["wiki_dpr_config"],
            "nprobe": int(phase["wiki_dpr_nprobe"]),
        },
    )
    corpus_manager.build(questions)
    top_k = int(phase["top_k_retrieval"])
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        query_embedding = encoder.encode_queries([item["question"]])[0]
        _, _, passages, _ = corpus_manager.retrieve_with_embeddings(query_embedding, top_k)
        if len(passages) != top_k:
            raise RankDepthConfigError(f"retrieval returned {len(passages)} passages, expected {top_k}")
        answers = [str(answer) for answer in item.get("answers", []) if str(answer).strip()]
        if not answers:
            raise RankDepthConfigError(f"question {item['id']} has no non-empty gold answer")
        first_rank: int | None = None
        for rank, passage in enumerate(passages, start=1):
            if any(answer_has_match_in_text(answer, passage) for answer in answers):
                first_rank = rank
                break
        samples.append({
            "question_id": str(item["id"]),
            "first_answer_rank": first_rank,
            **{
                f"answer_hit_at_{cutoff}": first_rank is not None and first_rank <= cutoff
                for cutoff in EXPECTED_CUTOFFS
            },
        })
        if index % 25 == 0 or index == len(questions):
            print(f"  Retrieval rank depth: {index}/{len(questions)}")

    result = {
        "schema_version": 1,
        "phase": phase["name"],
        "diagnostic_only": True,
        "selection_mutation": False,
        "generation": False,
        "answer_scorer": False,
        "config": {
            "dataset": phase["dataset"],
            "split": phase["split"],
            "sample_offset": phase["sample_offset"],
            "max_samples": phase["max_samples"],
            "corpus_mode": phase["corpus_mode"],
            "wiki_dpr_config": phase["wiki_dpr_config"],
            "wiki_dpr_nprobe": phase["wiki_dpr_nprobe"],
            "top_k_retrieval": phase["top_k_retrieval"],
            "cutoffs": phase["cutoffs"],
            "seed": phase["seed"],
            "encoder_type": phase["encoder"]["type"],
        },
        "samples": samples,
    }
    forbidden = _find_forbidden(result)
    if forbidden:
        raise RankDepthConfigError(f"result contains forbidden fields: {forbidden[:5]}")
    _write_json(run_dir / "result.json", result)
    summary = summarize_rank_depth(result, gate=phase["gate"])
    _write_json(run_dir / "summary.json", summary)
    metadata.update({
        "status": "completed",
        "timing_ms": {"retrieval_and_matching": (time.perf_counter() - started) * 1000.0},
        "summary": {
            "path": str(run_dir / "summary.json"),
            "n_questions": summary["n_questions"],
            "primary_bottleneck": summary["decision"]["primary_bottleneck"],
        },
    })
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 9C retrieval rank-depth diagnostic: {run_dir}")
    print(f"Primary bottleneck: {summary['decision']['primary_bottleneck']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (RankDepthConfigError, RankDepthError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
