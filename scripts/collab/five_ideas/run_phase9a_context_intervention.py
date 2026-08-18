#!/usr/bin/env python3
"""Run the collaborator-only Phase 9A context bottleneck diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

try:
    from scripts.collab.five_ideas.phase9a_metrics import (
        Phase9AError,
        summarize_context_intervention,
    )
except ImportError:  # pragma: no cover - direct script execution
    from phase9a_metrics import Phase9AError, summarize_context_intervention


class Phase9AConfigError(RuntimeError):
    """Raised when the frozen Phase 9A contract is malformed."""


EXPECTED_CONFIGS = (
    "qore_as_full",
    "qore_as_uniform_head_050",
    "qore_as_uniform_head_075",
    "qore_as_reader_window_050",
    "qore_as_reader_window_075",
)
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
FORBIDDEN_FIELDS = {"question", "passages", "gold_answers", "prediction", "raw_prompt"}


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase9AConfigError("cannot locate project root from the runner path")


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


def _plugin_tree_hash(root: Path) -> str:
    paths = (
        root / "applications/rag/context_transform.py",
        root / "scripts/rag/eval/eval_rag_refactored.py",
        root / "scripts/collab/five_ideas/phase9a_metrics.py",
        root / "scripts/collab/five_ideas/run_phase9a_context_intervention.py",
    )
    entries = [
        {"path": str(path.relative_to(root)), "sha256": _sha256_file(path)}
        for path in paths
    ]
    payload = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _append_args(command: list[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.extend([f"--{key}", str(value)])


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    phase = document.get("phase")
    if not isinstance(phase, dict) or int(phase.get("schema_version", -1)) != 1:
        raise Phase9AConfigError("configuration must contain phase.schema_version: 1")
    required = {
        "name", "schema_version", "diagnostic_only", "selection_mutation", "dataset",
        "split", "sample_offset", "max_samples", "corpus_mode", "wiki_dpr_config",
        "wiki_dpr_nprobe", "top_k_retrieval", "shared_args", "configurations", "gate",
        "max_new_tokens", "generator", "outputs",
    }
    missing = sorted(required - set(phase))
    if missing:
        raise Phase9AConfigError(f"configuration missing keys: {missing}")
    if phase["name"] != "phase9a_context_intervention":
        raise Phase9AConfigError("unexpected Phase 9A configuration name")
    if phase["diagnostic_only"] is not True or phase["selection_mutation"] is not False:
        raise Phase9AConfigError("Phase 9A must remain diagnostic_only with selection_mutation=false")
    if (phase["dataset"], phase["split"]) != ("nq_open", "validation"):
        raise Phase9AConfigError("Phase 9A is frozen to nq_open validation")
    if int(phase["sample_offset"]) != 200 or int(phase["max_samples"]) != 50:
        raise Phase9AConfigError("Phase 9A is frozen to validation questions 200-249")
    if phase["corpus_mode"] != "wiki_dpr" or phase["wiki_dpr_config"] != "psgs_w100.nq.compressed":
        raise Phase9AConfigError("Phase 9A requires the compressed Wiki-DPR corpus")
    if int(phase["wiki_dpr_nprobe"]) != 64 or int(phase["top_k_retrieval"]) != 50:
        raise Phase9AConfigError("Wiki-DPR retrieval settings are not frozen")
    shared = phase["shared_args"]
    if not isinstance(shared, dict):
        raise Phase9AConfigError("shared_args must be a mapping")
    frozen_shared = {
        "dataset": "nq_open", "split": "validation", "sample_offset": 200,
        "max_samples": 50, "corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed",
        "wiki_dpr_nprobe": 64, "top_k_retrieval": 50, "method": "qore", "K": 5,
        "seed": 42, "gamma": 1.0, "use_answer_scorer": True, "answer_scorer_backend": "dpr",
        "max_new_tokens": 32,
    }
    for key, expected in frozen_shared.items():
        if shared.get(key) != expected:
            raise Phase9AConfigError(f"shared_args.{key} is not frozen: {shared.get(key)!r}")
    configurations = phase["configurations"]
    if not isinstance(configurations, list) or [item.get("name") for item in configurations] != list(EXPECTED_CONFIGS):
        raise Phase9AConfigError(f"configuration order must be {EXPECTED_CONFIGS}")
    for item in configurations:
        if not isinstance(item, dict) or not isinstance(item.get("args"), dict):
            raise Phase9AConfigError("each configuration needs an args mapping")
        args = item["args"]
        if args.get("context_transform") not in {"full", "uniform_head", "reader_window"}:
            raise Phase9AConfigError(f"invalid context transform in {item.get('name')}")
        ratio = float(args.get("context_budget_ratio", -1))
        if not 0.0 < ratio <= 1.0:
            raise Phase9AConfigError(f"invalid context budget in {item.get('name')}")
        if item["name"] == "qore_as_full" and (args["context_transform"], ratio) != ("full", 1.0):
            raise Phase9AConfigError("full configuration must be baseline-equivalent")
        if item["name"] != "qore_as_full" and args["context_transform"] == "full":
            raise Phase9AConfigError("only qore_as_full may use full context")
    generator = phase["generator"]
    if not isinstance(generator, dict) or generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION:
        raise Phase9AConfigError("generator model ID/revision is not pinned")
    outputs = phase["outputs"]
    if not isinstance(outputs, dict) or outputs.get("compact_only") is not True:
        raise Phase9AConfigError("Phase 9A outputs must be compact_only")
    return phase


def _resolve_generator(root: Path, specification: Mapping[str, Any], override: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    for raw in specification.get("model_path_candidates", []):
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    candidates.extend([
        root / "models" / "llama3-8b",
        Path.home() / ".cache" / "huggingface" / "hub" / (
            "models--NousResearch--Meta-Llama-3-8B-Instruct"),
    ])
    cache_root = os.environ.get("HF_HOME")
    if cache_root:
        candidates.append(Path(cache_root) / "hub" / "models--NousResearch--Meta-Llama-3-8B-Instruct")

    attempted: list[str] = []
    for candidate in candidates:
        attempted.append(str(candidate))
        direct_config = candidate / "config.json"
        if direct_config.is_file():
            resolved = candidate.resolve()
        else:
            snapshot = candidate / "snapshots" / MODEL_REVISION
            if not (snapshot / "config.json").is_file():
                continue
            resolved = snapshot.resolve()
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_path": str(resolved),
            "config_path": str(resolved / "config.json"),
            "config_sha256": _sha256_file(resolved / "config.json"),
            "resolution": "cli_override" if override else "local_cache",
        }
    raise Phase9AConfigError(
        "reference generator not found in local candidates/cache; attempted: "
        + ", ".join(attempted)
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase9a_context_intervention.yaml")
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
        raise Phase9AConfigError("--bootstrap-reps must be at least 100")
    generator = _resolve_generator(root, phase["generator"], args.model_path)
    if args.validate_only:
        print(f"Validated Phase 9A config and resolved {generator['model_path']}")
        return None

    output_root_raw = args.output_root or phase["outputs"].get("root", "exchange/five_ideas/phase9a_context_intervention")
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
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")},
        "python": {"executable": sys.executable, "version": sys.version},
        "generator": generator,
        "plugins": {
            "allowlist": ["full_context_baseline", "uniform_head_context", "reader_window_context"],
            "tree_sha256": _plugin_tree_hash(root),
            "diagnostic_outputs_used_for_selection": False,
        },
        "dataset": {
            "name": phase["dataset"], "split": phase["split"],
            "sample_offset": phase["sample_offset"], "max_samples": phase["max_samples"],
            "corpus_mode": phase["corpus_mode"], "wiki_dpr_config": phase["wiki_dpr_config"],
        },
        "seeds": {"selection": phase["shared_args"]["seed"], "bootstrap": phase["gate"]["bootstrap_seed"]},
        "configurations": [],
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    results: dict[str, dict[str, Any]] = {}
    shared_args = dict(phase["shared_args"])
    for spec in phase["configurations"]:
        name = str(spec["name"])
        output_dir = run_dir / name
        output_dir.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-m", "scripts.rag.eval.eval_rag_refactored"]
        _append_args(command, shared_args)
        _append_args(command, spec["args"])
        command.extend([
            "--model_path", generator["model_path"],
            "--output_dir", str(output_dir), "--output_file", "result.json",
        ])
        (output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        started = time.perf_counter()
        with (output_dir / "log.txt").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
        entry = {
            "name": name,
            "command": command,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "returncode": completed.returncode,
            "result": str(output_dir / "result.json"),
        }
        metadata["configurations"].append(entry)
        _write_json(run_dir / "run_metadata.json", metadata)
        if completed.returncode != 0:
            raise Phase9AConfigError(f"configuration {name} failed; inspect {output_dir / 'log.txt'}")
        with (output_dir / "result.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        forbidden = _find_forbidden(payload)
        if forbidden:
            raise Phase9AConfigError(f"result {name} contains forbidden fields: {forbidden[:5]}")
        results[name] = payload

    gate = dict(phase["gate"])
    if args.bootstrap_reps is not None:
        gate["bootstrap_repetitions"] = args.bootstrap_reps
    summary = summarize_context_intervention(results, baseline_name="qore_as_full", gate=gate)
    _write_json(run_dir / "summary.json", summary)
    metadata["summary"] = {"path": str(run_dir / "summary.json"), "outcome": summary["outcome"]}
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed Phase 9A context diagnostic: {run_dir}")
    print(f"Outcome: {summary['outcome']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (Phase9AConfigError, Phase9AError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
