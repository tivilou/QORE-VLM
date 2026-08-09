#!/usr/bin/env python3
"""Run a configuration-driven, matched five-idea development gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gate_manifest import ManifestError, load_manifest


class GateError(RuntimeError):
    """Raised when a gate cannot be run safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=project_root, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def append_cli_args(command: list[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=Path("configs/experiments/five_idea_development_gate.yaml"),
        help="YAML gate manifest (default: five_idea_development_gate.yaml)",
    )
    parser.add_argument("--smoke", action="store_true", help="Use the 20-question smoke manifest")
    parser.add_argument("--max-samples", type=int, help="Override gate.shared_args.max_samples")
    parser.add_argument("--seed", type=int, help="Override gate.shared_args.seed")
    parser.add_argument("--output-root", type=Path, default=Path("exchange/five_ideas/development_gate"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--max-enumerations", type=int, default=100000)
    parser.add_argument("--max-artifact-mb", type=int, default=10)
    return parser.parse_args(argv)


def run_gate(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    manifest_path = args.gate_config
    if args.smoke and manifest_path == Path("configs/experiments/five_idea_development_gate.yaml"):
        manifest_path = Path("configs/experiments/five_idea_development_gate_smoke.yaml")
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    shared_args = dict(manifest["gate"]["shared_args"])
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise GateError("--max-samples must be positive")
        shared_args["max_samples"] = args.max_samples
    if args.seed is not None:
        if args.seed < 0:
            raise GateError("--seed must be non-negative")
        shared_args["seed"] = args.seed
    if args.skip_generation:
        shared_args["skip_generation"] = True
    if args.bootstrap_reps < 100 or args.max_enumerations < 1 or args.max_artifact_mb < 1:
        raise GateError("bootstrap/artifact limits are invalid")
    python_bin = Path(args.python_bin)
    if not python_bin.exists() or not os.access(python_bin, os.X_OK):
        raise GateError(f"Python executable not found or not executable: {args.python_bin}")

    if not args.allow_dirty:
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=project_root).returncode != 0
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=project_root).returncode != 0
        if dirty or staged:
            raise GateError("tracked worktree changes detected; use --allow-dirty for a debug run")

    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output_root = output_root.resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "gate": manifest["gate"]["name"],
        "gate_config": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "overrides": {
            "max_samples": args.max_samples,
            "seed": args.seed,
            "skip_generation": args.skip_generation,
        },
        "python": {"executable": str(python_bin.resolve()), "version": sys.version},
        "git": {
            "commit": git_output(project_root, "rev-parse", "HEAD"),
            "status": git_output(project_root, "status", "--short", "--branch"),
        },
        "configurations": [],
    }
    (run_dir / "gate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    evaluator_module = "scripts.rag.eval.eval_rag_refactored"
    for spec in manifest["gate"]["configurations"]:
        name = spec["name"]
        output_dir = run_dir / name
        output_dir.mkdir(parents=True, exist_ok=False)
        command = [str(python_bin), "-m", evaluator_module]
        append_cli_args(command, shared_args)
        append_cli_args(command, spec["args"])
        command.extend(["--output_dir", str(output_dir), "--output_file", "result.json"])
        (output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        started = time.perf_counter()
        with (output_dir / "log.txt").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=project_root, stdout=log, stderr=subprocess.STDOUT, text=True
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        entry = {
            "name": name,
            "kind": spec["kind"],
            "command": command,
            "elapsed_ms": elapsed_ms,
            "returncode": completed.returncode,
            "result": str(output_dir / "result.json"),
        }
        metadata["configurations"].append(entry)
        entry["status"] = "completed" if completed.returncode == 0 else "failed"
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            raise GateError(f"configuration {name} failed; inspect {output_dir / 'log.txt'}")

    analysis_command = [
        str(python_bin),
        str(project_root / "scripts/collab/five_ideas/analyze_diagnostics_pilot.py"),
        str(run_dir),
        "--gate-config",
        str(manifest_path),
        "--bootstrap-reps",
        str(args.bootstrap_reps),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--max-enumerations",
        str(args.max_enumerations),
        "--max-artifact-mb",
        str(args.max_artifact_mb),
    ]
    started = time.perf_counter()
    completed = subprocess.run(analysis_command, cwd=project_root, text=True)
    metadata["analysis"] = {
        "command": analysis_command,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "returncode": completed.returncode,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise GateError("analysis failed")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run_dir = run_gate(args)
        print(f"Completed development gate: {run_dir}")
        return 0
    except (GateError, ManifestError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
