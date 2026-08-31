#!/usr/bin/env python3
"""Strict full-population DPR positive-evidence bottleneck audit.

The runner is observation-only.  Official DPR contexts only construct an
offline canonical-ID observer; they never reach retrieval, QORE, prompting,
the Generator, or answer evaluation.  The full population is first accounted
without loading Wiki-DPR or a model, then the strictly mapped eligible subset
is observed through the frozen normal RAG path.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import yaml

_SCRIPT_PATH = Path(__file__).resolve()
for _candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_candidate / "configs").is_dir() and (_candidate / "applications").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from applications.rag.dpr_positive_gold_alignment import (
    DprQuestionEvidence,
    OfficialPositiveContext,
    align_dpr_positives_to_wiki,
    normalized_question_identity,
    passage_identity,
)
from applications.rag.dpr_positive_gold_eligible_universe import (
    make_compact_row,
    summarize_universe,
    validate_universe_result,
)


PHASE_NAME = "dpr_positive_gold_eligible_universe_audit"
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
ALL_QUESTIONS = 3610
OFFICIAL_NONEMPTY_CONTEXT_QUESTIONS = 1868
RESULT_GIT_MAX_BYTES = 1_048_576
EXPECTED_PLUGINS = (
    "dpr_question_identity_join",
    "official_dpr_eligible_population_accounting",
    "wiki_dpr_identity_alignment",
    "frozen_retrieval_qore_generator_observer",
    "eligible_universe_bottleneck_combiner",
)
EXPECTED_SELECTION = {
    "method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42,
    "gamma": 1.0, "delta": 0.0, "complementarity_method": None,
    "qore_prefilter_size": None, "direct_solve_max_n": 20,
    "lambda_mmr": 0.7, "saturation_alpha": 1.0, "lambda_submodular": 0.5,
    "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8,
    "use_answer_scorer": True, "answer_scorer_backend": "dpr",
}


class DprPositiveAlignmentConfigError(RuntimeError):
    """Raised when the frozen eligible-universe audit contract is invalid."""


def _project_root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise DprPositiveAlignmentConfigError("cannot locate project root")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project_relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _plugin_tree_hash(root: Path) -> str:
    paths = (
        root / "applications/rag/dpr_positive_gold_alignment.py",
        root / "applications/rag/dpr_positive_gold_eligible_universe.py",
        root / "scripts/collab/five_ideas/run_dpr_positive_gold_eligible_universe_audit.py",
    )
    entries = [
        {"path": _project_relative(root, path), "sha256": _sha256_file(path)}
        for path in paths
    ]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise DprPositiveAlignmentConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, Mapping) or phase.get("name") != PHASE_NAME:
        raise DprPositiveAlignmentConfigError("unexpected eligible-universe configuration")
    if int(phase.get("schema_version", -1)) != 1:
        raise DprPositiveAlignmentConfigError("eligible-universe config must be schema 1")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise DprPositiveAlignmentConfigError("audit must remain diagnostic-only and selection-frozen")
    expected_dataset = {
        "name": "nq_open", "split": "validation", "max_samples": ALL_QUESTIONS,
        "fresh_population": True,
    }
    if dict(phase.get("dataset") or {}) != expected_dataset:
        raise DprPositiveAlignmentConfigError("full NQ-Open validation population is frozen")
    if phase.get("retrieval") != {
        "corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64, "top_k": 50,
    }:
        raise DprPositiveAlignmentConfigError("retrieval contract is not frozen")
    if dict(phase.get("selection") or {}) != EXPECTED_SELECTION:
        raise DprPositiveAlignmentConfigError("QORE + DPR Answer Scorer contract is not frozen")
    alignment = phase.get("alignment")
    if not isinstance(alignment, Mapping) or alignment.get("scan_mode") != "full_corpus":
        raise DprPositiveAlignmentConfigError("full Wiki-DPR identity scan is required")
    if alignment.get("eligible_context_rule") != "official_context_is_nonempty_string":
        raise DprPositiveAlignmentConfigError("eligible universe must use non-empty official context only")
    if not isinstance(alignment.get("dpr_positive_candidates"), list):
        raise DprPositiveAlignmentConfigError("official DPR source candidates are missing")
    download = alignment.get("official_dpr_download")
    if not isinstance(download, Mapping) or not all(
        isinstance(download.get(key), str) and download[key].strip()
        for key in ("source_name", "url", "cache_path")
    ):
        raise DprPositiveAlignmentConfigError("official DPR source is not frozen")
    generator = phase.get("generator")
    if not isinstance(generator, Mapping) or generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION:
        raise DprPositiveAlignmentConfigError("Generator identity is not frozen")
    if int(generator.get("max_new_tokens", -1)) != 32:
        raise DprPositiveAlignmentConfigError("Generator decoding budget is not frozen")
    if tuple((phase.get("plugins") or {}).get("allowlist", ())) != EXPECTED_PLUGINS:
        raise DprPositiveAlignmentConfigError("plugin allowlist/order is not frozen")
    gate = phase.get("gate")
    if not isinstance(gate, Mapping) or float(gate.get("minimum_mapping_rate", -1)) != 0.80:
        raise DprPositiveAlignmentConfigError("strict mapping gate must remain 0.80")
    expected_accounting = {
        "all_questions": ALL_QUESTIONS,
        "official_nonempty_context_questions": OFFICIAL_NONEMPTY_CONTEXT_QUESTIONS,
    }
    if dict(phase.get("expected_source_accounting") or {}) != expected_accounting:
        raise DprPositiveAlignmentConfigError("official source accounting is not preregistered")
    outputs = phase.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("compact_only") is not True:
        raise DprPositiveAlignmentConfigError("outputs must remain compact_only")
    return dict(phase)


def _load_questions() -> list[dict[str, Any]]:
    # Importing the data adapter is deferred until after --validate-only so
    # configuration validation cannot initialize an online RAG component.
    from applications.rag.data import load_dataset_for_rag

    questions = load_dataset_for_rag("nq_open", "validation", ALL_QUESTIONS)
    if len(questions) != ALL_QUESTIONS:
        raise DprPositiveAlignmentConfigError(
            f"expected {ALL_QUESTIONS} NQ-Open validation questions, found {len(questions)}"
        )
    seen_ids: set[str] = set()
    for row in questions:
        question_id = str(row.get("id") or "")
        if not question_id or question_id in seen_ids:
            raise DprPositiveAlignmentConfigError("NQ-Open question IDs must be unique and non-empty")
        if not normalized_question_identity(row.get("question")):
            raise DprPositiveAlignmentConfigError(f"{question_id}: empty NQ-Open question")
        seen_ids.add(question_id)
    return questions


def _is_nonempty_context(value: Any) -> bool:
    """Apply the preregistered source eligibility rule without coercion."""
    return isinstance(value, str) and bool(value.strip())


def _official_gold_info_rows(path: Path) -> list[Mapping[str, Any]]:
    """Read the fixed official gold-info container without exporting its text."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise DprPositiveAlignmentConfigError(
            "official DPR gold-info must be a JSON list or a {data: [...]} container"
        )
    return list(rows)


def _resolve_official_gold_info_source(
    root: Path,
    phase: Mapping[str, Any],
    override: str | None,
    target_question_identities: set[str],
) -> dict[str, Any]:
    """Resolve the fixed source without importing the online audit path."""
    specification = phase["alignment"]["official_dpr_download"]
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    for environment_name in ("DPR_NQ_GOLD_INFO_PATH", "DPR_NQ_POSITIVES_PATH"):
        value = os.environ.get(environment_name)
        if value:
            candidates.append(Path(os.path.expanduser(os.path.expandvars(value))))
    for raw in phase["alignment"]["dpr_positive_candidates"]:
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    cache_path = Path(str(specification["cache_path"]))
    cache_path = cache_path if cache_path.is_absolute() else root / cache_path
    candidates.append(cache_path)

    source_path: Path | None = None
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            source_path = resolved
            break

    downloaded = False
    if source_path is None:
        resolved_cache = cache_path.resolve()
        try:
            resolved_cache.relative_to(root.resolve())
        except ValueError as exc:
            raise DprPositiveAlignmentConfigError(
                "official DPR cache_path must remain within the project root"
            ) from exc
        partial = resolved_cache.with_suffix(resolved_cache.suffix + ".part")
        partial.parent.mkdir(parents=True, exist_ok=True)
        try:
            request = Request(
                str(specification["url"]),
                headers={"User-Agent": "QORE-VLM-gold-audit/1"},
            )
            with urlopen(request, timeout=int(specification["timeout_seconds"])) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
        except (OSError, URLError) as exc:
            raise DprPositiveAlignmentConfigError(
                f"cannot obtain official DPR gold-info source: {exc}"
            ) from exc
        if not partial.is_file() or partial.stat().st_size == 0:
            raise DprPositiveAlignmentConfigError(
                "official DPR gold-info download is empty"
            )
        partial.replace(resolved_cache)
        source_path = resolved_cache
        downloaded = True

    records = _official_gold_info_rows(source_path)
    return {
        "source_name": str(specification["source_name"]),
        "source_url": str(specification["url"]),
        "path": _project_relative(root, source_path),
        "byte_count": source_path.stat().st_size,
        "sha256": _sha256_file(source_path),
        "records_scanned": len(records),
        "target_records_retained": sum(
            normalized_question_identity(row.get("question"))
            in target_question_identities
            for row in records
        ),
        "downloaded_for_this_run": downloaded,
    }


def _official_context_only_diagnostics(
    path: Path, questions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate source-only eligibility facts using context, not title+context."""
    target_keys = {
        normalized_question_identity(row.get("question"))
        for row in questions
        if normalized_question_identity(row.get("question"))
    }
    target_rows = [
        row for row in _official_gold_info_rows(path)
        if normalized_question_identity(row.get("question")) in target_keys
    ]
    context_present = sum("context" in row for row in target_rows)
    context_string = sum(isinstance(row.get("context"), str) for row in target_rows)
    context_nonempty = sum(_is_nonempty_context(row.get("context")) for row in target_rows)
    title_nonempty = sum(_is_nonempty_context(row.get("title")) for row in target_rows)
    return {
        "diagnostic_only": True,
        "privacy": "aggregate_counts_only",
        "target_question_record_count": len(target_rows),
        "context_field": {
            "present_record_count": context_present,
            "string_record_count": context_string,
            "nonempty_record_count": context_nonempty,
            "empty_or_missing_record_count": len(target_rows) - context_nonempty,
        },
        "title_field": {
            "nonempty_record_count": title_nonempty,
            "context_nonempty_title_empty_record_count": context_nonempty - sum(
                _is_nonempty_context(row.get("context"))
                and _is_nonempty_context(row.get("title"))
                for row in target_rows
            ),
        },
        "eligibility_rule": "official_context_is_nonempty_string",
    }


def _build_context_only_question_index(
    source_path: Path,
) -> dict[str, DprQuestionEvidence]:
    """Build the new audit adapter without changing the production aligner.

    `nq-test_gold_info` has no canonical Wiki-DPR passage ID.  A non-empty
    context therefore remains eligible with an absent title and can only be
    resolved later by exact title/full-text or unique full-text identity.
    """
    grouped: dict[str, list[tuple[OfficialPositiveContext, ...]]] = {}
    for row in _official_gold_info_rows(source_path):
        key = normalized_question_identity(row.get("question"))
        if not key:
            continue
        context = row.get("context")
        positives: tuple[OfficialPositiveContext, ...]
        if _is_nonempty_context(context):
            positives = (OfficialPositiveContext(
                title=str(row.get("title") or ""),
                text=context,
                source_passage_id=None,
            ),)
        else:
            positives = ()
        grouped.setdefault(key, []).append(positives)

    index: dict[str, DprQuestionEvidence] = {}
    for key, records_for_question in grouped.items():
        signatures = {
            tuple((positive.title_text_fingerprint, positive.text_fingerprint) for positive in positives)
            for positives in records_for_question
        }
        index[key] = DprQuestionEvidence(
            question_identity=key,
            positives=records_for_question[0] if len(signatures) == 1 else (),
            duplicate_record_count=len(records_for_question),
            ambiguous_records=len(signatures) > 1,
        )
    return index


def _joins_for_population(
    questions: Sequence[Mapping[str, Any]], index: Mapping[str, DprQuestionEvidence]
) -> tuple[dict[str, tuple[str, Any]], list[dict[str, Any]]]:
    joins: dict[str, tuple[str, Any]] = {}
    eligible: list[dict[str, Any]] = []
    for raw in questions:
        row = dict(raw)
        question_id = str(row["id"])
        evidence = index.get(normalized_question_identity(row["question"]))
        if evidence is None:
            status = "no_question_join"
        elif evidence.ambiguous_records:
            status = "ambiguous_question_join"
        elif not evidence.positives:
            status = "no_positive_context"
        else:
            status = "joined"
        joins[question_id] = (status, evidence)
        if status == "joined":
            eligible.append(row)
    return joins, eligible


def _source_preflight(
    questions: Sequence[Mapping[str, Any]],
    joins: Mapping[str, tuple[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
    dpr_source: Mapping[str, Any],
    phase: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    status_counts = collections.Counter(status for status, _ in joins.values())
    source_path = Path(str(dpr_source["path"]))
    if not source_path.is_absolute():
        source_path = root / source_path
    source_diagnostics = _official_context_only_diagnostics(source_path.resolve(), questions)
    all_count = len(questions)
    expected = phase["expected_source_accounting"]
    exact_join_count = sum(status != "no_question_join" for status, _ in joins.values())
    accounting = {
        "all_questions": all_count,
        "source_records_scanned": int(dpr_source.get("records_scanned", 0)),
        "source_target_records_retained": int(dpr_source.get("target_records_retained", 0)),
        "exact_question_joined": exact_join_count,
        "eligible_nonempty_context_questions": len(eligible),
        "empty_context_questions": int(status_counts.get("no_positive_context", 0)),
        "no_question_join_questions": int(status_counts.get("no_question_join", 0)),
        "ambiguous_question_join_questions": int(status_counts.get("ambiguous_question_join", 0)),
        "official_source_nonempty_context_questions": int(
            source_diagnostics["context_field"]["nonempty_record_count"]
        ),
    }
    source_population_match = (
        accounting["all_questions"] == expected["all_questions"]
        and accounting["source_target_records_retained"] == expected["all_questions"]
        and accounting["exact_question_joined"] == expected["all_questions"]
        and accounting["eligible_nonempty_context_questions"] == expected["official_nonempty_context_questions"]
        and accounting["official_source_nonempty_context_questions"] == expected["official_nonempty_context_questions"]
        and accounting["empty_context_questions"] == expected["all_questions"] - expected["official_nonempty_context_questions"]
        and accounting["ambiguous_question_join_questions"] == 0
    )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "model_loaded": False,
        "wiki_dpr_started": False,
        "selection_mutation": False,
        "accounting": accounting,
        "join_status_counts": {
            status: int(status_counts.get(status, 0))
            for status in ("joined", "no_positive_context", "no_question_join", "ambiguous_question_join")
        },
        "source_population_match": source_population_match,
        "source_diagnostics": source_diagnostics,
        "decision": {
            "status": "preflight_passed" if source_population_match else "blocked_source_accounting",
            "may_start_wiki_dpr": source_population_match,
            "interpretation": (
                "The official eligible universe is fully accounted for; strict Wiki-DPR mapping may start."
                if source_population_match
                else "Do not start Wiki-DPR, models, retrieval, selection, or generation until source accounting is repaired."
            ),
        },
    }


def _resolve_generator(root: Path, specification: Mapping[str, Any], override: str | None) -> dict[str, Any]:
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
    raise DprPositiveAlignmentConfigError("reference Generator not found; attempted: " + ", ".join(attempted))


def _retrieved_rows(manager: Any, query_embedding: np.ndarray, top_k: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise DprPositiveAlignmentConfigError("Wiki-DPR manager does not expose its indexed dataset")
    scores, retrieved = dataset.get_nearest_examples("embeddings", np.asarray(query_embedding, dtype=np.float32), k=top_k)
    ids = retrieved.get("id", retrieved.get("passage_id", retrieved.get("psg_id", [])))
    rows: list[dict[str, Any]] = []
    for index, raw_text in enumerate(retrieved.get("text", [])):
        title = str(retrieved.get("title", [""] * top_k)[index] or "")
        text = str(raw_text or "")
        rows.append({
            "id": ids[index] if index < len(ids) else None,
            "title": title,
            "text": text,
            "prompt_passage": f"{title}. {text}" if title else text,
            "score": float(scores[index]),
        })
    return rows, np.asarray(retrieved.get("embeddings", []), dtype=np.float32)


def _passage_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        passage_identity(row.get("id"), row.get("title"), row.get("text"), index)
        for index, row in enumerate(rows)
    ]


def _generator_bucket(selected_hit: bool, baseline_em: float) -> str:
    if not selected_hit:
        return "no_selected_strict_gold"
    return "selected_hit_answer_correct" if baseline_em >= 1.0 else "selected_hit_generation_error"


def _publication_metadata(root: Path, run_dir: Path, result_path: Path) -> dict[str, Any]:
    byte_count = result_path.stat().st_size
    return {
        "schema_version": 1,
        "result_path": _project_relative(root, result_path),
        "result_byte_count": byte_count,
        "result_sha256": _sha256_file(result_path),
        "github_result_json_allowed": byte_count <= RESULT_GIT_MAX_BYTES,
        "github_max_result_json_bytes": RESULT_GIT_MAX_BYTES,
        "required_github_artifacts": ["summary.json", "run_metadata.json", "artifact_publication.json"],
        "exchange_directory": _project_relative(root, run_dir),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/dpr_positive_gold_eligible_universe_audit.yaml")
    parser.add_argument("--output-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dpr-positives", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config_path = config_path.resolve()
    phase = _load_config(config_path)
    if args.validate_only:
        print(json.dumps({
            "status": "valid", "phase": PHASE_NAME, "model_loaded": False,
            "wiki_dpr_started": False, "selection_mutation": False,
            "plugins": list(EXPECTED_PLUGINS),
        }, sort_keys=True))
        return None

    output_root = Path(args.output_root or phase["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    questions = _load_questions()
    target_question_identities = {normalized_question_identity(row["question"]) for row in questions}
    dpr_source = _resolve_official_gold_info_source(
        root, phase, args.dpr_positives, target_question_identities
    )
    source_path = Path(str(dpr_source["path"]))
    if not source_path.is_absolute():
        source_path = root / source_path
    context_only_index = _build_context_only_question_index(source_path.resolve())
    joins, eligible = _joins_for_population(questions, context_only_index)
    preflight = _source_preflight(questions, joins, eligible, dpr_source, phase, root)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "phase": PHASE_NAME,
        "status": "running",
        "diagnostic_only": True,
        "selection_mutation": False,
        "model_loaded": False,
        "wiki_dpr_started": False,
        "config": {"path": _project_relative(root, config_path), "sha256": _sha256_file(config_path)},
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")},
        "python": {"executable": sys.executable, "version": sys.version},
        "dataset": {**phase["dataset"], "dpr_positive_source": dpr_source},
        "preflight": preflight,
        "retrieval": phase["retrieval"],
        "selection": phase["selection"],
        "plugins": {"allowlist": list(EXPECTED_PLUGINS), "tree_sha256": _plugin_tree_hash(root), "diagnostic_outputs_used_for_selection": False},
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    _write_json(run_dir / "preflight.json", preflight)
    if args.preflight_only:
        metadata["status"] = preflight["decision"]["status"]
        _write_json(run_dir / "run_metadata.json", metadata)
        print(f"Completed DPR eligible-universe preflight: {run_dir}")
        print(json.dumps(preflight["accounting"], sort_keys=True))
        if not preflight["decision"]["may_start_wiki_dpr"]:
            raise DprPositiveAlignmentConfigError("source population accounting does not match the preregistered eligible universe")
        return run_dir
    if not preflight["decision"]["may_start_wiki_dpr"]:
        metadata["status"] = "blocked_source_accounting"
        _write_json(run_dir / "run_metadata.json", metadata)
        raise DprPositiveAlignmentConfigError("source preflight failed; Wiki-DPR and models were not started")

    started = time.perf_counter()
    corpus_manager = make_corpus_manager("wiki_dpr", {
        "wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"]),
    })
    corpus_manager.build(eligible)
    metadata["wiki_dpr_started"] = True
    alignment = align_dpr_positives_to_wiki(
        getattr(corpus_manager, "_dataset"), joins,
        progress_every=int(phase["alignment"].get("progress_every", 1_000_000)),
    )
    _write_json(run_dir / "alignment_index.json", {
        "schema_version": 1,
        "matches": {key: {**value, "verified": sorted(value.get("verified", set()))} for key, value in alignment.items()},
    })
    mapped_rows = [
        make_compact_row(row["id"], row["question"], joins[str(row["id"])][0], alignment.get(str(row["id"]), {}))
        for row in eligible
    ]
    mapping_payload = {"schema_version": 1, "phase": PHASE_NAME, "diagnostic_only": True, "selection_mutation": False, "samples": mapped_rows}
    validate_universe_result(mapping_payload)
    mapping_summary = summarize_universe(mapped_rows, all_questions=ALL_QUESTIONS, official_nonempty_context_questions=OFFICIAL_NONEMPTY_CONTEXT_QUESTIONS, minimum_mapping_rate=float(phase["gate"]["minimum_mapping_rate"]))
    _write_json(run_dir / "mapping_result.json", mapping_payload)
    _write_json(run_dir / "mapping_summary.json", mapping_summary)
    if not mapping_summary["mapping"]["mapping_gate_pass"]:
        metadata.update({"status": "blocked_mapping_gate", "timing_ms": {"mapping": (time.perf_counter() - started) * 1000.0}, "decision": "do_not_load_retriever_or_generator"})
        _write_json(run_dir / "run_metadata.json", metadata)
        raise DprPositiveAlignmentConfigError("strict Wiki-DPR mapping rate is below 0.80; retrieval, QORE, and Generator were not started")

    # The online baseline components are intentionally unavailable to both
    # --validate-only and every failed preflight/mapping-gate execution.
    from applications.rag.answer_scorer import make_answer_scorer
    from applications.rag.data import make_corpus_manager
    from applications.rag.evaluation import evaluate_answer
    from applications.rag.generation import Generator
    from applications.rag.retrieval import make_encoder
    from applications.rag.selector import select_passages

    generator_identity = _resolve_generator(root, phase["generator"], args.model_path)
    encoder = make_encoder("dpr")
    answer_scorer = make_answer_scorer(backend="dpr")
    generator = Generator(generator_identity["model_path"], max_new_tokens=int(phase["generator"]["max_new_tokens"]), use_chat_template=True)
    metadata["model_loaded"] = True
    metadata["generator"] = generator_identity
    strict_mapped = [
        item for item in eligible
        if alignment.get(str(item["id"]), {}).get("mapping_status") == "mapped"
    ]
    generated_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(strict_mapped, start=1):
        question_id = str(item["id"])
        question = str(item["question"])
        query_embedding = encoder.encode_queries([question])[0]
        retrieved_rows, embeddings = _retrieved_rows(corpus_manager, query_embedding, int(phase["retrieval"]["top_k"]))
        if len(retrieved_rows) != 50 or embeddings.shape[0] != 50:
            raise DprPositiveAlignmentConfigError(f"{question_id}: incomplete frozen Top-50 retrieval")
        prompt_passages = [row["prompt_passage"] for row in retrieved_rows]
        relevance_scores = answer_scorer.score_passages(question, prompt_passages)
        selected_local = select_passages(
            query_embedding, embeddings, K=5, method="qore", num_reads=100, lam=2.0,
            gamma=1.0, delta=0.0, complementarity_method=None, qore_prefilter_size=None,
            direct_solve_max_n=20, lambda_mmr=0.7, saturation_alpha=1.0,
            lambda_submodular=0.5, dpp_quality_scale=2.0, dpp_jitter=1.0e-8,
            answer_scorer=answer_scorer, passage_texts=prompt_passages, question=question,
            seed=42, relevance_scores=relevance_scores,
        )
        if len(selected_local) != 5:
            raise DprPositiveAlignmentConfigError(f"{question_id}: QORE did not return Top-5")
        retrieved_ids = _passage_ids(retrieved_rows)
        selected_ids = [retrieved_ids[int(local)] for local in selected_local]
        preliminary = make_compact_row(question_id, question, joins[question_id][0], alignment[question_id], retrieved_ids, selected_ids)
        prediction = generator.generate(question, [prompt_passages[int(local)] for local in selected_local])
        scores = evaluate_answer(prediction, [str(value) for value in item.get("answers", [])])
        generated_by_id[question_id] = make_compact_row(
            question_id, question, joins[question_id][0], alignment[question_id], retrieved_ids, selected_ids,
            baseline_em=float(scores["em"]), baseline_f1=float(scores["f1"]),
            generator_bucket=_generator_bucket(bool(preliminary["selected_hit"]), float(scores["em"])),
        )
        if index % 25 == 0 or index == len(strict_mapped):
            print(f"  DPR eligible-universe observer: {index}/{len(strict_mapped)} strict-mapped")

    base_by_id = {
        str(item["id"]): make_compact_row(
            item["id"], item["question"], joins[str(item["id"])][0],
            alignment.get(str(item["id"]), {}),
        )
        for item in eligible
    }
    base_by_id.update(generated_by_id)
    samples = [base_by_id[str(item["id"])] for item in eligible]
    result = {"schema_version": 1, "phase": PHASE_NAME, "diagnostic_only": True, "selection_mutation": False, "samples": samples}
    validate_universe_result(result)
    result_path = run_dir / "result.json"
    _write_json(result_path, result)
    summary = summarize_universe(samples, all_questions=ALL_QUESTIONS, official_nonempty_context_questions=OFFICIAL_NONEMPTY_CONTEXT_QUESTIONS, minimum_mapping_rate=float(phase["gate"]["minimum_mapping_rate"]))
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "artifact_publication.json", _publication_metadata(root, run_dir, result_path))
    metadata.update({"status": "completed", "timing_ms": {"total": (time.perf_counter() - started) * 1000.0}, "summary": {"path": _project_relative(root, run_dir / "summary.json"), "status": summary["decision"]["status"]}})
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed DPR eligible-universe audit: {run_dir}")
    print(f"Decision: {summary['decision']['status']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (DprPositiveAlignmentConfigError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
