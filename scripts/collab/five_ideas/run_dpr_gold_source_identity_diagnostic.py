"""Read-only diagnosis of the official DPR gold source versus Wiki-DPR metadata.

This diagnostic is deliberately cheaper than the strict full-corpus identity
audit.  It reads an already-local ``nq-test_gold_info`` file and metadata files
already present in the Hugging Face cache.  It never downloads data, loads a
Dataset/FAISS index, loads a model, runs retrieval, or writes passage/question
content to the report.  Its purpose is to distinguish a source/corpus
representation mismatch from an algorithmic retrieval failure.
"""

from __future__ import annotations

import collections
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

PHASE_NAME = "dpr_gold_source_identity_diagnostic"
EXPECTED_CONFIG = "psgs_w100.nq.compressed"
EXPECTED_ALL_QUESTIONS = 3610
EXPECTED_NONEMPTY_CONTEXTS = 1868
SOURCE_CANDIDATES = (
    "data/dpr/nq-test_gold_info.json",
    "data/dpr/nq-test_gold_info.json.gz",
    "data/nq/nq-test_gold_info.json",
    "data/nq/nq-test_gold_info.json.gz",
    "data/gold_passages_info/nq-test_gold_info.json",
    "data/gold_passages_info/nq-test_gold_info.json.gz",
)
PREFLIGHT_GLOB = "exchange/five_ideas/dpr_positive_gold_eligible_universe_audit/*/preflight.json"
KNOWN_SOURCE_FIELDS = (
    "question",
    "question_tokens",
    "title",
    "context",
    "example_id",
    "id",
    "passage_id",
    "psg_id",
    "docid",
    "positive_ctxs",
    "positive_contexts",
)
IDENTITY_FIELDS = ("example_id", "id", "passage_id", "psg_id", "docid")
MAX_METADATA_FILES = 128


class DiagnosticError(RuntimeError):
    """Raised when the local-only diagnostic contract cannot be satisfied."""


def _root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise DiagnosticError("cannot locate project root")


def _sha256(path: Path) -> str:
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
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _relative_label(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    return type(value).__name__


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return bool(value)
    return value is not None


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).lower()


def _quantile(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return int(ordered[0])
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return int(ordered[low])
    return int(round(ordered[low] + (ordered[high] - ordered[low]) * (position - low)))


def _length_stats(values: Iterable[int], *, buckets: Mapping[str, tuple[int, int | None]]) -> dict[str, Any]:
    observed = [int(value) for value in values if int(value) > 0]
    counts = {name: 0 for name in buckets}
    for value in observed:
        for name, (lower, upper) in buckets.items():
            if value >= lower and (upper is None or value <= upper):
                counts[name] += 1
                break
    return {
        "nonempty_count": len(observed),
        "min": min(observed) if observed else None,
        "p25": _quantile(observed, 0.25),
        "median": _quantile(observed, 0.50),
        "p75": _quantile(observed, 0.75),
        "p95": _quantile(observed, 0.95),
        "max": max(observed) if observed else None,
        "bucket_counts": counts,
    }


def _read_rows(path: Path) -> tuple[str, list[Mapping[str, Any]]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    container = "object_data" if isinstance(payload, Mapping) else "array"
    rows = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise DiagnosticError("official DPR source must be a JSON list or an object with data[]")
    return container, list(rows)


def _source_analysis(path: Path, root: Path) -> dict[str, Any]:
    container, rows = _read_rows(path)
    field_stats: dict[str, Any] = {}
    for field in KNOWN_SOURCE_FIELDS:
        present = [row[field] for row in rows if field in row]
        field_stats[field] = {
            "present_count": len(present),
            "nonempty_count": sum(_nonempty(value) for value in present),
            "type_counts": dict(collections.Counter(_value_type(value) for value in present)),
        }

    contexts = [row.get("context") for row in rows]
    titles = [row.get("title") for row in rows]
    nonempty_contexts = [value for value in contexts if isinstance(value, str) and value.strip()]
    context_chars = [len(value) for value in nonempty_contexts]
    context_words = [len(value.split()) for value in nonempty_contexts]
    context_hashes = {
        hashlib.sha256(_normalize(value).encode("utf-8")).hexdigest()
        for value in nonempty_contexts
    }
    title_context_hashes = {
        hashlib.sha256((
            _normalize(row.get("title")) + "\0" + _normalize(row.get("context"))
        ).encode("utf-8")).hexdigest()
        for row in rows
        if isinstance(row.get("context"), str) and row.get("context", "").strip()
    }
    word_buckets = {
        "1_63": (1, 63),
        "64_89": (64, 89),
        "90_110": (90, 110),
        "111_127": (111, 127),
        "128_255": (128, 255),
        "256_511": (256, 511),
        "512_plus": (512, None),
    }
    char_buckets = {
        "1_255": (1, 255),
        "256_511": (256, 511),
        "512_1023": (512, 1023),
        "1024_2047": (1024, 2047),
        "2048_plus": (2048, None),
    }
    in_100_word_band = sum(90 <= value <= 110 for value in context_words)
    over_110_words = sum(value > 110 for value in context_words)
    denominator = len(context_words)
    if not denominator:
        length_class = "no_nonempty_context"
    elif over_110_words / denominator >= 0.50:
        length_class = "long_context_dominant"
    elif in_100_word_band / denominator >= 0.50:
        length_class = "100_word_band_dominant"
    else:
        length_class = "mixed_context_lengths"

    identity_fields = {
        field + "_key": {
            "present_count": field_stats[field]["present_count"],
            "nonempty_count": field_stats[field]["nonempty_count"],
        }
        for field in IDENTITY_FIELDS
    }
    candidate_id_fields = [
        field for field in IDENTITY_FIELDS[1:]
        if identity_fields[field + "_key"]["nonempty_count"] > 0
    ]
    return {
        "source_file": {
            "path": _relative_label(root, path),
            "byte_count": path.stat().st_size,
            "sha256": _sha256(path),
            "compressed": path.suffix.lower() == ".gz",
            "container": container,
            "record_count": len(rows),
        },
        "field_stats": field_stats,
        "identity_field_stats": identity_fields,
        "identity_interpretation": {
            "canonical_passage_id_field_candidates": candidate_id_fields,
            "example_id_is_not_assumed_to_be_wiki_dpr_passage_id": True,
        },
        "context_lengths": {
            "characters": _length_stats(context_chars, buckets=char_buckets),
            "whitespace_words": _length_stats(context_words, buckets=word_buckets),
            "nonempty_context_count": len(nonempty_contexts),
            "empty_or_nonstring_context_count": len(rows) - len(nonempty_contexts),
            "unique_normalized_context_hashes": len(context_hashes),
            "unique_normalized_title_context_hashes": len(title_context_hashes),
        },
        "hundred_word_compatibility": {
            "classification": length_class,
            "band_definition_words": [90, 110],
            "band_count": in_100_word_band,
            "over_110_count": over_110_words,
            "band_rate": (in_100_word_band / denominator) if denominator else None,
            "over_110_rate": (over_110_words / denominator) if denominator else None,
            "interpretation": (
                "Official contexts are predominantly longer than a 100-word passage; "
                "strict equality to psgs_w100 is likely a representation/version issue, "
                "not evidence of a retrieval miss."
                if length_class == "long_context_dominant"
                else "Length alone does not identify the source/corpus mismatch."
            ),
        },
    }


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    hf_home = os.environ.get("HF_HOME")
    if datasets_cache:
        roots.append(Path(os.path.expanduser(os.path.expandvars(datasets_cache))))
    if hf_home:
        roots.extend([
            Path(os.path.expanduser(os.path.expandvars(hf_home))) / "datasets",
            Path(os.path.expanduser(os.path.expandvars(hf_home))) / "hub",
        ])
    roots.extend([
        Path.home() / ".cache/huggingface/datasets",
        Path.home() / ".cache/huggingface/hub",
    ])
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            unique.append(Path(key))
    return unique


def _metadata_files() -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for cache_root in _cache_roots():
        if not cache_root.is_dir():
            continue
        try:
            walker = os.walk(cache_root)
            for current, directories, names in walker:
                directories[:] = [
                    name for name in directories
                    if name not in {"blobs", ".git", "__pycache__"}
                ]
                if "wiki_dpr" not in current.lower() and "wiki-dpr" not in current.lower():
                    continue
                for name in names:
                    if name not in {"dataset_info.json", "dataset_infos.json"}:
                        continue
                    path = Path(current) / name
                    key = str(path.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append((cache_root, path))
                        if len(found) >= MAX_METADATA_FILES:
                            return found
        except OSError:
            continue
    return found


def _feature_presence(features: Any) -> dict[str, bool]:
    if not isinstance(features, Mapping):
        return {"id": False, "title": False, "text": False, "embeddings": False}
    names = {str(key).lower() for key in features}
    return {
        "id": "id" in names or "passage_id" in names or "psg_id" in names,
        "title": "title" in names,
        "text": "text" in names or "context" in names,
        "embeddings": "embeddings" in names,
    }


def _dataset_info_summary(cache_root: Path, path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    infos: list[tuple[str, Mapping[str, Any]]] = []
    if path.name == "dataset_info.json" and isinstance(payload, Mapping):
        infos.append((EXPECTED_CONFIG, payload))
    elif isinstance(payload, Mapping):
        for config_name, info in payload.items():
            if isinstance(info, Mapping) and (EXPECTED_CONFIG in str(config_name) or "wiki" in str(config_name).lower()):
                infos.append((str(config_name), info))
    if not infos:
        return None
    output: list[dict[str, Any]] = []
    for config_name, info in infos[:8]:
        splits = info.get("splits")
        split_summary: dict[str, Any] = {}
        if isinstance(splits, Mapping):
            for split_name, split_info in splits.items():
                if isinstance(split_info, Mapping):
                    split_summary[str(split_name)] = {
                        "num_examples": split_info.get("num_examples"),
                        "num_bytes": split_info.get("num_bytes"),
                    }
        output.append({
            "config_name": config_name,
            "version": str(info.get("version")) if info.get("version") is not None else None,
            "description_present": bool(info.get("description")),
            "feature_presence": _feature_presence(info.get("features")),
            "splits": split_summary,
            "download_size": info.get("download_size"),
            "dataset_size": info.get("dataset_size"),
            "size_in_bytes": info.get("size_in_bytes"),
        })
    return {
        "metadata_label": f"{cache_root.name}/{path.relative_to(cache_root).as_posix()}",
        "infos": output,
    }


def _wiki_metadata_analysis() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for cache_root, path in _metadata_files():
        summary = _dataset_info_summary(cache_root, path)
        if summary is not None:
            candidates.append(summary)
    return {
        "cache_roots_checked": [str(root) for root in _cache_roots()],
        "metadata_files_found": len(candidates),
        "candidates": candidates,
        "corpus_contract": {
            "requested_config": EXPECTED_CONFIG,
            "upstream_semantics": "psgs_w100 is documented by DPR as 100-word Wikipedia passages without overlap",
            "semantics_provenance": "facebookresearch/DPR dpr/data/download_data.py RESOURCES_MAP",
            "dataset_rows_or_revision_not_loaded": True,
        },
    }


def _latest_preflight(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob(PREFLIGHT_GLOB), key=lambda path: path.as_posix(), reverse=True)
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        accounting = payload.get("accounting") if isinstance(payload, Mapping) else None
        if not isinstance(accounting, Mapping):
            continue
        return {
            "status": "verified_previous_preflight",
            "path": _relative_label(root, path),
            "sha256": _sha256(path),
            "accounting": {
                "all_questions": accounting.get("all_questions"),
                "exact_question_joined": accounting.get("exact_question_joined"),
                "eligible_nonempty_context_questions": accounting.get("eligible_nonempty_context_questions"),
                "empty_context_questions": accounting.get("empty_context_questions"),
                "no_question_join_questions": accounting.get("no_question_join_questions"),
                "ambiguous_question_join_questions": accounting.get("ambiguous_question_join_questions"),
            },
        }
    return {"status": "previous_preflight_artifact_unavailable"}


def _resolve_source(root: Path) -> tuple[Path | None, list[str]]:
    candidates: list[Path] = []
    for name in ("DPR_NQ_GOLD_INFO_PATH", "DPR_NQ_POSITIVES_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(os.path.expanduser(os.path.expandvars(value))))
    candidates.extend(root / relative for relative in SOURCE_CANDIDATES)
    attempted: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        label = _relative_label(root, resolved)
        if label in seen:
            continue
        seen.add(label)
        attempted.append(label)
        if resolved.is_file():
            return resolved, attempted
    return None, attempted


def _load_config(root: Path, path: Path | None) -> tuple[Path, Mapping[str, Any]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - only validate/run needs YAML
        raise DiagnosticError("PyYAML is required to validate the diagnostic config") from exc
    config_path = (path or (root / "configs/experiments/dpr_gold_source_identity_diagnostic.yaml")).resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise DiagnosticError(f"cannot read diagnostic config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, Mapping) or phase.get("name") != PHASE_NAME:
        raise DiagnosticError("unexpected source identity diagnostic config")
    if phase.get("diagnostic_only") is not True or phase.get("data_download") is not False:
        raise DiagnosticError("diagnostic must remain local-only and diagnostic-only")
    if phase.get("wiki_dpr_config") != EXPECTED_CONFIG:
        raise DiagnosticError("Wiki-DPR config is not frozen")
    return config_path, phase


def _validate_only(root: Path, config_path: Path, phase: Mapping[str, Any]) -> None:
    print(json.dumps({
        "status": "valid",
        "phase": PHASE_NAME,
        "config": _relative_label(root, config_path),
        "diagnostic_only": True,
        "data_download": False,
        "wiki_dpr_started": False,
        "model_loaded": False,
        "selection_mutation": False,
        "wiki_dpr_config": phase.get("wiki_dpr_config"),
    }, sort_keys=True))


def run(*, config: Path | None = None, output_root: Path | None = None, validate_only: bool = False) -> Path | None:
    root = _root()
    config_path, phase = _load_config(root, config)
    if validate_only:
        _validate_only(root, config_path, phase)
        return None

    source, attempted = _resolve_source(root)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_root or (root / str(phase.get("output_root", "exchange/five_ideas/dpr_gold_source_identity_diagnostic")))
    if not destination.is_absolute():
        destination = root / destination
    run_dir = destination / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = destination / f"{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "phase": PHASE_NAME,
        "status": "running",
        "diagnostic_only": True,
        "data_download": False,
        "wiki_dpr_started": False,
        "model_loaded": False,
        "selection_mutation": False,
        "config": {"path": _relative_label(root, config_path), "sha256": _sha256(config_path)},
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")},
        "python": {"executable": sys.executable, "version": sys.version},
        "source_resolution": {"attempted_local_paths": attempted, "resolved": source is not None},
        "wiki_dpr_config": EXPECTED_CONFIG,
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    if source is None:
        metadata["status"] = "blocked_source_unavailable"
        _write_json(run_dir / "run_metadata.json", metadata)
        raise DiagnosticError("local official DPR gold source not found; no download was attempted")

    source_report = _source_analysis(source, root)
    preflight_report = _latest_preflight(root)
    wiki_report = _wiki_metadata_analysis()
    length_class = source_report["hundred_word_compatibility"]["classification"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": PHASE_NAME,
        "diagnostic_only": True,
        "data_download": False,
        "wiki_dpr_started": False,
        "model_loaded": False,
        "selection_mutation": False,
        "source": source_report,
        "previous_preflight": preflight_report,
        "wiki_dpr_metadata": wiki_report,
        "decision": {
            "status": "source_corpus_version_diagnosis",
            "strict_mapping_result_is_not_reinterpreted": True,
            "likely_representation_mismatch": length_class == "long_context_dominant",
            "interpretation": (
                "The official context length profile is incompatible with treating each context as one "
                "psgs_w100 passage; seek an official same-corpus ID or a reproducible chunk-level adapter. "
                "Do not rerun the full identity scan yet."
                if length_class == "long_context_dominant"
                else "Length profile alone is inconclusive; inspect the cached dataset revision/features before designing an adapter."
            ),
            "next_gate": "Require an official, traceable same-corpus identity or stop this gold-alignment route; never use fuzzy/answer matching as a substitute.",
        },
    }
    _write_json(run_dir / "report.json", report)
    metadata["status"] = "completed"
    metadata["report"] = {
        "path": _relative_label(root, run_dir / "report.json"),
        "sha256": _sha256(run_dir / "report.json"),
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed local DPR source identity diagnostic: {run_dir}")
    print(json.dumps(report["decision"], sort_keys=True))
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(config=args.config, output_root=args.output_root, validate_only=args.validate_only)
        return 0
    except (DiagnosticError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
