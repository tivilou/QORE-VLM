#!/usr/bin/env python3
"""Audit official DPR positive contexts against frozen retrieval and QORE.

This collaborator-only runner is observation-only. It uses official DPR
positive contexts only to construct an offline identity index; labels never
reach retrieval, selection, prompting, generation, or evaluation.
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
from typing import Any, Iterable, Mapping, Sequence
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

from applications.rag.answer_scorer import make_answer_scorer
from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.dpr_positive_gold_alignment import (
    DPR_MAPPING_STATUSES,
    align_dpr_positives_to_wiki,
    build_dpr_question_index,
    evidence_for_question,
    normalized_question_identity,
    passage_identity,
    passage_text_fingerprint,
    summarize_dpr_positive_alignment,
    validate_dpr_compact_result,
)
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages


class DprPositiveAlignmentConfigError(RuntimeError):
    """Raised when the frozen v2 audit contract is malformed."""


EXPECTED_SELECTION = {
    "method": "qore", "K": 5, "num_reads": 100, "lam": 2.0, "seed": 42,
    "gamma": 1.0, "delta": 0.0, "complementarity_method": None,
    "qore_prefilter_size": None, "direct_solve_max_n": 20,
    "lambda_mmr": 0.7, "saturation_alpha": 1.0, "lambda_submodular": 0.5,
    "dpp_quality_scale": 2.0, "dpp_jitter": 1.0e-8,
    "use_answer_scorer": True, "answer_scorer_backend": "dpr",
}


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


def _load_config(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise DprPositiveAlignmentConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, Mapping) or phase.get("name") != "dpr_positive_gold_alignment_audit":
        raise DprPositiveAlignmentConfigError("unexpected DPR positive alignment configuration")
    if phase.get("schema_version") != 2 or phase.get("diagnostic_only") is not True:
        raise DprPositiveAlignmentConfigError("audit must be schema 2 and diagnostic_only")
    if phase.get("selection_mutation") is not False:
        raise DprPositiveAlignmentConfigError("selection_mutation must remain false")
    dataset = phase.get("dataset", {}).get(stage)
    expected_dataset = {
        "name": "nq_open", "split": "validation", "sample_offset": 200,
        "max_samples": 50, "fresh_slice": True,
    }
    if dict(dataset or {}) != expected_dataset:
        raise DprPositiveAlignmentConfigError(f"dataset.{stage} slice is not frozen")
    if phase.get("retrieval") != {
        "corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64, "top_k": 50,
    }:
        raise DprPositiveAlignmentConfigError("retrieval contract is not frozen")
    if dict(phase.get("selection", {})) != EXPECTED_SELECTION:
        raise DprPositiveAlignmentConfigError("selection contract is not frozen")
    alignment = phase.get("alignment")
    if not isinstance(alignment, Mapping) or alignment.get("scan_mode") != "full_corpus":
        raise DprPositiveAlignmentConfigError("full_corpus alignment scan is required")
    if not isinstance(alignment.get("dpr_positive_candidates"), list):
        raise DprPositiveAlignmentConfigError("DPR positive source candidates are missing")
    download = alignment.get("official_dpr_download")
    if not isinstance(download, Mapping) or not all(
        isinstance(download.get(key), str) and download[key].strip()
        for key in ("source_name", "url", "cache_path")
    ):
        raise DprPositiveAlignmentConfigError("official DPR download source is not frozen")
    timeout_seconds = download.get("timeout_seconds")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise DprPositiveAlignmentConfigError("official DPR download timeout must be a positive integer")
    outputs = phase.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("compact_only") is not True:
        raise DprPositiveAlignmentConfigError("outputs.compact_only must be true")
    return dict(phase), dict(dataset)


def _iter_json_array_records(path: Path) -> Iterable[Mapping[str, Any]]:
    """Stream a JSON array so the DPR release is never held in memory."""
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False
    with opener(path, "rt", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(1024 * 1024)
            eof = not chunk
            buffer += chunk
            if not started:
                buffer = buffer.lstrip()
                if not buffer:
                    if eof:
                        raise DprPositiveAlignmentConfigError(f"empty DPR positive source: {path}")
                    continue
                if buffer[0] != "[":
                    raise DprPositiveAlignmentConfigError(
                        f"DPR positive source must be a JSON array or JSONL: {path}"
                    )
                buffer = buffer[1:]
                started = True
            while buffer:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    if buffer[1:].strip():
                        raise DprPositiveAlignmentConfigError(
                            f"trailing content in DPR positive source: {path}"
                        )
                    finished = True
                    buffer = ""
                    break
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                buffer = buffer[end:]
                if isinstance(value, Mapping):
                    yield value
            if eof:
                if not finished:
                    raise DprPositiveAlignmentConfigError(
                        f"incomplete DPR positive JSON array: {path}"
                    )
                return


def _iter_dpr_positive_records(path: Path) -> Iterable[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DprPositiveAlignmentConfigError(
                        f"invalid DPR JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if isinstance(value, Mapping):
                    yield value
        return
    yield from _iter_json_array_records(path)


def _record_question_identity(record: Mapping[str, Any]) -> str:
    value = record.get("question") or record.get("query") or record.get("question_text")
    if isinstance(value, Mapping):
        value = value.get("text") or value.get("question")
    return normalized_question_identity(value)


def _project_relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _download_official_dpr_source(
    *, url: str, destination: Path, timeout_seconds: int
) -> None:
    """Download the immutable run input through a non-destructive partial path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise DprPositiveAlignmentConfigError(
            f"stale partial DPR download exists: {partial}; inspect or remove it before retrying"
        )
    request = Request(url, headers={"User-Agent": "QORE-VLM-DPR-alignment-audit/2"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response, partial.open("xb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
    except (OSError, URLError) as exc:
        partial.unlink(missing_ok=True)
        raise DprPositiveAlignmentConfigError(
            f"cannot download official DPR positives from {url}: {exc}"
        ) from exc
    if not partial.is_file() or partial.stat().st_size == 0:
        raise DprPositiveAlignmentConfigError(
            f"official DPR download produced an empty file: {partial}"
        )
    partial.replace(destination)


def _collect_target_dpr_records(
    path: Path, target_question_identities: set[str]
) -> tuple[list[Mapping[str, Any]], int]:
    retained: list[Mapping[str, Any]] = []
    scanned = 0
    for record in _iter_dpr_positive_records(path):
        scanned += 1
        if _record_question_identity(record) in target_question_identities:
            retained.append(record)
    return retained, scanned


def _load_dpr_positive_records(
    root: Path,
    phase: Mapping[str, Any],
    override: str | None,
    target_question_identities: set[str],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    for env_name in ("DPR_NQ_POSITIVES_PATH", "DPR_NQ_DEV_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(os.path.expanduser(os.path.expandvars(value))))
    for raw in phase["alignment"]["dpr_positive_candidates"]:
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    download = phase["alignment"]["official_dpr_download"]
    cache_path = Path(str(download["cache_path"]))
    cache_path = cache_path if cache_path.is_absolute() else root / cache_path
    candidates.append(cache_path)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            records, scanned = _collect_target_dpr_records(
                resolved, target_question_identities
            )
            return records, {
                "source_name": str(download["source_name"]),
                "source_url": str(download["url"]),
                "path": _project_relative_path(root, resolved),
                "byte_count": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
                "records_scanned": scanned,
                "target_records_retained": len(records),
                "downloaded_for_this_run": False,
            }

    resolved_cache = cache_path.resolve()
    try:
        resolved_cache.relative_to(root.resolve())
    except ValueError as exc:
        raise DprPositiveAlignmentConfigError(
            "official DPR cache_path must remain within the project root"
        ) from exc
    _download_official_dpr_source(
        url=str(download["url"]),
        destination=resolved_cache,
        timeout_seconds=int(download["timeout_seconds"]),
    )
    records, scanned = _collect_target_dpr_records(
        resolved_cache, target_question_identities
    )
    return records, {
        "source_name": str(download["source_name"]),
        "source_url": str(download["url"]),
        "path": _project_relative_path(root, resolved_cache),
        "byte_count": resolved_cache.stat().st_size,
        "sha256": _sha256_file(resolved_cache),
        "records_scanned": scanned,
        "target_records_retained": len(records),
        "downloaded_for_this_run": True,
    }


def _value(values: Any, index: int, default: Any = None) -> Any:
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _retrieved_rows(
    manager: Any, query_embedding: np.ndarray, top_k: int
) -> tuple[list[dict[str, Any]], np.ndarray]:
    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise DprPositiveAlignmentConfigError(
            "wiki_dpr manager does not expose the indexed dataset"
        )
    scores, retrieved = dataset.get_nearest_examples(
        "embeddings", np.asarray(query_embedding, dtype=np.float32), k=top_k
    )
    texts = retrieved.get("text", [])
    titles = retrieved.get("title", [])
    ids = retrieved.get("id", retrieved.get("passage_id", retrieved.get("psg_id", [])))
    embeddings = np.asarray(retrieved.get("embeddings", []), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for index in range(len(texts)):
        rows.append({
            "id": _value(ids, index),
            "title": str(_value(titles, index, "") or ""),
            "text": str(_value(texts, index, "") or ""),
            "score": float(_value(scores, index, 0.0)),
        })
    return rows, embeddings


def _sample_row(
    question_id: str,
    question_identity_verified: bool,
    join_status: str,
    identity: Mapping[str, Any],
    retrieved_rows: Sequence[Mapping[str, Any]],
    selected_local: Sequence[int],
) -> dict[str, Any]:
    verified_gold = set(identity.get("verified", set()))
    retrieved_ids = {
        passage_identity(row.get("id"), row.get("title"), row.get("text"), index)
        for index, row in enumerate(retrieved_rows)
    }
    selected_ids = {
        passage_identity(
            retrieved_rows[int(index)].get("id"),
            retrieved_rows[int(index)].get("title"),
            retrieved_rows[int(index)].get("text"),
            int(index),
        )
        for index in selected_local
    }
    top50 = retrieved_ids & verified_gold
    top5 = selected_ids & verified_gold
    mapping_status = identity.get("mapping_status", join_status)
    return {
        "question_id": str(question_id),
        "question_identity_verified": bool(question_identity_verified),
        "mapping_status": mapping_status if mapping_status in DPR_MAPPING_STATUSES else "no_question_join",
        "official_positive_count": int(identity.get("official_positive_count", 0)),
        "verified_positive_count": int(identity.get("verified_positive_count", 0)),
        "unresolved_positive_count": int(identity.get("unresolved_positive_count", 0)),
        "ambiguous_positive_count": int(identity.get("ambiguous_positive_count", 0)),
        "gold_passage_count": len(verified_gold),
        "top50_gold_count": len(top50),
        "top5_gold_count": len(top5),
        "retrieval_hit": bool(top50),
        "selected_hit": bool(top5),
        "retrieval_top_k": len(retrieved_rows),
        "selection_k": len(selected_local),
    }


def _question_join_preflight(
    questions: Sequence[Mapping[str, Any]],
    joins: Mapping[str, tuple[str, Any]],
    *,
    minimum_mapping_rate: float,
) -> dict[str, Any]:
    """Summarize the cheap question/source join before any Wiki-DPR work."""
    status_counts = collections.Counter(status for status, _ in joins.values())
    sample_count = len(questions)
    joined_count = int(status_counts.get("joined", 0))
    required_join_count = int(math.ceil(float(minimum_mapping_rate) * sample_count))
    return {
        "sample_count": sample_count,
        "joined_count": joined_count,
        "required_join_count": required_join_count,
        "join_rate": (joined_count / sample_count) if sample_count else None,
        "join_status_counts": {
            status: int(status_counts.get(status, 0))
            for status in ("joined", "no_question_join", "ambiguous_question_join", "no_positive_context")
        },
        "can_reach_mapping_gate": joined_count >= required_join_count,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/experiments/dpr_positive_gold_alignment_audit.yaml",
    )
    parser.add_argument("--stage", choices=("screen",), default="screen")
    parser.add_argument("--dpr-positives", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the target-question/DPR-source join without starting Wiki-DPR or loading models",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _project_root()
    config_path = (args.config if args.config.is_absolute() else root / args.config).resolve()
    phase, stage = _load_config(config_path, args.stage)
    if args.validate_only:
        print(json.dumps({
            "status": "valid", "phase": phase["name"], "schema_version": 2,
            "stage": args.stage, "slice": stage, "selection_mutation": False,
            "scan_mode": phase["alignment"]["scan_mode"],
            "model_loaded": False, "wiki_dpr_started": False,
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

    sample_end = int(stage["sample_offset"]) + int(stage["max_samples"])
    questions = load_dataset_for_rag("nq_open", "validation", sample_end)
    questions = questions[int(stage["sample_offset"]):sample_end]
    if len(questions) != int(stage["max_samples"]):
        raise DprPositiveAlignmentConfigError("nq_open fresh slice length mismatch")
    target_question_identities = {
        normalized_question_identity(item["question"]) for item in questions
    }
    dpr_records, dpr_source = _load_dpr_positive_records(
        root, phase, args.dpr_positives, target_question_identities
    )

    dpr_index = build_dpr_question_index(dpr_records)
    joins: dict[str, tuple[str, Any]] = {}
    question_verified: dict[str, bool] = {}
    for item in questions:
        question_id = str(item["id"])
        status, evidence = evidence_for_question(item["question"], dpr_index)
        joins[question_id] = (status, evidence)
        question_verified[question_id] = status == "joined"

    join_preflight = _question_join_preflight(
        questions,
        joins,
        minimum_mapping_rate=float(phase["gate"]["minimum_mapping_rate"]),
    )

    metadata: dict[str, Any] = {
        "schema_version": 2,
        "phase": phase["name"],
        "stage": args.stage,
        "status": "running",
        "diagnostic_only": True,
        "selection_mutation": False,
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "status": _git(root, "status", "--short", "--branch"),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "dataset": {"name": "nq_open", "split": "validation", **stage,
                    "dpr_positive_source": dpr_source, "dpr_target_record_count": len(dpr_records)},
        "preflight": join_preflight,
        "retrieval": phase["retrieval"],
        "selection": phase["selection"],
        "identity_rules": phase["alignment"]["identity_rules"],
        "outputs": {"compact_only": True, "alignment_cache": "alignment_index.json"},
        "plugins": {
            "allowlist": [
                "dpr_question_identity_join",
                "official_dpr_positive_contexts",
                "wiki_dpr_identity_alignment",
                "retrieval_top50_observer",
                "qore_top5_observer",
            ],
            "diagnostic_outputs_used_for_selection": False,
        },
    }
    if args.preflight_only:
        metadata["status"] = "preflight_completed"
        _write_json(run_dir / "run_metadata.json", metadata)
        _write_json(run_dir / "preflight.json", {
            "schema_version": 2,
            "phase": phase["name"],
            "stage": args.stage,
            "diagnostic_only": True,
            "selection_mutation": False,
            "dataset": {"name": "nq_open", "split": "validation", **stage},
            "dpr_positive_source": dpr_source,
            "preflight": join_preflight,
        })
        print(f"Completed DPR question-join preflight: {run_dir}")
        print(json.dumps(join_preflight, sort_keys=True))
        return run_dir
    if not join_preflight["can_reach_mapping_gate"]:
        metadata["status"] = "blocked_preflight"
        metadata["decision"] = "question_join_below_mapping_gate"
        _write_json(run_dir / "run_metadata.json", metadata)
        _write_json(run_dir / "preflight.json", {
            "schema_version": 2,
            "phase": phase["name"],
            "stage": args.stage,
            "diagnostic_only": True,
            "selection_mutation": False,
            "dataset": {"name": "nq_open", "split": "validation", **stage},
            "dpr_positive_source": dpr_source,
            "preflight": join_preflight,
            "decision": "do_not_start_wiki_dpr",
        })
        raise DprPositiveAlignmentConfigError(
            "target-question/DPR join cannot reach the mapping gate; "
            f"joined {join_preflight['joined_count']}/{join_preflight['sample_count']}, "
            f"required {join_preflight['required_join_count']}; "
            "run --preflight-only after fixing the dataset/source identity"
        )
    _write_json(run_dir / "run_metadata.json", metadata)

    started = time.perf_counter()
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager(
        "wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": 64}
    )
    corpus_manager.build(questions)
    alignment = align_dpr_positives_to_wiki(
        getattr(corpus_manager, "_dataset"),
        joins,
        progress_every=int(phase["alignment"].get("progress_every", 1_000_000)),
    )
    _write_json(run_dir / "alignment_index.json", {
        "schema_version": 2,
        "identity_rules": phase["alignment"]["identity_rules"],
        "matches": {
            key: {
                **value,
                "verified": sorted(value.get("verified", set())),
            }
            for key, value in alignment.items()
        },
    })

    answer_scorer = make_answer_scorer(backend="dpr")
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        question_id = str(item["id"])
        question = str(item["question"])
        query_embedding = encoder.encode_queries([question])[0]
        retrieved_rows, embeddings = _retrieved_rows(corpus_manager, query_embedding, 50)
        if len(retrieved_rows) != 50 or embeddings.shape[0] != 50:
            raise DprPositiveAlignmentConfigError(f"{question_id}: Top-50 retrieval length mismatch")
        texts = [row["text"] for row in retrieved_rows]
        relevance_scores = answer_scorer.score_passages(question, texts)
        selected_local = select_passages(
            query_embedding, embeddings, K=5, method="qore", num_reads=100, lam=2.0,
            gamma=1.0, delta=0.0, complementarity_method=None, qore_prefilter_size=None,
            direct_solve_max_n=20, lambda_mmr=0.7, saturation_alpha=1.0,
            lambda_submodular=0.5, dpp_quality_scale=2.0, dpp_jitter=1.0e-8,
            answer_scorer=answer_scorer, passage_texts=texts, question=question,
            seed=42, relevance_scores=relevance_scores,
        )
        if len(selected_local) != 5:
            raise DprPositiveAlignmentConfigError(f"{question_id}: QORE Top-5 length mismatch")
        samples.append(_sample_row(
            question_id,
            question_verified[question_id],
            joins[question_id][0],
            alignment.get(question_id, {}),
            retrieved_rows,
            selected_local,
        ))
        if index % 5 == 0 or index == len(questions):
            print(f"  DPR positive alignment {args.stage}: {index}/{len(questions)}")

    result = {
        "schema_version": 2,
        "phase": phase["name"],
        "stage": args.stage,
        "diagnostic_only": True,
        "selection_mutation": False,
        "samples": samples,
    }
    validate_dpr_compact_result(result)
    _write_json(run_dir / "result.json", result)
    summary = summarize_dpr_positive_alignment(
        samples, min_mapping_rate=float(phase["gate"]["minimum_mapping_rate"])
    )
    _write_json(run_dir / "summary.json", summary)
    metadata.update({
        "status": "completed",
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "summary": {"path": str(run_dir / "summary.json"), "status": summary["decision"]["status"]},
    })
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed DPR positive gold alignment audit: {run_dir}")
    print(f"Decision: {summary['decision']['status']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (DprPositiveAlignmentConfigError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
