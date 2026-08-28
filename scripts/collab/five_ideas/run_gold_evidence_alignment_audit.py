#!/usr/bin/env python3
"""Audit strict NQ gold evidence through Wiki-DPR Top-50 and QORE Top-5.

This is a collaborator-only, observation-only diagnostic.  It performs no
generation and does not change the selector.  The full Wiki-DPR scan is used
only to construct a private ID alignment index; the published result contains
scalar IDs/counts and no question, passage, answer, or model output text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

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
from applications.rag.gold_evidence_alignment import (
    GoldAlignmentError,
    GoldEvidence,
    align_corpus,
    extract_gold_evidence,
    passage_identity,
    retrieved_matches,
    summarize_alignment,
    validate_compact_result,
)
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages


class GoldAlignmentConfigError(RuntimeError):
    """Raised when the frozen gold-evidence audit contract is malformed."""


MODEL_INDEPENDENT = True
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
    raise GoldAlignmentConfigError("cannot locate project root")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_config(path: Path, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise GoldAlignmentConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, Mapping) or phase.get("name") != "gold_evidence_alignment_audit":
        raise GoldAlignmentConfigError("unexpected gold-evidence alignment configuration")
    if phase.get("schema_version") != 1 or phase.get("diagnostic_only") is not True:
        raise GoldAlignmentConfigError("audit must be schema 1 and diagnostic_only")
    if phase.get("selection_mutation") is not False:
        raise GoldAlignmentConfigError("selection_mutation must remain false")
    if stage not in phase.get("dataset", {}):
        raise GoldAlignmentConfigError(f"dataset stage is not configured: {stage}")
    dataset = phase["dataset"][stage]
    expected = {"name": "nq_open", "split": "validation", "sample_offset": 200,
                "max_samples": 50, "fresh_slice": True}
    if dict(dataset) != expected:
        raise GoldAlignmentConfigError(f"dataset.{stage} slice is not frozen")
    retrieval = phase.get("retrieval")
    if retrieval != {"corpus_mode": "wiki_dpr", "wiki_dpr_config": "psgs_w100.nq.compressed",
                     "nprobe": 64, "top_k": 50}:
        raise GoldAlignmentConfigError("retrieval contract is not frozen")
    if dict(phase.get("selection", {})) != EXPECTED_SELECTION:
        raise GoldAlignmentConfigError("selection contract is not frozen")
    alignment = phase.get("alignment")
    if not isinstance(alignment, Mapping) or alignment.get("scan_mode") != "full_corpus":
        raise GoldAlignmentConfigError("full_corpus alignment scan is required")
    outputs = phase.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("compact_only") is not True:
        raise GoldAlignmentConfigError("outputs.compact_only must be true")
    return dict(phase), dict(dataset)


def _parse_json_records(path: Path) -> list[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        if path.name.endswith((".jsonl", ".jsonl.gz")):
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("examples", "data", "records"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, Mapping)]
    raise GoldAlignmentConfigError(f"NQ annotation file is not a list/JSONL: {path}")


def _load_nq_annotations(root: Path, phase: Mapping[str, Any], override: str | None) -> tuple[list[Mapping[str, Any]], str]:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(override))))
    env_path = os.environ.get("NQ_ANNOTATIONS_PATH")
    if env_path:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(env_path))))
    for raw in phase.get("alignment", {}).get("local_annotation_candidates", []):
        candidate = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    for candidate in candidates:
        if candidate.is_file():
            return _parse_json_records(candidate), f"local:{candidate.resolve()}"

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise GoldAlignmentConfigError("datasets is required for NQ annotation loading") from exc
    names = list(phase.get("alignment", {}).get("annotation_dataset_names", []))
    errors: list[str] = []
    for name in names:
        try:
            dataset = load_dataset(name, split="validation", trust_remote_code=True)
            return [dict(item) for item in dataset], f"hf:{name}"
        except Exception as exc:  # datasets builders differ across environments
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
    raise GoldAlignmentConfigError(
        "cannot load original NQ annotations. Set NQ_ANNOTATIONS_PATH or pass "
        "--nq-annotations to a JSON/JSONL export. Attempts: " + "; ".join(errors)
    )


def _question_key(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _raw_question_map(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = _question_key(record.get("question", {}).get("text") if isinstance(record.get("question"), Mapping) else record.get("question"))
        if key:
            result.setdefault(key, []).append(record)
    return result


def _value(values: Any, index: int, default: Any = None) -> Any:
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _retrieved_rows(manager: Any, query_embedding: np.ndarray, top_k: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise GoldAlignmentConfigError("wiki_dpr manager does not expose the indexed dataset")
    scores, retrieved = dataset.get_nearest_examples("embeddings", np.asarray(query_embedding, dtype=np.float32), k=top_k)
    texts = retrieved.get("text", [])
    titles = retrieved.get("title", [])
    ids = retrieved.get("id", [])
    embeddings = np.asarray(retrieved.get("embeddings", []), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for index in range(len(texts)):
        rows.append({"id": _value(ids, index), "title": str(_value(titles, index, "") or ""),
                     "text": str(_value(texts, index, "") or ""), "score": float(_value(scores, index, 0.0))})
    return rows, embeddings


def _alignment_cache_key(evidence_by_question: Mapping[str, GoldEvidence], phase: Mapping[str, Any]) -> str:
    payload = {
        "algorithm": "gold_evidence_alignment_v1",
        "corpus": phase["retrieval"],
        "evidence": {
            str(key): {"title": value.document_title, "strict": value.strict_answers,
                       "support": value.support_answers}
            for key, value in sorted(evidence_by_question.items())
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_or_scan_alignment(
    root: Path, phase: Mapping[str, Any], evidence_by_question: Mapping[str, GoldEvidence],
    manager: Any,
) -> tuple[dict[str, dict[str, Any]], str]:
    cache_root = root / str(phase["outputs"]["root"])
    cache_path = cache_root / "alignment_index.json"
    cache_key = _alignment_cache_key(evidence_by_question, phase)
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("cache_key") == cache_key:
                cached = payload.get("matches")
                if isinstance(cached, Mapping) and set(cached) == set(evidence_by_question):
                    return {str(key): {"strict": set(value.get("strict", [])),
                                       "support": set(value.get("support", [])),
                                       "title_seen": int(value.get("title_seen", 0))}
                            for key, value in cached.items()}, "cache"
        except (OSError, ValueError, TypeError):
            pass
    print("  Scanning Wiki-DPR metadata for strict gold evidence (one-time index)...")
    matches = align_corpus(getattr(manager, "_dataset"), evidence_by_question,
                           progress_every=int(phase["alignment"].get("progress_every", 1_000_000)))
    serializable = {key: {"strict": sorted(value["strict"]), "support": sorted(value["support"]),
                          "title_seen": int(value.get("title_seen", 0))}
                    for key, value in matches.items()}
    _write_json(cache_path, {"schema_version": 1, "cache_key": cache_key, "matches": serializable})
    return matches, "full_corpus_scan"


def _sample_row(
    question_id: str, evidence: GoldEvidence, corpus_matches: Mapping[str, Any],
    retrieved_rows: Sequence[Mapping[str, Any]], selected_local: Sequence[int],
) -> dict[str, Any]:
    strict_gold = set(corpus_matches.get("strict", set()))
    support_gold = set(corpus_matches.get("support", set()))
    strict_retrieved, support_retrieved = retrieved_matches(retrieved_rows, evidence)
    selected_rows = [retrieved_rows[int(index)] for index in selected_local]
    strict_selected, support_selected = retrieved_matches(selected_rows, evidence)
    top50_strict = strict_retrieved & strict_gold if strict_gold else set()
    top5_strict = strict_selected & strict_gold if strict_gold else set()
    top50_support = support_retrieved & support_gold if support_gold else set()
    top5_support = support_selected & support_gold if support_gold else set()
    title_seen = int(corpus_matches.get("title_seen", 0))
    strict_count = len(strict_gold)
    status = "mapped" if strict_count else ("support_only" if support_gold else ("no_strict_match" if title_seen else "no_matching_title"))
    return {
        "question_id": str(question_id),
        "mapping_status": status,
        "mapping_source": evidence.source,
        "has_short_span": bool(evidence.has_short_span),
        "gold_passage_count": strict_count,
        "support_passage_count": len(support_gold),
        "top50_gold_count": len(top50_strict),
        "top5_gold_count": len(top5_strict),
        "top50_support_count": len(top50_support),
        "top5_support_count": len(top5_support),
        "retrieval_hit": bool(top50_strict),
        "selected_hit": bool(top5_strict),
        "support_retrieval_hit": bool(top50_support),
        "support_selected_hit": bool(top5_support),
        "gold_recall_at_top50": (len(top50_strict) / strict_count) if strict_count else None,
        "gold_recall_at_top5": (len(top5_strict) / strict_count) if strict_count else None,
        "conditional_selection_recall": (len(top5_strict) / len(top50_strict)) if top50_strict else None,
        "retrieval_top_k": len(retrieved_rows),
        "selection_k": len(selected_rows),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/gold_evidence_alignment_audit.yaml")
    parser.add_argument("--stage", choices=("screen",), default="screen")
    parser.add_argument("--nq-annotations", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    root = _project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config_path = config_path.resolve()
    phase, stage = _load_config(config_path, args.stage)
    if args.validate_only:
        print(json.dumps({"status": "valid", "phase": phase["name"], "stage": args.stage,
                          "slice": stage, "selection_mutation": False,
                          "scan_mode": phase["alignment"]["scan_mode"],
                          "model_loaded": False, "wiki_dpr_started": False}, sort_keys=True))
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

    annotation_records, annotation_source = _load_nq_annotations(root, phase, args.nq_annotations)
    sample_end = int(stage["sample_offset"]) + int(stage["max_samples"])
    questions = load_dataset_for_rag("nq_open", "validation", sample_end)
    questions = questions[int(stage["sample_offset"]):sample_end]
    if len(questions) != int(stage["max_samples"]):
        raise GoldAlignmentConfigError("nq_open fresh slice length mismatch")
    raw_map = _raw_question_map(annotation_records)
    evidence_by_question: dict[str, GoldEvidence] = {}
    for offset, item in enumerate(questions, start=int(stage["sample_offset"])):
        question_id = str(item["id"])
        key = _question_key(item.get("question"))
        raw_candidates = raw_map.get(key, [])
        raw = raw_candidates[0] if raw_candidates else {}
        evidence_by_question[question_id] = extract_gold_evidence(raw, item.get("answers", []))

    metadata = {
        "schema_version": 1, "phase": phase["name"], "stage": args.stage,
        "status": "running", "diagnostic_only": True, "selection_mutation": False,
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "status": _git(root, "status", "--short", "--branch")},
        "python": {"executable": sys.executable, "version": sys.version},
        "dataset": {"name": "nq_open", "split": "validation", **stage,
                     "annotation_source": annotation_source, "annotation_record_count": len(annotation_records)},
        "retrieval": phase["retrieval"], "selection": phase["selection"],
        "outputs": {"compact_only": True, "gold_alignment_cache": "alignment_index.json"},
        "plugins": {"allowlist": ["nq_annotation_extractor", "wiki_dpr_gold_alignment", "retrieval_top50_observer", "qore_top5_observer"],
                    "diagnostic_outputs_used_for_selection": False},
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    started = time.perf_counter()
    encoder = make_encoder("dpr")
    corpus_manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64})
    corpus_manager.build(questions)
    alignment, alignment_source = _load_or_scan_alignment(root, phase, evidence_by_question, corpus_manager)
    answer_scorer = make_answer_scorer(backend="dpr")
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        question_id = str(item["id"])
        question = str(item["question"])
        evidence = evidence_by_question[question_id]
        query_embedding = encoder.encode_queries([question])[0]
        retrieved_rows, embeddings = _retrieved_rows(corpus_manager, query_embedding, 50)
        if len(retrieved_rows) != 50 or embeddings.shape[0] != 50:
            raise GoldAlignmentConfigError(f"{question_id}: Top-50 retrieval length mismatch")
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
            raise GoldAlignmentConfigError(f"{question_id}: QORE Top-5 length mismatch")
        samples.append(_sample_row(question_id, evidence, alignment.get(question_id, {}),
                                   retrieved_rows, selected_local))
        if index % 5 == 0 or index == len(questions):
            print(f"  gold alignment {args.stage}: {index}/{len(questions)}")

    result = {"schema_version": 1, "phase": phase["name"], "stage": args.stage,
              "diagnostic_only": True, "selection_mutation": False,
              "alignment_source": alignment_source, "samples": samples}
    validate_compact_result(result)
    _write_json(run_dir / "result.json", result)
    summary = summarize_alignment(samples, min_mapping_rate=float(phase["gate"]["minimum_mapping_rate"]))
    _write_json(run_dir / "summary.json", summary)
    metadata.update({"status": "completed", "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
                     "summary": {"path": str(run_dir / "summary.json"), "status": summary["decision"]["status"]}})
    _write_json(run_dir / "run_metadata.json", metadata)
    print(f"Completed gold-evidence alignment audit: {run_dir}")
    print(f"Decision: {summary['decision']['status']}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parse_args(argv))
        return 0
    except (GoldAlignmentConfigError, GoldAlignmentError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
