#!/usr/bin/env python3
"""Check all official NQ gold titles against a local Wiki-DPR title column.

Read-only and offline: no download, model, retrieval, selector, or gold claim.
Use ``--title`` only when a single title is being inspected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_CONFIG = "psgs_w100.nq.compressed"
SOURCE_CANDIDATES = (
    "data/dpr/nq-test_gold_info.json", "data/dpr/nq-test_gold_info.json.gz",
    "data/dpr/nq_test_gold_info.json", "data/dpr/nq_test_gold_info.json.gz",
    "data/nq/nq-test_gold_info.json", "data/nq/nq-test_gold_info.json.gz",
    "data/nq/nq_test_gold_info.json", "data/nq/nq_test_gold_info.json.gz",
    "data/gold_passages_info/nq-test_gold_info.json",
    "data/gold_passages_info/nq-test_gold_info.json.gz",
    "data/gold_passages_info/nq_test_gold_info.json",
    "data/gold_passages_info/nq_test_gold_info.json.gz",
)


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _project_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise RuntimeError("cannot locate project root")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_gold_source(root: Path) -> tuple[Path | None, list[str]]:
    candidates: list[Path] = []
    for variable in ("DPR_NQ_GOLD_INFO_PATH", "DPR_NQ_POSITIVES_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(os.path.expanduser(os.path.expandvars(value))))
    candidates.extend(root / relative for relative in SOURCE_CANDIDATES)
    attempted: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        label = _relative(root, resolved)
        if label in seen:
            continue
        seen.add(label)
        attempted.append(label)
        if resolved.is_file():
            return resolved, attempted
    return None, attempted


def _read_gold_rows(path: Path) -> list[Mapping[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise RuntimeError("official DPR gold-info must be a JSON list or object with data[]")
    return list(rows)


def _gold_title_records(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, set[str]], dict[str, int]]:
    title_to_questions: dict[str, set[str]] = {}
    duplicate_pairs = 0
    records = nonempty_titles = nonempty_contexts = 0
    for row in rows:
        records += 1
        question = re.sub(r"\s+", " ", str(row.get("question") or "")).strip()
        title_key = normalize_title(row.get("title"))
        if title_key:
            nonempty_titles += 1
            if question in title_to_questions.setdefault(title_key, set()):
                duplicate_pairs += 1
            title_to_questions[title_key].add(question)
        context = row.get("context")
        if isinstance(context, str) and context.strip():
            nonempty_contexts += 1
    return title_to_questions, {
        "records": records,
        "nonempty_title_records": nonempty_titles,
        "unique_normalized_titles": len(title_to_questions),
        "unique_questions_with_nonempty_title": len({q for values in title_to_questions.values() for q in values if q}),
        "nonempty_context_records": nonempty_contexts,
        "duplicate_question_title_pairs": duplicate_pairs,
    }


def _load_wiki_dataset(config: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("missing datasets package") from exc
    dataset = load_dataset("facebook/wiki_dpr", config, split="train", cache_dir=cache_dir, trust_remote_code=True)
    if "title" not in dataset.column_names:
        raise RuntimeError(f"Wiki-DPR dataset has no title column: {dataset.column_names}")
    columns = [name for name in ("id", "title", "text") if name in dataset.column_names]
    return dataset.select_columns(columns)


def _scan_titles(dataset: Any, target_titles: set[str], batch_size: int) -> tuple[dict[str, int], int]:
    counts = {title: 0 for title in target_titles}
    scanned = 0
    for batch in dataset.iter(batch_size=batch_size):
        for raw_title in batch.get("title", ()):
            scanned += 1
            key = normalize_title(raw_title)
            if key in counts:
                counts[key] += 1
    return counts, scanned


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default=None, help="optional single title; default checks all official gold titles")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    root = _project_root()
    source, attempted = _resolve_gold_source(root)
    if source is None and args.title is None:
        print(json.dumps({"status": "blocked_source_unavailable", "attempted_local_paths": attempted}, ensure_ascii=False, indent=2))
        return 4

    if args.title is not None:
        title_to_questions = {normalize_title(args.title): {"single_title_query"}}
        source_accounting = None
    else:
        assert source is not None
        title_to_questions, source_accounting = _gold_title_records(_read_gold_rows(source))
        if not title_to_questions:
            print(json.dumps({"status": "blocked_no_nonempty_gold_titles", "source": _relative(root, source)}, ensure_ascii=False, indent=2))
            return 5

    try:
        dataset = _load_wiki_dataset(args.config, args.cache_dir)
        match_counts, rows_scanned = _scan_titles(dataset, set(title_to_questions), args.batch_size)
    except Exception as exc:  # noqa: BLE001 - compact diagnostic report
        print(json.dumps({"status": "error", "error": str(exc), "offline": True}, ensure_ascii=False, indent=2))
        return 3

    matched_titles = {key for key, count in match_counts.items() if count > 0}
    missing_titles = set(match_counts) - matched_titles
    matched_question_set = {q for key in matched_titles for q in title_to_questions[key] if q}
    all_question_set = {q for values in title_to_questions.values() for q in values if q}
    matched_questions = len(matched_question_set)
    total_questions = len(all_question_set)
    report: dict[str, Any] = {
        "status": "ok", "diagnostic_only": True, "data_download": False,
        "model_loaded": False, "retrieval_started": False, "selector_started": False,
        "dataset": "facebook/wiki_dpr", "config": args.config,
        "gold_source": ({"path": _relative(root, source), "byte_count": source.stat().st_size, "sha256": _sha256(source)} if source is not None else None),
        "source_accounting": source_accounting,
        "title_normalization": "NFKC + whitespace collapse + casefold",
        "wiki_dpr_rows_scanned": rows_scanned,
        "unique_gold_titles_checked": len(title_to_questions),
        "gold_questions_checked": total_questions,
        "matched_title_count": len(matched_titles), "missing_title_count": len(missing_titles),
        "title_match_rate": len(matched_titles) / len(title_to_questions),
        "questions_with_title_match": matched_questions,
        "question_title_match_rate": matched_questions / total_questions if total_questions else None,
        "matched_title_multiplicity": {
            "min_rows": min((match_counts[key] for key in matched_titles), default=None),
            "median_rows": sorted(match_counts[key] for key in matched_titles)[len(matched_titles) // 2] if matched_titles else None,
            "max_rows": max((match_counts[key] for key in matched_titles), default=None),
        },
        "sample_matched_titles": sorted(matched_titles)[:10],
        "sample_missing_titles": sorted(missing_titles)[:10],
        "interpretation": "Title presence only; a match does not prove the official NQ context is the same Wiki-DPR passage or establish retrieval/selector recall.",
    }
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "exchange/five_ideas/dpr_title_presence_check" / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / "exchange/five_ideas/dpr_title_presence_check" / f"{timestamp}_{suffix}"
        suffix += 1
    _write_json(run_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_path={_relative(root, run_dir / 'report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
