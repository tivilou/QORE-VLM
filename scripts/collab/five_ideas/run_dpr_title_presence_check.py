#!/usr/bin/env python3
"""Check whether one article title exists in a locally cached Wiki-DPR corpus.

This is a read-only, offline diagnostic.  It does not load a model, build a
retrieval index, or claim that a title match is a passage-level gold match.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_TITLE = "List of Nobel laureates in Physics"
DEFAULT_CONFIG = "psgs_w100.nq.compressed"


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _project_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    return script.parents[3]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    # Keep the diagnostic local-only even when the collaborator has no network
    # restrictions configured in the shell.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(json.dumps({"status": "error", "error": "missing datasets package"}))
        return 2

    try:
        dataset = load_dataset(
            "facebook/wiki_dpr",
            args.config,
            split="train",
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        columns = [name for name in ("id", "title", "text") if name in dataset.column_names]
        if "title" not in columns:
            raise RuntimeError(f"Wiki-DPR dataset has no title column: {dataset.column_names}")
        dataset = dataset.select_columns(columns)
    except Exception as exc:  # noqa: BLE001 - compact diagnostic report
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 3

    target = normalize_title(args.title)
    matches: list[dict[str, Any]] = []
    exact_count = 0
    scanned = 0
    for batch in dataset.iter(batch_size=args.batch_size):
        titles = batch.get("title", ())
        ids = batch.get("id", ())
        texts = batch.get("text", ())
        for offset, raw_title in enumerate(titles):
            scanned += 1
            if normalize_title(raw_title) != target:
                continue
            exact_count += 1
            if len(matches) < 5:
                raw_text = texts[offset] if offset < len(texts) else ""
                matches.append(
                    {
                        "id": _jsonable(ids[offset]) if offset < len(ids) else None,
                        "title": str(raw_title),
                        "text_preview": str(raw_text or "")[:240],
                    }
                )

    report = {
        "status": "ok",
        "dataset": "facebook/wiki_dpr",
        "config": args.config,
        "title_queried": args.title,
        "title_normalization": "NFKC + whitespace collapse + casefold",
        "rows_scanned": scanned,
        "normalized_title_match_count": exact_count,
        "article_title_present": exact_count > 0,
        "sample_matches": matches,
        "interpretation": (
            "Title presence only; this does not prove that an official NQ gold "
            "context is the same Wiki-DPR passage."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
