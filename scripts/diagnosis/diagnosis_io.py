#!/usr/bin/env python3
"""Shared result-loading for the Phase 1 diagnosis scripts.

All diagnoses consume the JSON written by `scripts/rag/eval_rag_refactored.py`,
whose shape is:

    {"config": {...}, "corpus_metadata": {...},
     "metrics": {"mean_f1": ..., ...},          # dataset-level aggregate
     "samples": [{"question_id": ..., "recall": ..., "f1": ...,
                  "question": ..., "selected_passages": [...]}, ...]}

Two things bit us before and are guarded here:

1. The per-sample list lives under "samples", not "results", and each sample
   is FLAT — there is no nested "metrics" sub-dict. Reading the wrong key
   yielded zero records and the scripts still exited 0 with a written report,
   so a missing field looked like a negative finding.
2. "question" / "selected_passages" are only present when the eval was run
   with --dump_passages. Without them the passage-level diagnoses cannot say
   anything, so we raise instead of reporting a false conclusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class DiagnosisInputError(RuntimeError):
    """Raised when a result file cannot support the requested diagnosis."""


def load_samples(results_file: Path, require: Iterable[str] = ()) -> list[dict]:
    """Load per-sample records, verifying the fields a diagnosis needs.

    Args:
        results_file: a result.json from eval_rag_refactored.
        require: field names that must be present on the samples, e.g.
            ("selected_passages",) or ("question", "f1").

    Raises:
        DiagnosisInputError: file unreadable, no samples, or a required field
            is absent — with the concrete re-run command needed to produce it.
    """
    results_file = Path(results_file)
    if not results_file.exists():
        raise DiagnosisInputError(f"result file not found: {results_file}")

    try:
        with open(results_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DiagnosisInputError(f"{results_file} is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise DiagnosisInputError(
            f"{results_file}: expected a JSON object, got {type(data).__name__}"
        )

    samples = data.get("samples")
    if samples is None:
        # Tolerate the legacy key so old artifacts stay analysable.
        samples = data.get("results")
        if samples is None:
            raise DiagnosisInputError(
                f"{results_file}: no 'samples' key. Found: {sorted(data)}. "
                "Was this written by scripts/rag/eval_rag_refactored.py?"
            )

    if not samples:
        raise DiagnosisInputError(f"{results_file}: 'samples' is empty")

    missing = [k for k in require if not any(k in s for s in samples)]
    if missing:
        raise DiagnosisInputError(
            f"{results_file}: samples lack required field(s) {missing}.\n"
            f"  Sample keys present: {sorted(samples[0])}\n"
            "  'question'/'gold_answers'/'selected_passages' require "
            "--dump_passages on the eval run.\n"
            "  'f1'/'em' require generation, i.e. do NOT pass --skip_generation."
        )
    return samples


def passage_texts(sample: dict) -> list[str]:
    """Selected passage texts for one sample ([] if none were dumped)."""
    return [p.get("text", "") for p in sample.get("selected_passages") or []]


def describe_coverage(samples: list[dict], field: str) -> str:
    """How many samples actually carry `field` — for report provenance."""
    n = sum(1 for s in samples if s.get(field) is not None)
    return f"{n}/{len(samples)}"
