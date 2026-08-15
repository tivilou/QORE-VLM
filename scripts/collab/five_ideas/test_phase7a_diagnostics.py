"""Configuration safety tests for the Phase 7A collaborator runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.collab.five_ideas.run_phase7a_diagnostics import DiagnosticError, _load_config


def test_committed_config_is_bounded_and_wiki_dpr():
    root = Path(__file__).resolve().parents[3]
    phase = _load_config(root / "configs/experiments/phase7a_answer_identity_diagnostics.yaml")
    assert phase["corpus_mode"] == "wiki_dpr"
    assert 50 <= phase["max_samples"] <= 100


def test_unbounded_sample_count_is_rejected():
    payload = """phase:
  schema_version: 1
  name: unsafe
  dataset: nq_open
  split: validation
  max_samples: 3610
  seed: 42
  corpus_mode: wiki_dpr
  wiki_dpr_config: compressed
  wiki_dpr_nprobe: 64
  top_k_retrieval: 50
  selection_K: 5
  selection_method: qore
  answer_scorer_backend: dpr
  answer_top_m: 3
  max_answer_tokens: 10
  high_conflict_threshold: 0.25
  generator: {enabled: false}
  outputs: {root: exchange/test}
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "unsafe.yaml"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(DiagnosticError, match="between 1 and 100"):
            _load_config(path)
