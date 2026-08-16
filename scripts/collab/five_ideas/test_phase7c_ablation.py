"""Configuration contracts for the Phase 7C collaborator ablation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.collab.five_ideas.run_phase7c_ablation import (
    CalibrationError,
    _load_config,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/experiments/phase7c_answer_identity_ablation.yaml"


def test_committed_config_is_new_slice_and_narrow_mechanism_ablation():
    phase = _load_config(CONFIG)
    assert phase["sample_offset"] >= 100
    assert phase["K_values"] == [5]
    assert phase["seeds"] == [42]
    assert {item["name"] for item in phase["selection_variants"]} == {
        "qore_answer", "topk_answer", "qore_agreement", "qore_corroboration"
    }
    assert [item["name"] for item in phase["generators"]] == ["reference"]


def test_non_held_out_offset_is_rejected():
    text = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.yaml"
        path.write_text(text.replace("sample_offset: 100", "sample_offset: 50"), encoding="utf-8")
        with pytest.raises(CalibrationError, match="Phases 7A and 7B"):
            _load_config(path)


def test_topk_variant_cannot_enable_qore_enhancers():
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace(
        "enhancers: []\n      enhancer_configs: {}",
        "enhancers: [baseline]\n      enhancer_configs: {baseline: {gamma: 1.0}}",
        1,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(CalibrationError, match="only qore variants"):
            _load_config(path)
