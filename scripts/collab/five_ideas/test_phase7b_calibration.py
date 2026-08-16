"""Configuration safety tests for the Phase 7B collaborator gate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.collab.five_ideas.run_phase7b_calibration import (
    CalibrationError,
    _load_config,
)


ROOT = Path(__file__).resolve().parents[3]


def test_committed_config_is_held_out_multi_k_multi_seed(monkeypatch):
    monkeypatch.setenv("PHASE7B_TRANSFER_MODEL", "/models/transfer")
    phase = _load_config(
        ROOT / "configs/experiments/phase7b_answer_identity_calibration.yaml"
    )
    assert phase["sample_offset"] >= 50
    assert len(phase["K_values"]) >= 2
    assert len(phase["seeds"]) >= 2
    assert {item["name"] for item in phase["selection_variants"]} == {
        "qore_answer", "topk_answer", "qore_dpr"
    }
    assert len(phase["generators"]) >= 2


def test_generator_overrides_remove_machine_path_assumptions(monkeypatch):
    monkeypatch.delenv("PHASE7B_TRANSFER_MODEL", raising=False)
    phase = _load_config(
        ROOT / "configs/experiments/phase7b_answer_identity_calibration.yaml",
        [
            {"name": "reference", "model_path": "/models/reference"},
            {"name": "transfer", "model_path": "/models/transfer"},
        ],
    )
    assert [item["name"] for item in phase["generators"]] == ["reference", "transfer"]


def test_unresolved_transfer_model_is_rejected(monkeypatch):
    monkeypatch.delenv("PHASE7B_TRANSFER_MODEL", raising=False)
    with pytest.raises(CalibrationError, match="resolved model_path"):
        _load_config(ROOT / "configs/experiments/phase7b_answer_identity_calibration.yaml")


def test_non_held_out_offset_is_rejected(monkeypatch):
    monkeypatch.setenv("PHASE7B_TRANSFER_MODEL", "/models/transfer")
    text = (ROOT / "configs/experiments/phase7b_answer_identity_calibration.yaml").read_text()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.yaml"
        path.write_text(text.replace("sample_offset: 50", "sample_offset: 0"), encoding="utf-8")
        with pytest.raises(CalibrationError, match="held out"):
            _load_config(path)
