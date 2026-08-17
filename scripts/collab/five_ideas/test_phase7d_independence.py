"""Configuration contracts for the Phase 7D collaborator diagnostic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.collab.five_ideas.run_phase7d_independence import (
    DiagnosticError,
    _load_config,
    parse_args,
    run,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/experiments/phase7d_independence_residual.yaml"


def test_committed_config_is_diagnostic_only_and_held_out():
    phase = _load_config(CONFIG)
    assert phase["diagnostic_only"] is True
    assert phase["sample_offset"] >= 150
    assert phase["K_values"] == [5]
    assert phase["seeds"] == [42]
    assert phase["diagnostic_plugins"] == ["answer_evidence_observer", "independence_residual"]
    assert list(phase["residualization"]["variants"]) == [
        "confidence_only", "lexical_only", "embedding_only", "all_nuisances"
    ]
    assert [item["name"] for item in phase["generators"]] == ["reference"]


def test_old_slice_is_rejected():
    text = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.yaml"
        path.write_text(text.replace("sample_offset: 150", "sample_offset: 100"), encoding="utf-8")
        with pytest.raises(DiagnosticError, match="new held-out slice"):
            _load_config(path)


def test_residual_plugin_order_is_explicit():
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace(
        "- answer_evidence_observer\n    - independence_residual",
        "- independence_residual\n    - answer_evidence_observer",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(DiagnosticError, match="plugin order"):
            _load_config(path)


def test_smoke_manifest_freezes_diagnostic_provenance_without_dataset_access(tmp_path: Path):
    path = tmp_path / "manifest.json"
    assert run(parse_args(["--smoke-manifest", str(path)])) == path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "smoke_validated"
    assert manifest["diagnostic_only"] is True
    assert manifest["selection_mutation"] is False
    assert manifest["dataset"]["sample_offset"] == 150
    assert manifest["design"]["gold_labels_used_for_residual_fit"] is False
    assert manifest["design"]["diagnostic_outputs_used_for_selection"] is False
    assert len(manifest["design"]["plugin_tree_hash"]) == 64
