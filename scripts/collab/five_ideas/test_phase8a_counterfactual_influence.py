"""Contract tests for the Phase 8A collaborator-only diagnostic runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

try:
    from applications.rag.counterfactual_influence import (
        build_context_probes,
        validate_compact_payload,
    )
except ImportError:  # pragma: no cover - direct local module tests
    from counterfactual_influence import build_context_probes, validate_compact_payload
from run_phase8a_counterfactual_influence import (
    DiagnosticError,
    _load_config,
    _resolve_generator,
    parse_args,
    run,
)


_HERE = Path(__file__).resolve()
_REMOTE_ROOT = _HERE.parents[3]
if (_REMOTE_ROOT / "configs/experiments/phase8a_counterfactual_influence.yaml").is_file():
    ROOT = _REMOTE_ROOT
    CONFIG = ROOT / "configs/experiments/phase8a_counterfactual_influence.yaml"
else:
    ROOT = _HERE.parents[2]
    CONFIG = ROOT / ".codex-tmp" / "phase8a-plan" / "phase8a_counterfactual_influence.yaml"


def test_frozen_config_and_allowlisted_plugins():
    phase = _load_config(CONFIG)
    assert phase["diagnostic_only"] is True
    assert phase["selection_mutation"] is False
    assert phase["sample_offset"] == 200
    assert phase["max_samples"] == 50
    assert phase["diagnostic_plugins"] == [
        "full_context_probe", "balanced_subset_probe", "doubly_robust_influence"
    ]
    assert phase["randomization"]["max_generations_per_question"] == 18
    assert phase["estimator"]["covariates"] == [
        "answer_score", "retrieval_rank", "original_selection_position",
        "passage_token_count", "other_context_size",
        "other_context_answer_score_sum", "other_context_token_count",
        "other_context_embedding_redundancy",
    ]


def test_plugin_order_and_selection_mutation_are_rejected():
    text = CONFIG.read_text(encoding="utf-8")
    bad_order = text.replace(
        "    - full_context_probe\n    - balanced_subset_probe",
        "    - balanced_subset_probe\n    - full_context_probe",
        1,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.yaml"
        path.write_text(bad_order, encoding="utf-8")
        with pytest.raises(DiagnosticError, match="plugin order"):
            _load_config(path)
        bad_mutation = text.replace("selection_mutation: false", "selection_mutation: true", 1)
        path.write_text(bad_mutation, encoding="utf-8")
        with pytest.raises(DiagnosticError, match="selection_mutation"):
            _load_config(path)


def test_full_context_probe_preserves_baseline_order():
    selected = [37, 4, 19, 2, 31]
    probes = build_context_probes("q", selected, seed=8101)
    full = [probe for probe in probes if probe.anchor == "full"]
    assert len(full) == 1
    assert full[0].mask == (1, 1, 1, 1, 1)
    assert full[0].ordered_ranks == tuple(selected)
    assert full[0].propensity == 1.0
    assert all(probe.anchor != "full" or probe.pair_index is None for probe in probes)


def test_cli_override_resolves_path_without_changing_identity(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    phase = _load_config(CONFIG)
    spec = phase["generators"][0]
    resolved = _resolve_generator(
        spec,
        [{"name": "reference", "model_path": str(model_path)}],
        require_resolved=True,
    )
    assert resolved["resolution"] == "resolved"
    assert resolved["source"] == "cli_override"
    assert resolved["model_id"] == "NousResearch/Meta-Llama-3-8B-Instruct"
    assert resolved["revision"] == "53346005fb0ef11d3b6a83b12c895cca40156b6c"
    assert len(resolved["config_sha256"]) == 64


def test_smoke_manifest_is_compact_and_does_not_load_data_or_model(tmp_path: Path):
    path = tmp_path / "manifest.json"
    result = run(parse_args(["--config", str(CONFIG), "--smoke-manifest", str(path)]))
    assert result == path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "smoke_validated"
    assert manifest["runtime"] == {"data_access": False, "model_load": False}
    assert manifest["selection_mutation"] is False
    assert manifest["dataset"]["sample_offset"] == 200
    assert manifest["design"]["diagnostic_outputs_used_for_selection"] is False
    assert len(manifest["design"]["plugin_tree_hash"]) == 64
    validate_compact_payload(manifest, {"question", "passages", "gold_answers", "prediction", "raw_prompt"})


def test_validate_only_does_not_require_model_path(capsys):
    assert run(parse_args(["--config", str(CONFIG), "--validate-only"])) is None
    output = capsys.readouterr().out
    assert '"status": "valid"' in output
    assert "phase8a_counterfactual_influence" in output
