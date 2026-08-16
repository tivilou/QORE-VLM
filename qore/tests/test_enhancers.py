"""Contract and compatibility tests for the enhancer plugin framework."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from qore.enhancers import (
    EnhancerPipeline,
    QUBOEnhancer,
    create_pipeline,
    create_pipeline_from_config,
    list_enhancers,
    load_enhancer_config,
)
from applications.rag.signals_rag import passage_complementarity_dpr


class LengthScorer:
    def score_passages(self, question: str, passages: list[str]) -> np.ndarray:
        return np.asarray([len(text.split()) / 10.0 for text in passages])


def test_builtin_plugins_are_discovered_automatically():
    assert {"baseline", "idea4", "idea6", "idea7", "answer_corroboration"} <= set(list_enhancers())


def test_baseline_preserves_original_objective():
    b = np.array([[0.0, 0.4], [0.4, 0.0]])
    pipeline = create_pipeline(["baseline"], {"baseline": {"gamma": 0.5}})

    w = pipeline.enhance(np.array([0.8, 0.3]), b, {})

    np.testing.assert_allclose(w, 0.5 * b)


def test_idea6_preserves_original_objective():
    a = np.array([0.8, 0.3])
    b = np.array([[0.0, 0.4], [0.4, 0.0]])
    passages = ["alpha answer", "beta supporting evidence"]
    scorer = LengthScorer()
    context = {
        "question": "question",
        "passages": passages,
        "answer_scorer": scorer,
    }
    pipeline = create_pipeline(
        ["idea6"],
        {"idea6": {"gamma": 0.5, "delta": 0.1, "method": "dpr"}},
    )

    w = pipeline.enhance(a, b, context)
    complementarity = passage_complementarity_dpr(
        "question", passages, scorer
    )

    np.testing.assert_allclose(w, 0.5 * b - 0.1 * complementarity)


def test_root_then_additive_plugin_composes_without_overwrite():
    a = np.array([0.8, 0.3])
    b = np.array([[0.0, 0.4], [0.4, 0.0]])
    pipeline = create_pipeline(
        ["baseline", "idea4"],
        {
            "baseline": {"gamma": 0.5},
            "idea4": {"alpha": 0.1},
        },
        strict_composition=True,
    )
    context = {
        "passages_meta": [
            {"doc_id": "doc", "rank": 1},
            {"doc_id": "doc", "rank": 2},
        ]
    }

    w, trace = pipeline.enhance_with_diagnostics(a, b, context)

    np.testing.assert_allclose(w, np.array([[0.0, 0.1], [0.1, 0.0]]))
    assert [entry["mode"] for entry in trace] == ["replace", "add"]
    assert not any(entry["overwrote_nonzero_input"] for entry in trace)


def test_answer_corroboration_adds_only_a_negative_pair_reward():
    a = np.array([0.8, 0.3, 0.2])
    b = np.array([
        [0.0, 0.4, 0.2],
        [0.4, 0.0, 0.1],
        [0.2, 0.1, 0.0],
    ])
    feature = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
    ])
    pipeline = create_pipeline(
        ["baseline", "answer_corroboration"],
        {
            "baseline": {"gamma": 1.0},
            "answer_corroboration": {
                "mode": "agreement", "strength": 0.25,
            },
        },
        strict_composition=True,
    )
    w = pipeline.enhance(
        a, b, {"answer_evidence_matrices": {"agreement": feature}}
    )
    assert w[0, 1] < b[0, 1]
    assert w[0, 2] == pytest.approx(b[0, 2])
    np.testing.assert_allclose(w, w.T)
    assert np.allclose(np.diag(w), 0.0)


def test_answer_corroboration_discounted_duplicate_support_is_distinct():
    b = np.ones((3, 3), dtype=float) - np.eye(3)
    matrices = {
        "corroboration": np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
            [1.0, 0.5, 0.0],
        ])
    }
    pipeline = create_pipeline(
        ["baseline", "answer_corroboration"],
        {
            "baseline": {},
            "answer_corroboration": {
                "mode": "corroboration", "strength": 0.25,
            },
        },
        strict_composition=True,
    )
    w = pipeline.enhance(np.ones(3), b, {"answer_evidence_matrices": matrices})
    assert w[0, 1] == pytest.approx(b[0, 1])
    assert w[0, 2] < w[1, 2] < b[1, 2]


def test_answer_corroboration_requires_precomputed_evidence_when_enabled():
    pipeline = create_pipeline(
        ["baseline", "answer_corroboration"],
        {"baseline": {}, "answer_corroboration": {"strength": 0.1}},
        strict_composition=True,
    )
    with pytest.raises(ValueError, match="answer_evidence_matrices"):
        pipeline.enhance(np.ones(2), np.zeros((2, 2)), {})


def test_zero_strength_answer_corroboration_is_a_null_adapter():
    b = np.array([[0.0, 0.4], [0.4, 0.0]])
    pipeline = create_pipeline(
        ["baseline", "answer_corroboration"],
        {"baseline": {"gamma": 0.5}, "answer_corroboration": {"strength": 0.0}},
        strict_composition=True,
    )
    w = pipeline.enhance(np.ones(2), b, {})
    np.testing.assert_allclose(w, 0.5 * b)


def test_strict_config_rejects_multiple_root_objectives():
    config = {
        "selection": {
            "enhancers": [
                {"name": "baseline", "config": {"gamma": 0.5}},
                {"name": "idea6", "config": {"delta": 0.1}},
            ]
        }
    }

    with pytest.raises(ValueError, match="Ambiguous enhancer composition"):
        create_pipeline_from_config(config)


def test_programmatic_legacy_composition_keeps_old_behavior():
    with pytest.warns(DeprecationWarning, match="Ambiguous enhancer composition"):
        pipeline = create_pipeline(
            ["baseline", "idea6"],
            {
                "baseline": {"gamma": 0.2},
                "idea6": {"gamma": 0.7, "delta": 0.0},
            },
        )
    b = np.array([[0.0, 0.4], [0.4, 0.0]])

    w, trace = pipeline.enhance_with_diagnostics(np.array([1.0, 0.0]), b, {})

    np.testing.assert_allclose(w, 0.7 * b)
    assert trace[-1]["overwrote_nonzero_input"] is True


def test_yaml_loader_accepts_existing_experiment_shape(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
selection:
  method: qore
  enhancers:
    - name: baseline
      config:
        gamma: 0.3
    - name: idea4
      config:
        alpha: 0.1
""".strip(),
        encoding="utf-8",
    )

    names, configs = load_enhancer_config(config_path)

    assert names == ["baseline", "idea4"]
    assert configs == {
        "baseline": {"gamma": 0.3},
        "idea4": {"alpha": 0.1},
    }
    assert create_pipeline_from_config(config_path).names() == names


def test_all_experiment_configs_build_strict_pipelines():
    project_root = Path(__file__).parents[2]
    config_paths = sorted((project_root / "configs" / "experiments").glob("*.yaml"))
    assert config_paths

    plugin_config_paths = []
    for config_path in config_paths:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        selection = document.get("selection", {})
        if isinstance(selection, dict) and selection.get("enhancers"):
            plugin_config_paths.append(config_path)

    assert plugin_config_paths
    for config_path in plugin_config_paths:
        pipeline = create_pipeline_from_config(config_path)
        assert pipeline.names(), config_path


def test_pipeline_rejects_invalid_plugin_output():
    class AsymmetricEnhancer(QUBOEnhancer):
        @property
        def name(self) -> str:
            return "asymmetric"

        def enhance(
            self,
            w: np.ndarray,
            a: np.ndarray,
            b: np.ndarray,
            context: dict[str, Any],
        ) -> np.ndarray:
            return np.array([[0.0, 1.0], [0.0, 0.0]])

    pipeline = EnhancerPipeline([AsymmetricEnhancer()])

    with pytest.raises(ValueError, match="non-symmetric"):
        pipeline.enhance(np.array([1.0, 0.0]), np.zeros((2, 2)), {})
